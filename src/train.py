"""Streaming training pipeline — loads data, engineers features, trains LightGBM."""

from src.config import DATA_CONFIG, MODEL_CONFIG, LIGHTGBM_CONFIG, MLFLOW_CONFIG
import logging
import os
import pathlib
import re
import sys
from typing import Tuple, Dict, Any

import joblib
import lightgbm as lgb
import mlflow
import polars as pl
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")


def load_data(config: Dict[str, Any]) -> pl.DataFrame:
    """Stream data from Hugging Face or load a local CSV."""
    max_samples = config.get("max_samples", 50000)

    if config.get("raw_path"):
        csv_path = ROOT / config["raw_path"]
        logger.info("LOAD_LOCAL_CSV: %s", csv_path)
        dataframe = pl.read_csv(csv_path)
        if dataframe.height > max_samples:
            dataframe = dataframe.head(max_samples)
            logger.info("TRUNCATED_TO_MAX_SAMPLES: %d", max_samples)
        logger.info("RAW_SHAPE: %d rows, %d cols", dataframe.height, dataframe.width)
        return dataframe

    dataset_name = config["dataset"]
    hf_token = os.environ.get("HF_TOKEN", None)

    logger.info("LOAD_HF_DATASET: %s (streaming=True, max_samples=%d)", dataset_name, max_samples)

    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
    except Exception as exc:
        logger.error("HF_DATASET_FAILED: %s - %s", dataset_name, exc)
        raise RuntimeError(f"Failed to load dataset '{dataset_name}'. Original error: {exc}")

    records = []
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        records.append(sample)
        if (idx + 1) % max(max_samples // 10, 1000) == 0:
            logger.info("STREAM_PROGRESS: %d/%d rows collected", idx + 1, max_samples)

    dataframe = pl.DataFrame(records)
    logger.info("STREAM_COMPLETE: %d rows, %d cols", dataframe.height, dataframe.width)
    return dataframe


def _clean_features(dataframe: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
    """Drop unused columns, normalize numerics, filter target to the binary set."""
    dataframe = dataframe.clone()
    drop_cols = [c for c in config["drop_columns"] if c in dataframe.columns]
    if drop_cols:
        dataframe = dataframe.drop(drop_cols)
        logger.info("DROPPED_COLUMNS: %s", drop_cols)

    for tc_col in ["TotalCharges", "Total Charges"]:
        if tc_col in dataframe.columns:
            if dataframe[tc_col].dtype == pl.Utf8:
                dataframe = dataframe.with_columns(
                    pl.when(pl.col(tc_col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(tc_col))
                    .alias(tc_col)
                )
                dataframe = dataframe.with_columns(pl.col(tc_col).cast(pl.Float64).fill_null(0.0))
            break

    for sc_col in ["SeniorCitizen", "Senior Citizen"]:
        if sc_col in dataframe.columns:
            dataframe = dataframe.with_columns(pl.col(sc_col).cast(pl.Utf8))
            break

    pos = config["positive_class"]
    neg = config["negative_class"]
    target_col = config["target_column"]

    dataframe = dataframe.filter(pl.col(target_col).is_in([pos, neg]))

    if target_col != "Churn":
        dataframe = dataframe.with_columns((pl.col(target_col) == pos).cast(pl.Int8).alias("Churn"))
        dataframe = dataframe.drop(target_col)
    else:
        dataframe = dataframe.with_columns(pl.col("Churn").cast(pl.Int8))

    return dataframe


def _encode_features(dataframe: pl.DataFrame) -> Tuple[pl.DataFrame, pl.Series]:
    """One-hot encode categoricals and split off the target column."""
    cat_cols = [c for c in dataframe.columns if dataframe[c].dtype == pl.Utf8]
    if cat_cols:
        dataframe = dataframe.to_dummies(columns=cat_cols)

    features_x = dataframe.drop("Churn")
    target_y = dataframe["Churn"]

    sanitized = {c: re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in features_x.columns}
    features_x = features_x.rename(sanitized)

    return features_x, target_y


def engineer_features(dataframe: pl.DataFrame, config: Dict[str, Any]) -> Tuple[pl.DataFrame, pl.Series]:
    """Clean + encode; return feature matrix X and target vector y."""
    dataframe = _clean_features(dataframe, config)
    features_x, target_y = _encode_features(dataframe)
    logger.info("FEATURE_ENGINEERING_DONE: %d features, %d samples", features_x.width, features_x.height)
    return features_x, target_y


def _train_model(
        x_train: pl.DataFrame,
        y_train: pl.Series,
        x_test: pl.DataFrame,
        y_test: pl.Series,
        seed: int) -> lgb.LGBMClassifier:
    """Fit an LGBMClassifier with the configured hyperparameters."""
    lgb_params = {
        **LIGHTGBM_CONFIG,
        "random_state": seed,
    }

    mlflow.log_params(lgb_params)
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
    )
    return model


def main() -> None:
    """Run the full training pipeline: load, split, train, score, persist."""
    seed = MODEL_CONFIG["random_state"]
    dataframe = load_data(DATA_CONFIG)
    features_x, target_y = engineer_features(dataframe, DATA_CONFIG)

    x_train, x_test, y_train, y_test = train_test_split(
        features_x.to_pandas(), target_y.to_pandas(),
        test_size=MODEL_CONFIG["test_size"],
        random_state=seed,
        stratify=target_y.to_pandas(),
    )
    logger.info("TRAIN_TEST_SPLIT: train=%d  test=%d", len(x_train), len(x_test))

    mlflow.set_tracking_uri(MLFLOW_CONFIG.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(MLFLOW_CONFIG.get("experiment_name", "churn_prediction"))

    with mlflow.start_run() as run:
        logger.info("MLFLOW_RUN_STARTED: run_id=%s", run.info.run_id)
        model = _train_model(x_train, y_train, x_test, y_test, seed)

        y_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        logger.info("TEST_ROC_AUC: %.4f", roc_auc)

        model_path = ROOT / MODEL_CONFIG["save_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {"pipeline": model}
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("MODEL_SAVED: %s", model_path)


if __name__ == "__main__":
    main()
