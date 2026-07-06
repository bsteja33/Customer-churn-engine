"""Inference module — loads the persisted model and runs single/batch prediction."""

import argparse
import json
import logging
import pathlib
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_CONFIG
from src.feature_engineering import engineer_features_inference

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = ROOT / MODEL_CONFIG.get("save_path", "models/churn_model.pkl")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

_ARTIFACT_CACHE: dict | None = None


def _load_artifact(model_path: pathlib.Path) -> dict:
    """Load and cache the model artifact."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run 'python src/train.py' first."
            )
        _ARTIFACT_CACHE = joblib.load(model_path)
        pipeline = _ARTIFACT_CACHE.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            _ARTIFACT_CACHE = None
            raise RuntimeError(
                f"Model artifact at '{model_path}' is corrupted: "
                "missing pipeline with predict_proba."
            )
        logger.info("MODEL_LOADED: %s", model_path)
    return _ARTIFACT_CACHE


def predict_single(
    customer: Dict[str, Any],
    model_path: pathlib.Path = _DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Predict churn probability for a single customer record."""
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]

    dataframe = pd.DataFrame([customer])
    dataframe = engineer_features_inference(dataframe)

    _threshold = float(MODEL_CONFIG.get("threshold", 0.5))

    churn_proba = float(pipeline.predict_proba(dataframe)[0][1])
    prediction = int(churn_proba >= _threshold)

    if churn_proba >= 0.70:
        risk = "High"
    elif churn_proba >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    result = {
        "prediction": prediction,
        "churn_probability": round(churn_proba, 4),
        "retention_risk": risk,
    }

    logger.info("PREDICTION_GENERATED: INPUT=%s OUTPUT=%s", json.dumps(customer), json.dumps(result))

    return result


def predict_batch(
    dataframe: pd.DataFrame,
    model_path: pathlib.Path = _DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Run batch predictions and return a copy with prediction columns appended."""
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]

    probas = pipeline.predict_proba(dataframe)[:, 1]
    preds = (probas >= float(MODEL_CONFIG.get("threshold", 0.5))).astype(int)

    result = dataframe.copy()
    result["churn_probability"] = np.round(probas, 4)
    result["prediction"] = preds
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Customer Churn Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="JSON string of a single customer record")
    group.add_argument("--csv", type=str, help="Path to a CSV for batch prediction")
    parser.add_argument("--model", default=str(_DEFAULT_MODEL_PATH), help="Model path")
    args = parser.parse_args()

    mpath = pathlib.Path(args.model)

    if args.input:
        record = json.loads(args.input)
        out_result = predict_single(record, model_path=mpath)
        logger.info("\n%s", json.dumps(out_result, indent=2))
    else:
        df_in = pd.read_csv(args.csv)
        out_df = predict_batch(df_in, model_path=mpath)
        logger.info("\n%s", out_df[["churn_probability", "prediction"]].to_string())
