"""Centralized configuration for the Churn Prediction system."""

import os
from pathlib import Path
from typing import Dict, Any

ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data Configuration
DATA_CONFIG: Dict[str, Any] = {
    "raw_path": None,
    "dataset": "aai510-group1/telco-customer-churn",
    "max_samples": 50000,
    "drop_columns": [
        "Customer ID",
        "Churn Category",
        "Churn Reason",
        "Churn Score",
        "Customer Status",
        "City",
        "Country",
        "Lat Long",
        "Latitude",
        "Longitude",
        "State",
        "Zip Code",
        "Quarter",
        "Population",
    ],
    "target_column": "Churn",
    "positive_class": 1,
    "negative_class": 0,
}

# Model Configuration
MODEL_CONFIG: Dict[str, Any] = {
    "save_path": "models/churn_model.pkl",
    "random_state": 42,
    "test_size": 0.20,
    "threshold": 0.5,
}

# LightGBM Configuration
LIGHTGBM_CONFIG: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 31,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "verbosity": -1,
}

# MLflow Configuration
MLFLOW_CONFIG: Dict[str, Any] = {
    "tracking_uri": "mlruns",
    "experiment_name": "churn_prediction_polars",
}

# API Configuration
LLM_PROVIDER_API_KEY: str = os.environ.get("LLM_PROVIDER_API_KEY", "").strip()
