"""Tests for src/ modules — feature_engineering, predict, train.
All external APIs (datasets, joblib, MLflow) are mocked."""

import sys
import pathlib
from unittest.mock import patch, MagicMock
import pandas as pd
import polars as pl
import pytest

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from src.train import engineer_features as polars_engineer, load_data  # noqa: E402
from src import predict as predict_mod  # noqa: E402
from src.feature_engineering import engineer_features as pandas_engineer  # noqa: E402


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Senior Citizen": [0, 1],
        "Partner": [1, 0],
        "Dependents": [0, 1],
        "Tenure in Months": [12, 5],
        "Phone Service": [1, 0],
        "Multiple Lines": [0, 1],
        "Internet Service": [1, 0],
        "Online Security": [1, 0],
        "Online Backup": [0, 1],
        "Device Protection Plan": [0, 1],
        "Premium Tech Support": [1, 0],
        "Streaming TV": [1, 0],
        "Streaming Movies": [0, 1],
        "Contract": ["Month-to-Month", "One Year"],
        "Paperless Billing": [1, 0],
        "Payment Method": ["Bank Withdrawal", "Credit Card"],
        "Monthly Charge": [75.0, 50.0],
        "Total Charges": [900.0, 250.0],
        "Total Revenue": [1200.0, 300.0],
        "Married": [1, 0],
        "Number of Dependents": [0, 2],
        "Number of Referrals": [0, 1],
        "Satisfaction Score": [3, 4],
        "Internet Type": ["Fiber Optic", "DSL"],
        "Offer": ["Offer A", "None"],
        "Age": [45, 30],
        "Avg Monthly GB Download": [50, 20],
        "Avg Monthly Long Distance Charges": [15.0, 5.0],
        "CLTV": [4000, 2500],
        "Under 30": [0, 0],
        "Unlimited Data": [1, 0],
        "Streaming Music": [1, 0],
        "Referred a Friend": [0, 1],
        "Total Refunds": [0.0, 10.0],
        "Total Extra Data Charges": [5, 0],
        "Total Long Distance Charges": [45.0, 10.0],
        "Churn": [1, 0],
    })


class TestPandasFeatureEngineering:
    def test_returns_tuple_of_four(self, sample_df):
        X, y, cat_cols, num_cols = pandas_engineer(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(cat_cols, list)
        assert isinstance(num_cols, list)
        assert "Churn" not in X.columns

    def test_creates_revenue_per_tenure(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Revenue_per_Tenure" in X.columns


class TestPredictSingle:
    @patch("joblib.load")
    def test_predict_single_returns_dict(self, mock_joblib):
        import numpy as np
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        result = predict_mod.predict_single({"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0})
        assert isinstance(result, dict)
        assert "prediction" in result

    @patch("joblib.load")
    def test_predict_single_high_risk(self, mock_joblib):
        import numpy as np
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])  # class 1 prob = 0.8
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        result = predict_mod.predict_single({"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0})
        assert result["retention_risk"] == "High"

    @patch("joblib.load")
    def test_predict_single_medium_risk(self, mock_joblib):
        import numpy as np
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.4, 0.6]])  # class 1 prob = 0.6
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        result = predict_mod.predict_single({"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0})
        assert result["retention_risk"] == "Medium"

    @patch("joblib.load")
    def test_predict_single_low_risk(self, mock_joblib):
        import numpy as np
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])  # class 1 prob = 0.2
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        result = predict_mod.predict_single({"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0})
        assert result["retention_risk"] == "Low"

    def test_predict_single_missing_model(self):
        predict_mod._ARTIFACT_CACHE = None
        import pathlib
        missing_path = pathlib.Path("does_not_exist.pkl")
        with pytest.raises(FileNotFoundError):
            predict_mod.predict_single({"Gender": "Male"}, model_path=missing_path)


class TestPredictBatch:
    @patch("joblib.load")
    def test_predict_batch_returns_dataframe(self, mock_joblib):
        import numpy as np
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        df = pd.DataFrame({"feature": [1, 2]})
        result = predict_mod.predict_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert "churn_probability" in result.columns


class TestTrainEngineerFeatures:
    @patch("src.train.load_dataset")
    def test_engineer_features_polars_returns_tuple(self, mock_load_dataset):
        cfg = {
            "drop_columns": ["Customer ID"],
            "target_column": "Churn",
            "positive_class": 1,
            "negative_class": 0,
        }
        df = pl.DataFrame({
            "Churn": [1, 0, 1],
            "Gender": ["M", "F", "M"],
            "Senior Citizen": ["0", "1", "0"],
            "Customer ID": ["a", "b", "c"],
        })
        X, y = polars_engineer(df, cfg)
        assert "Customer ID" not in X.columns


class TestTrainLoadData:
    @patch("src.train.pl.read_csv")
    def test_load_data_from_local_csv(self, mock_read_csv):
        mock_df = pl.DataFrame({"Churn": [1, 0], "Gender": ["M", "F"]})
        mock_read_csv.return_value = mock_df
        cfg = {
            "raw_path": "some/path.csv",
            "max_samples": 2,
        }
        df = load_data(cfg)
        assert df.height == 2


class TestTrainCleanEncode:
    def test_clean_features_drops_and_handles_total_charges(self):
        from src.train import _clean_features
        import polars as pl
        df = pl.DataFrame({
            "DropMe": [1, 2],
            "TotalCharges": ["100.5", " "],
            "Churn": [1, 0]
        })
        config = {
            "drop_columns": ["DropMe"],
            "positive_class": 1,
            "negative_class": 0,
            "target_column": "Churn"
        }
        df_clean = _clean_features(df, config)
        assert "DropMe" not in df_clean.columns
        assert df_clean["TotalCharges"].dtype == pl.Float64
        assert df_clean["TotalCharges"].to_list() == [100.5, 0.0]

    def test_encode_features_separates_and_sanitizes(self):
        from src.train import _encode_features
        import polars as pl
        df = pl.DataFrame({
            "Gender": ["M", "F"],
            "Churn": [1, 0]
        })
        X, y = _encode_features(df)
        assert "Gender_M" in X.columns or "Gender_F" in X.columns
        assert "Churn" not in X.columns
        assert list(y) == [1, 0]

    @patch("lightgbm.LGBMClassifier")
    @patch("mlflow.log_params")
    def test_train_model(self, mock_log_params, mock_lgbm):
        from src.train import _train_model
        import polars as pl
        X_train = pl.DataFrame({"feature": [1, 2]})
        y_train = pl.Series([0, 1])
        X_test = pl.DataFrame({"feature": [3]})
        y_test = pl.Series([0])

        mock_model_instance = MagicMock()
        mock_lgbm.return_value = mock_model_instance

        result = _train_model(X_train, y_train, X_test, y_test, 42)
        assert result == mock_model_instance
        mock_model_instance.fit.assert_called_once()
