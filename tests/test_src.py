"""Tests for src/ modules - feature_engineering, predict, train.

All external APIs (datasets, joblib, MLflow) are mocked. The
train/serve parity tests run against the real committed artifact.
"""

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
from src.feature_engineering import (  # noqa: E402
    encode_dataset_frame,
    engineer_features_inference,
)


MODEL_PATH = pathlib.Path(ROOT) / "models" / "churn_model.pkl"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="committed model artifact not present"
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset-named records (the training data's column convention)."""
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


# API-shaped record keyed by the CustomerFeatures field names. The
# dataset-named twin below must produce the same encoded row.
HIGH_RISK_RECORD = {
    "Gender": "Male", "SeniorCitizen": 1, "Partner": 0, "Dependents": 1,
    "tenure": 3, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
    "Contract": "Month-to-Month", "PaperlessBilling": 1,
    "PaymentMethod": "Mailed Check", "MonthlyCharges": 95.5,
    "TotalCharges": 280.0, "Married": 0, "NumberOfDependents": 2,
    "NumberOfReferrals": 1, "SatisfactionScore": 1, "InternetType": "Fiber Optic",
    "Offer": "None", "Age": 55, "AvgMonthlyGBDownload": 20,
    "AvgMonthlyLongDistanceCharges": 12.5, "CLTV": 3500, "Under30": 0,
    "UnlimitedData": 1, "StreamingMusic": 0, "ReferredAFriend": 1,
    "TotalRefunds": 0.0, "TotalExtraDataCharges": 0,
    "TotalLongDistanceCharges": 40.0, "TotalRevenue": 350.0,
}

# Dataset-naming counterpart of HIGH_RISK_RECORD (col_map applied).
HIGH_RISK_RECORD_DATASET = {
    "Gender": "Male", "Senior Citizen": 1, "Partner": 0, "Dependents": 1,
    "Tenure in Months": 3, "Phone Service": 1, "Multiple Lines": 0,
    "Internet Service": 1, "Online Security": 0, "Online Backup": 0,
    "Device Protection Plan": 0, "Premium Tech Support": 0,
    "Streaming TV": 0, "Streaming Movies": 0,
    "Contract": "Month-to-Month", "Paperless Billing": 1,
    "Payment Method": "Mailed Check", "Monthly Charge": 95.5,
    "Total Charges": 280.0, "Married": 0, "Number of Dependents": 2,
    "Number of Referrals": 1, "Satisfaction Score": 1,
    "Internet Type": "Fiber Optic", "Offer": "None", "Age": 55,
    "Avg Monthly GB Download": 20,
    "Avg Monthly Long Distance Charges": 12.5, "CLTV": 3500,
    "Under 30": 0, "Unlimited Data": 1, "Streaming Music": 0,
    "Referred a Friend": 1, "Total Refunds": 0.0,
    "Total Extra Data Charges": 0, "Total Long Distance Charges": 40.0,
    "Total Revenue": 350.0,
}


class TestTrainServeParity:
    """Regression guard: training and inference MUST share one encoding.

    The committed artifact's ``feature_names_in_`` is the single source
    of truth; both paths must reproduce it exactly and identically.
    """

    @needs_model
    def test_both_paths_reproduce_artifact_feature_set(self):
        import joblib

        model = joblib.load(MODEL_PATH)["pipeline"]
        expected = set(model.feature_names_in_)

        api_frame = engineer_features_inference(pd.DataFrame([HIGH_RISK_RECORD]))
        train_frame = encode_dataset_frame(
            pd.DataFrame([HIGH_RISK_RECORD_DATASET])
        )

        assert set(api_frame.columns) == expected
        assert set(train_frame.columns) == expected

    @needs_model
    def test_api_and_train_frames_are_row_identical(self):
        api_frame = engineer_features_inference(pd.DataFrame([HIGH_RISK_RECORD]))
        train_frame = encode_dataset_frame(
            pd.DataFrame([HIGH_RISK_RECORD_DATASET])
        )
        # Column order is irrelevant to the model (it reindexes by
        # name), so compare as sorted sets of (column, value) pairs.
        api_cells = sorted(zip(api_frame.columns, api_frame.iloc[0]))
        train_cells = sorted(zip(train_frame.columns, train_frame.iloc[0]))
        assert api_cells == train_cells

    @needs_model
    def test_one_hot_always_emits_full_vocabulary(self):
        """A single record must produce every dummy column the model was
        fit on, regardless of which categories the record contains."""
        minimal = pd.DataFrame([{"Gender": "Female", "Contract": "Two Year"}])
        frame = engineer_features_inference(minimal)
        for col in (
            "Gender_Female", "Gender_Male",
            "Contract_Month_to_Month", "Contract_One_Year", "Contract_Two_Year",
            "Offer_null", "Internet_Type_null",
            "Senior_Citizen_0", "Senior_Citizen_1",
        ):
            assert col in frame.columns

    @needs_model
    def test_none_sentinel_maps_to_null_category(self):
        frame = engineer_features_inference(
            pd.DataFrame([{"Offer": "None", "InternetType": "None"}])
        )
        assert frame["Offer_null"].iloc[0] == 1
        assert frame["Internet_Type_null"].iloc[0] == 1

    @needs_model
    def test_prediction_is_deterministic(self):
        import joblib

        model = joblib.load(MODEL_PATH)["pipeline"]
        expected = list(model.feature_names_in_)
        frame = engineer_features_inference(pd.DataFrame([HIGH_RISK_RECORD]))
        p1 = float(model.predict_proba(frame[expected])[0][1])
        p2 = float(model.predict_proba(frame[expected])[0][1])
        assert p1 == p2
        assert 0.0 <= p1 <= 1.0

    @needs_model
    def test_encoded_frame_is_fully_numeric(self, sample_df):
        features = encode_dataset_frame(sample_df.drop(columns=["Churn"]))
        assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)


class TestPredictSingle:
    @patch("joblib.load")
    def test_predict_single_returns_dict(self, mock_joblib):
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        result = predict_mod.predict_single(
            {"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0}
        )
        assert isinstance(result, dict)
        assert "prediction" in result

    def test_predict_single_missing_artifact_raises(self):
        missing_path = pathlib.Path(ROOT) / "models" / "does_not_exist.pkl"
        predict_mod._ARTIFACT_CACHE = None
        with pytest.raises(FileNotFoundError):
            predict_mod.predict_single(
                {"Gender": "Male"}, model_path=missing_path
            )


class TestPredictBatch:
    @patch("joblib.load")
    def test_predict_batch_returns_dataframe(self, mock_joblib):
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7]]
        )
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        df = pd.DataFrame({"feature": [1, 2]})
        result = predict_mod.predict_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert "churn_probability" in result.columns


class TestTrainEngineerFeatures:
    def test_engineer_features_polars_returns_frame_and_target(self):
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
        df = load_data({"raw_path": "some/path.csv", "max_samples": 2})
        assert df.height == 2


class TestTrainCleanEncode:
    def test_clean_features_drops_and_handles_total_charges(self):
        from src.train import _clean_features

        df = pl.DataFrame({
            "DropMe": [1, 2],
            "TotalCharges": ["100.5", " "],
            "Churn": [1, 0],
        })
        config = {
            "drop_columns": ["DropMe"],
            "positive_class": 1,
            "negative_class": 0,
            "target_column": "Churn",
        }
        df_clean = _clean_features(df, config)
        assert "DropMe" not in df_clean.columns
        assert df_clean["TotalCharges"].dtype == pl.Float64
        assert df_clean["TotalCharges"].to_list() == [100.5, 0.0]

    def test_encode_features_uses_shared_core(self):
        """The training encode step must emit the same dummy columns as
        the inference path (single shared core, see
        TestTrainServeParity)."""
        from src.train import _encode_features

        df = pl.DataFrame({
            "Gender": ["Female", "Male"],
            "Contract": ["Two Year", "Month-to-Month"],
            "Churn": [1, 0],
        })
        X, y = _encode_features(df)
        assert "Churn" not in X.columns
        assert list(y) == [1, 0]
        for col in (
            "Gender_Female", "Gender_Male",
            "Contract_Two_Year", "Contract_Month_to_Month",
        ):
            assert col in X.columns

    @patch("lightgbm.LGBMClassifier")
    @patch("mlflow.log_params")
    def test_train_model(self, _mock_log_params, mock_lgbm):
        from src.train import _train_model

        X_train = pl.DataFrame({"feature": [1, 2]})
        y_train = pl.Series([0, 1])
        X_test = pl.DataFrame({"feature": [3]})
        y_test = pl.Series([0])

        mock_model_instance = MagicMock()
        mock_lgbm.return_value = mock_model_instance

        result = _train_model(X_train, y_train, X_test, y_test, 42)
        assert result == mock_model_instance
        mock_model_instance.fit.assert_called_once()
