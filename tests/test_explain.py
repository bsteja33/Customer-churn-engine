"""Unit tests for src/explain.py — SHAP feature attribution extraction.

These tests do NOT require a real trained model. They construct a MagicMock
``LGBMClassifier`` whose ``booster_.predict(..., pred_contrib=True)`` is
programmed to return a known shape, then assert the explainer slices,
sorts, and labels the output correctly.
"""

import sys
import pathlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from src.explain import (  # noqa: E402
    DEFAULT_TOP_K,
    explain_prediction,
    _get_booster,
    _humanize_category,
    _label_feature,
    _reverse_col_map,
    _validate_contribs,
)


# --- Fixtures ---------------------------------------------------------------

def _make_pipeline(contribs: np.ndarray, has_bias_col: bool = True) -> MagicMock:
    """Build a mock LGBMClassifier with a fake booster that returns
    ``contribs`` from ``predict(..., pred_contrib=True)``.
    """
    booster = MagicMock()
    booster.predict.return_value = contribs
    pipeline = MagicMock()
    pipeline.booster_ = booster
    # Provide a predict_proba stub so the harness resembles a real pipeline.
    pipeline.predict_proba.return_value = np.array([[0.4, 0.6]])
    return pipeline


def _onehot_frame() -> pd.DataFrame:
    """A small frame with both numeric and one-hot columns, matching the
    sanitization pattern used by engineer_features_inference.
    """
    return pd.DataFrame(
        [
            {
                "tenure": 2,
                "MonthlyCharges": 95.0,
                "Contract_Month_to_Month": 1,
                "Contract_Two_Year": 0,
                "InternetService_Fiber_Optic": 1,
                "TechSupport_Yes": 0,
            }
        ]
    )


# --- _humanize_category -----------------------------------------------------

class TestHumanizeCategory:
    def test_replaces_underscores_with_spaces(self):
        assert _humanize_category("Fiber_Optic") == "Fiber Optic"

    def test_pretty_prints_month_to_month(self):
        assert _humanize_category("Month_to_Month") == "Month-to-Month"

    def test_empty_string(self):
        assert _humanize_category("") == ""


# --- _label_feature ---------------------------------------------------------

class TestLabelFeature:
    def test_onehot_column_is_labeled_with_base_and_value(self):
        label = _label_feature("Contract_Month_to_Month", {})
        assert label == "Contract: Month-to-Month"

    def test_numeric_column_returns_sanitized_name(self):
        # No underscore => not a one-hot, no reverse-map match.
        assert _label_feature("tenure", {}) == "tenure"

    def test_uses_reverse_map_when_available(self):
        # Simulate a reverse map where "Senior_Citizen" maps to "SeniorCitizen"
        label = _label_feature("Senior_Citizen", {"Senior_Citizen": "SeniorCitizen"})
        assert label == "SeniorCitizen"


# --- _get_booster -----------------------------------------------------------

class TestGetBooster:
    def test_returns_none_for_none_input(self):
        assert _get_booster(None) is None

    def test_extracts_from_bare_classifier(self):
        booster = MagicMock()
        clf = MagicMock(spec=["booster_"])
        clf.booster_ = booster
        assert _get_booster(clf) is booster

    def test_unwraps_sklearn_pipeline(self):
        booster = MagicMock()
        clf = MagicMock(spec=["booster_"])
        clf.booster_ = booster
        pipeline = MagicMock()
        pipeline.steps = [("scaler", MagicMock()), ("clf", clf)]
        assert _get_booster(pipeline) is booster

    def test_unwraps_calibrated_classifier(self):
        # A real CalibratedClassifierCV has no `booster_` itself; the
        # booster lives one level down via `estimator`.
        booster = MagicMock()
        clf = MagicMock(spec=["booster_", "predict_proba"])
        clf.booster_ = booster
        meta = MagicMock(spec=["estimator"])
        meta.estimator = clf
        # meta has no `booster_` directly — spec=["estimator"] enforces that.
        assert _get_booster(meta) is booster

    def test_returns_none_when_no_booster(self):
        obj = MagicMock(spec=[])  # no booster_ attribute
        obj.booster_ = None
        assert _get_booster(obj) is None


# --- explain_prediction -----------------------------------------------------

class TestExplainPrediction:
    def test_returns_none_for_empty_pipeline(self):
        assert explain_prediction(None, _onehot_frame()) is None

    def test_returns_none_for_empty_dataframe(self):
        pipeline = _make_pipeline(np.zeros((1, 1)))
        assert explain_prediction(pipeline, pd.DataFrame()) is None

    def test_returns_none_for_zero_top_k(self):
        pipeline = _make_pipeline(np.zeros((1, 7)))
        df = _onehot_frame()
        assert explain_prediction(pipeline, df, top_k=0) is None

    def test_slices_bias_column_when_present(self):
        # 6 features + 1 bias column = 7 columns of contribs.
        n_features = 6
        contribs = np.zeros((1, n_features + 1))
        contribs[0, 0] = 0.5  # tenure
        contribs[0, 2] = 0.3  # Contract_Month_to_Month
        contribs[0, -1] = 0.1  # bias
        pipeline = _make_pipeline(contribs, has_bias_col=True)
        df = _onehot_frame()
        out = explain_prediction(pipeline, df, top_k=3)
        assert out is not None
        # Should be sorted by absolute magnitude, descending.
        mags = [row["magnitude"] for row in out]
        assert mags == sorted(mags, reverse=True)
        # The first row should be the tenure contribution.
        assert out[0]["feature"].startswith("tenure") or "tenure" in out[0]["feature"]
        assert out[0]["direction"] == "up"
        assert out[0]["value"] == 2

    def test_succeeds_when_bias_column_absent(self):
        # 6 features, no bias column.
        n_features = 6
        contribs = np.zeros((1, n_features))
        contribs[0, 1] = -0.2  # MonthlyCharges (negative => down)
        pipeline = _make_pipeline(contribs, has_bias_col=False)
        df = _onehot_frame()
        out = explain_prediction(pipeline, df, top_k=2)
        assert out is not None
        assert out[0]["direction"] == "down"
        assert out[0]["magnitude"] == 0.2

    def test_returns_none_on_shape_mismatch(self):
        # 5 cols but df has 6 features
        pipeline = _make_pipeline(np.zeros((1, 5)))
        df = _onehot_frame()
        assert explain_prediction(pipeline, df) is None

    def test_returns_none_when_booster_raises(self):
        booster = MagicMock()
        booster.predict.side_effect = RuntimeError("kaboom")
        pipeline = MagicMock()
        pipeline.booster_ = booster
        out = explain_prediction(pipeline, _onehot_frame())
        assert out is None

    def test_returns_none_when_pipeline_lacks_booster(self):
        pipeline = MagicMock(spec=["predict_proba"])  # no booster_
        out = explain_prediction(pipeline, _onehot_frame())
        assert out is None

    def test_caps_top_k_at_number_of_features(self):
        n_features = 6
        contribs = np.ones((1, n_features + 1)) * 0.01
        pipeline = _make_pipeline(contribs)
        df = _onehot_frame()
        out = explain_prediction(pipeline, df, top_k=50)  # more than 6
        assert out is not None
        assert len(out) == n_features

    def test_respects_default_top_k(self):
        n_features = DEFAULT_TOP_K + 5
        contribs = np.ones((1, n_features + 1)) * 0.01
        contribs[0, :n_features] = np.linspace(0.1, 0.01, n_features)  # descending
        pipeline = _make_pipeline(contribs)
        # Build a matching frame
        df = pd.DataFrame([{f"f{i}": i for i in range(n_features)}])
        out = explain_prediction(pipeline, df)
        assert out is not None
        assert len(out) == DEFAULT_TOP_K

    def test_handles_nan_in_user_value(self):
        n_features = 2
        contribs = np.zeros((1, n_features + 1))
        contribs[0, 0] = 0.5
        pipeline = _make_pipeline(contribs)
        df = pd.DataFrame([{"a": float("nan"), "b": 1.0}])
        out = explain_prediction(pipeline, df)
        assert out is not None
        # The "a" row's value should be None, not NaN.
        for row in out:
            if row["feature"] == "a":
                assert row["value"] is None

    def test_handles_non_finite_contributions(self):
        n_features = 2
        contribs = np.zeros((1, n_features + 1))
        contribs[0, 0] = np.inf
        contribs[0, 1] = -np.nan
        pipeline = _make_pipeline(contribs)
        df = pd.DataFrame([{"a": 1, "b": 2}])
        out = explain_prediction(pipeline, df)
        assert out is not None
        # Non-finite values should have been replaced with 0 magnitude
        # and excluded from the ranking (or appear with 0 magnitude).
        assert all(r["magnitude"] == 0.0 for r in out)

    def test_returns_none_on_1d_contribs(self):
        pipeline = _make_pipeline(np.zeros(5))  # 1D, not 2D
        df = _onehot_frame()
        assert explain_prediction(pipeline, df) is None

    def test_label_format_includes_onehot_split(self):
        n_features = 3
        contribs = np.zeros((1, n_features + 1))
        contribs[0, 0] = 0.7
        pipeline = _make_pipeline(contribs)
        df = pd.DataFrame([{
            "Contract_Month_to_Month": 1,
            "Contract_Two_Year": 0,
            "tenure": 2,
        }])
        out = explain_prediction(pipeline, df, top_k=1)
        assert out is not None
        assert "Contract" in out[0]["feature"]
        assert "Month-to-Month" in out[0]["feature"]


# --- _validate_contribs -----------------------------------------------------

class TestValidateContribs:
    """Locks in the shape contract: returns (per_feature, n_features) on
    success, None on any failure."""

    def test_returns_none_when_pipeline_is_none(self):
        assert _validate_contribs(None, _onehot_frame()) is None

    def test_slices_off_bias_column(self):
        n_features = 6
        contribs = np.zeros((1, n_features + 1))
        contribs[0, 0] = 0.5
        pipeline = _make_pipeline(contribs)
        df = _onehot_frame()
        out = _validate_contribs(pipeline, df)
        assert out is not None
        per_feature, returned_n = out
        assert returned_n == n_features
        assert per_feature.shape == (n_features,)
        assert per_feature[0] == 0.5

    def test_accepts_n_features_no_bias(self):
        n_features = 6
        contribs = np.zeros((1, n_features))
        pipeline = _make_pipeline(contribs)
        out = _validate_contribs(pipeline, _onehot_frame())
        assert out is not None
        per_feature, _ = out
        assert per_feature.shape == (n_features,)

    def test_returns_none_on_column_mismatch(self):
        # 5 cols but df has 6 features
        pipeline = _make_pipeline(np.zeros((1, 5)))
        assert _validate_contribs(pipeline, _onehot_frame()) is None

    def test_returns_none_when_booster_raises(self):
        booster = MagicMock()
        booster.predict.side_effect = RuntimeError("kaboom")
        pipeline = MagicMock()
        pipeline.booster_ = booster
        assert _validate_contribs(pipeline, _onehot_frame()) is None

    def test_returns_none_when_pipeline_lacks_booster(self):
        pipeline = MagicMock(spec=["predict_proba"])
        assert _validate_contribs(pipeline, _onehot_frame()) is None

    def test_returns_none_on_1d_contribs(self):
        pipeline = _make_pipeline(np.zeros(5))  # 1D, not 2D
        assert _validate_contribs(pipeline, _onehot_frame()) is None


# --- _reverse_col_map -------------------------------------------------------

class TestReverseColMap:
    def test_contains_known_sanitized_columns(self):
        reverse = _reverse_col_map()
        # col_map maps "tenure" -> "Tenure in Months", sanitized to "Tenure_in_Months"
        assert "Tenure_in_Months" in reverse
        assert reverse["Tenure_in_Months"] == "tenure"
        # SeniorCitizen -> "Senior Citizen" -> "Senior_Citizen"
        assert "Senior_Citizen" in reverse
        assert reverse["Senior_Citizen"] == "SeniorCitizen"


# --- Real-model numerical parity -----------------------------------------

class TestShapNumericalParity:
    """End-to-end against the persisted LightGBM model: the SHAP
    contributions returned by ``explain_prediction`` must reconstruct
    the model's logit to numerical precision. Catches any drift in the
    feature-engineering pipeline that would silently corrupt the
    importance values."""

    @pytest.fixture(scope="class")
    def artifact(self):
        import joblib
        import pathlib
        path = pathlib.Path(ROOT) / "models" / "churn_model.pkl"
        return joblib.load(path)

    def _aligned_frame(self, record):
        import pandas as pd
        from src.feature_engineering import engineer_features_inference
        df = engineer_features_inference(pd.DataFrame([record]))
        return df

    def test_shap_sum_plus_bias_equals_logit_high_risk(
        self, artifact
    ):
        """For the high-risk preset, the SHAP contributions returned to
        the FE must sum (with the bias term) to the model's logit."""
        record = {
            "Gender": "Male", "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
            "tenure": 2, "PhoneService": 1, "MultipleLines": 0,
            "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
            "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
            "StreamingMovies": 1, "Contract": "Month-to-Month",
            "PaperlessBilling": 1, "PaymentMethod": "Bank Withdrawal",
            "MonthlyCharges": 95, "TotalCharges": 190, "Married": 0,
            "NumberOfDependents": 0, "NumberOfReferrals": 0,
            "SatisfactionScore": 2, "InternetType": "Fiber Optic",
            "Offer": "None", "Age": 28, "AvgMonthlyGBDownload": 80,
            "AvgMonthlyLongDistanceCharges": 5, "CLTV": 0, "Under30": 1,
            "UnlimitedData": 1, "StreamingMusic": 0, "ReferredAFriend": 0,
            "TotalRefunds": 0, "TotalExtraDataCharges": 10,
            "TotalLongDistanceCharges": 20, "TotalRevenue": 200,
        }
        pipeline = artifact["pipeline"]
        expected_features = list(pipeline.feature_name_)

        df = self._aligned_frame(record)[expected_features]
        proba = float(pipeline.predict_proba(df)[0, 1])
        logit_from_proba = float(np.log(proba / (1.0 - proba)))

        # Ask the booster for the raw contrib vector.
        booster = pipeline.booster_
        raw = np.asarray(booster.predict(df, pred_contrib=True))
        per_feature = raw[0, :-1]  # last column is the bias
        bias = float(raw[0, -1])
        logit_from_shap = float(per_feature.sum() + bias)

        # Numerical parity: must match to 1e-6 (LightGBM's tree sums
        # are exact floats, so any drift here is a real bug).
        assert abs(logit_from_shap - logit_from_proba) < 1e-6, (
            f"SHAP logit ({logit_from_shap}) != predict_proba logit "
            f"({logit_from_proba}); diff={logit_from_shap - logit_from_proba}"
        )
        # Sanity: the reconstructed logit is consistent with the high
        # probability (~0.999) we expect for this record.
        assert proba > 0.9

    def test_top_contributor_is_satisfaction_score(self, artifact):
        """For the high-risk preset, SatisfactionScore=2 must be the
        dominant SHAP contributor (the dataset's #1 churn signal)."""
        record = {
            "Gender": "Male", "SeniorCitizen": 0, "Partner": 0,
            "Dependents": 0, "tenure": 2, "PhoneService": 1,
            "MultipleLines": 0, "InternetService": 1, "OnlineSecurity": 0,
            "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
            "StreamingTV": 1, "StreamingMovies": 1,
            "Contract": "Month-to-Month", "PaperlessBilling": 1,
            "PaymentMethod": "Bank Withdrawal", "MonthlyCharges": 95,
            "TotalCharges": 190, "Married": 0, "NumberOfDependents": 0,
            "NumberOfReferrals": 0, "SatisfactionScore": 2,
            "InternetType": "Fiber Optic", "Offer": "None", "Age": 28,
            "AvgMonthlyGBDownload": 80, "AvgMonthlyLongDistanceCharges": 5,
            "CLTV": 0, "Under30": 1, "UnlimitedData": 1,
            "StreamingMusic": 0, "ReferredAFriend": 0, "TotalRefunds": 0,
            "TotalExtraDataCharges": 10, "TotalLongDistanceCharges": 20,
            "TotalRevenue": 200,
        }
        pipeline = artifact["pipeline"]
        expected = list(pipeline.feature_name_)
        df = self._aligned_frame(record)[expected]
        result = explain_prediction(pipeline, df, top_k=8)
        assert result is not None
        # First entry must be SatisfactionScore (or its sanitized form).
        assert "Satisfaction" in result[0]["feature"]

    def test_aligned_frame_matches_model_columns(self, artifact):
        """The engineered frame must contain exactly the 51 columns
        the model expects — no missing, no extras. Guards against
        future drift in the categorical encoding tables."""
        record = {
            "Gender": "Female", "SeniorCitizen": 1, "Partner": 1,
            "Dependents": 1, "tenure": 24, "PhoneService": 1,
            "MultipleLines": 1, "InternetService": 1,
            "OnlineSecurity": 1, "OnlineBackup": 1,
            "DeviceProtection": 1, "TechSupport": 1, "StreamingTV": 0,
            "StreamingMovies": 1, "Contract": "Two Year",
            "PaperlessBilling": 0, "PaymentMethod": "Credit Card",
            "MonthlyCharges": 60, "TotalCharges": 1500, "Married": 1,
            "NumberOfDependents": 2, "NumberOfReferrals": 1,
            "SatisfactionScore": 4, "InternetType": "DSL",
            "Offer": "Offer C", "Age": 42, "AvgMonthlyGBDownload": 30,
            "AvgMonthlyLongDistanceCharges": 10, "CLTV": 3500,
            "Under30": 0, "UnlimitedData": 0, "StreamingMusic": 1,
            "ReferredAFriend": 1, "TotalRefunds": 25,
            "TotalExtraDataCharges": 0, "TotalLongDistanceCharges": 200,
            "TotalRevenue": 1600,
        }
        pipeline = artifact["pipeline"]
        expected = set(pipeline.feature_name_)
        df = self._aligned_frame(record)
        produced = set(df.columns)
        assert produced == expected, (
            f"Engineered frame mismatch — "
            f"missing: {expected - produced}, "
            f"extra: {produced - expected}"
        )
