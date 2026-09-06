"""Tests for the Churn Prediction API - validates health, feature engineering,
classification logic, and the /predict and /predict/batch endpoints.

All LLM Provider API calls are mocked - no external credentials required."""

from src.feature_engineering import col_map, BINARY_FIELDS, engineer_features_inference
from api.app import (
    app,
    _classify,
    _align_to_model,
    CustomerFeatures,
    _model_intelligence_score,
    _select_routed_models,
)
import json
import re
import os
import sys
import pathlib
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

assert isinstance(BINARY_FIELDS, set) and len(BINARY_FIELDS) > 0


# Mock helpers

# A small stand-in for the provider's live model list. Deliberately
# mixed: a flagship "max" build, two mid-size dense models, a MoE
# variant, a mini, and a non-chat entry that routing must exclude.
_FAKE_MODEL_CATALOG = [
    "demo-large-max-70b",
    "demo-mid-27b",
    "demo-moe-8x7b",
    "demo-small-9b",
    "demo-mini-3b",
    "demo-whisper-large-v3",
]


def _mock_llm_response(text: str = "We value you as a customer."):
    """Build a MagicMock that mimics LLM Provider's chat completion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_llm_mock(return_value=None, side_effect=None):
    """Create a mock LLM Provider instance that patches api.app.Groq.

    ``models.list()`` mirrors the provider SDK shape used by dynamic
    model routing, so discovery resolves against ``_FAKE_MODEL_CATALOG``
    without touching the network.
    """
    instance = MagicMock()
    instance.models.list.return_value = [
        SimpleNamespace(id=model_id) for model_id in _FAKE_MODEL_CATALOG
    ]
    if side_effect:
        instance.chat.completions.create.side_effect = side_effect
    else:
        instance.chat.completions.create.return_value = return_value or _mock_llm_response()
    return instance


def _expected_routed() -> dict:
    """The tier mapping dynamic routing must derive from the fake catalog."""
    return _select_routed_models(_FAKE_MODEL_CATALOG)


# Fixtures

@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient - the model artifact must exist at
    models/churn_model.pkl for the lifespan loader to succeed."""
    with TestClient(app) as c:
        yield c


# /health

class TestHealthEndpoint:
    def test_health_returns_200_and_healthy_status(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_reports_model_loaded(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "model_path" in data


# col_map integrity

class TestColumnMapping:
    def test_every_customer_features_field_has_mapping(self):
        mapping = col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        missing = model_fields - set(mapping.keys())
        assert not missing, f"Fields missing from col_map: {missing}"

    def test_every_mapped_column_corresponds_to_model_field(self):
        mapping = col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        extra = set(mapping.keys()) - model_fields
        assert not extra, f"col_map keys not in CustomerFeatures: {extra}"

    def test_binary_fields_all_have_renamed_column(self):
        mapping = col_map()
        for field_name, renamed in mapping.items():
            if renamed in BINARY_FIELDS:
                field_info = CustomerFeatures.model_fields.get(field_name)
                assert field_info is not None
                field_type = str(field_info.annotation)
                assert "int" in field_type, (
                    f"Binary field {field_name} maps to {renamed} "
                    f"but has type {field_type}"
                )


# engineer_features_inference

_VALID_RECORD = {
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 1,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": "Month-to-Month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0,
    "Married": 1,
    "NumberOfDependents": 0,
    "NumberOfReferrals": 0,
    "SatisfactionScore": 3,
    "InternetType": "Fiber Optic",
    "Offer": "Offer A",
    "Age": 45,
    "AvgMonthlyGBDownload": 50,
    "AvgMonthlyLongDistanceCharges": 15.0,
    "CLTV": 4000,
    "Under30": 0,
    "UnlimitedData": 1,
    "StreamingMusic": 1,
    "ReferredAFriend": 0,
    "TotalRefunds": 0.0,
    "TotalExtraDataCharges": 5,
    "TotalLongDistanceCharges": 45.0,
    "TotalRevenue": 1200.0,
}


class TestEngineerFeatures:
    def test_produces_non_empty_dataframe(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        assert not result.empty

    def test_all_output_columns_are_numeric_or_bool(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool, (
                f"Column '{col}' has non-numeric dtype {dtype}"
            )

    def test_all_column_names_are_sanitized(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        for col in result.columns:
            assert re.fullmatch(r"[a-zA-Z0-9_]+", col), (
                f"Column '{col}' contains illegal characters"
            )

    def test_binary_fields_stay_as_zero_one_columns(self):
        """Binary fields must remain a single 0/1 numeric column (matching
        training) - they must NOT be one-hot encoded into ``*_Yes`` / ``*_No``."""
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)

        assert "Partner" in result.columns, (
            f"Expected bare 'Partner' column. Got: {list(result.columns)}"
        )
        # The buggy version produced these - they should be gone.
        for forbidden in ("Partner_Yes", "Partner_No", "Phone_Service_Yes",
                          "Phone_Service_No", "Online_Security_Yes",
                          "Online_Security_No"):
            assert forbidden not in result.columns, (
                f"{forbidden!r} should not exist; model expects bare 0/1 columns"
            )
        # The bare Partner column should hold the 0/1 value from the input.
        assert int(result["Partner"].iloc[0]) == _VALID_RECORD["Partner"]
        assert int(result["Phone_Service"].iloc[0]) == _VALID_RECORD["PhoneService"]

    def test_offer_none_produces_offer_null_column(self):
        """Frontend sends ``Offer: "None"`` as the "no offer" sentinel.
        The model expects an ``Offer_null`` column (the dataset's null
        category, encoded by Polars' ``to_dummies(dummy_na=True)`` at
        training time). The inference path must normalize accordingly."""
        record = {**_VALID_RECORD, "Offer": "None"}
        result = engineer_features_inference(pd.DataFrame([record]))
        assert "Offer_null" in result.columns, (
            f"Expected Offer_null column. Got: {list(result.columns)}"
        )
        assert int(result["Offer_null"].iloc[0]) == 1
        for offer_col in ("Offer_Offer_A", "Offer_Offer_B", "Offer_Offer_C",
                          "Offer_Offer_D", "Offer_Offer_E"):
            assert int(result[offer_col].iloc[0]) == 0
        # The buggy version produced this - must be gone.
        assert "Offer_None" not in result.columns

    def test_internet_type_none_produces_internet_type_null_column(self):
        """Same normalization for InternetType - ``"None"`` -> ``null``."""
        record = {**_VALID_RECORD, "InternetType": "None"}
        result = engineer_features_inference(pd.DataFrame([record]))
        assert "Internet_Type_null" in result.columns, (
            f"Expected Internet_Type_null column. Got: {list(result.columns)}"
        )
        assert int(result["Internet_Type_null"].iloc[0]) == 1
        for col in ("Internet_Type_Cable", "Internet_Type_DSL",
                    "Internet_Type_Fiber_Optic"):
            assert int(result[col].iloc[0]) == 0
        assert "Internet_Type_None" not in result.columns

    def test_senior_citizen_stringified_to_zero_one_columns(self):
        """SeniorCitizen is the one binary field that gets stringified at
        training time, producing ``Senior_Citizen_0`` and ``Senior_Citizen_1``."""
        result = engineer_features_inference(pd.DataFrame([_VALID_RECORD]))
        assert "Senior_Citizen_0" in result.columns
        assert "Senior_Citizen_1" in result.columns
        assert int(result["Senior_Citizen_0"].iloc[0]) == (
            0 if _VALID_RECORD["SeniorCitizen"] == 1 else 1
        )
        assert int(result["Senior_Citizen_1"].iloc[0]) == _VALID_RECORD["SeniorCitizen"]

    def test_produced_columns_match_model_feature_name(self):
        """Parity check: for the high-risk preset, the produced frame
        must match the model's expected 51-column vocabulary exactly
        (no missing, no extras)."""
        import joblib

        from src.feature_engineering import engineer_features_inference

        artifact = joblib.load(
            pathlib.Path(ROOT) / "models" / "churn_model.pkl"
        )
        expected = set(artifact["pipeline"].feature_name_)

        high_risk_preset = {
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
        result = engineer_features_inference(pd.DataFrame([high_risk_preset]))
        produced = set(result.columns)
        missing = expected - produced
        extra = produced - expected
        assert not missing, f"Engineered frame is missing expected columns: {missing}"
        assert not extra, f"Engineered frame produced extra columns: {extra}"

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame()
        result = engineer_features_inference(df)
        assert result.empty or len(result.columns) == 0

    def test_partial_record_does_not_raise(self):
        partial = {
            "Gender": "Female",
            "tenure": 5,
            "MonthlyCharges": 50.0,
        }
        df = pd.DataFrame([partial])
        result = engineer_features_inference(df)
        assert not result.empty
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool


# _classify

class TestClassify:
    def test_high_risk_above_threshold(self):
        result = _classify(0.85)
        assert result["retention_risk"] == "High"
        assert result["prediction"] == 1

    def test_medium_risk(self):
        result = _classify(0.55)
        assert result["retention_risk"] == "Medium"

    def test_low_risk_below_medium_threshold(self):
        result = _classify(0.20)
        assert result["retention_risk"] == "Low"
        assert result["prediction"] == 0

    def test_boundary_high(self):
        result = _classify(0.70)
        assert result["retention_risk"] == "High"

    def test_boundary_medium_low(self):
        result = _classify(0.40)
        assert result["retention_risk"] == "Medium"

    def test_boundary_low(self):
        result = _classify(0.39)
        assert result["retention_risk"] == "Low"

    def test_churn_probability_rounded_to_four_decimals(self):
        result = _classify(0.123456)
        assert result["churn_probability"] == 0.1235

    def test_zero_probability(self):
        result = _classify(0.0)
        assert result["prediction"] == 0
        assert result["retention_risk"] == "Low"

    def test_certain_churn(self):
        result = _classify(1.0)
        assert result["prediction"] == 1
        assert result["retention_risk"] == "High"


# _align_to_model

class TestAlignToModel:
    """Tests for the strict-alignment helper used by /predict and
    /predict/batch. The helper is what guarantees the frame fed to
    LightGBM has the exact column vocabulary the model was fit on."""

    def test_no_op_when_frame_already_matches(self):
        df = pd.DataFrame([{"A": 1, "B": 0}])
        out = _align_to_model(df, ["A", "B"])
        assert list(out.columns) == ["A", "B"]
        assert int(out["A"].iloc[0]) == 1
        assert int(out["B"].iloc[0]) == 0

    def test_missing_columns_are_back_filled_with_zero(self):
        """Empty / partial payloads still return a usable frame - every
        missing expected column is back-filled with 0, the model's
        neutral default."""
        df = pd.DataFrame([{"A": 1}])
        out = _align_to_model(df, ["A", "B", "C"])
        assert list(out.columns) == ["A", "B", "C"]
        assert int(out["A"].iloc[0]) == 1
        assert int(out["B"].iloc[0]) == 0
        assert int(out["C"].iloc[0]) == 0

    def test_extra_columns_are_dropped(self):
        df = pd.DataFrame([{"A": 1, "B": 0, "Extra": 99}])
        out = _align_to_model(df, ["A", "B"])
        assert "Extra" not in out.columns
        assert list(out.columns) == ["A", "B"]

    def test_completely_empty_frame_raises(self):
        """If the engineering pipeline produces zero columns, the
        helper raises - feeding the model an empty frame is a worse
        failure mode than 422."""
        df = pd.DataFrame()  # no columns at all
        with pytest.raises(ValueError, match="Engineered feature frame is empty"):
            _align_to_model(df, ["A", "B"])

    def test_none_expected_returns_frame_as_is(self):
        """When the model has no feature_name_ (degraded mode), the
        helper is a no-op."""
        df = pd.DataFrame([{"A": 1, "B": 0}])
        out = _align_to_model(df, None)
        assert list(out.columns) == ["A", "B"]


# /predict

class TestPredictEndpoint:
    def test_valid_payload_returns_200(self, client: TestClient):
        response = client.post("/predict", json=_VALID_RECORD)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "retention_risk" in data

    def test_valid_payload_types_are_correct(self, client: TestClient):
        response = client.post("/predict", json=_VALID_RECORD)
        data = response.json()
        assert isinstance(data["prediction"], int)
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["retention_risk"], str)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["retention_risk"] in ("High", "Medium", "Low")

    def test_malformed_payload_returns_422(self, client: TestClient):
        payload = {"Gender": 123}  # string expected
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_numeric_type_returns_422(self, client: TestClient):
        payload = {**_VALID_RECORD, "SeniorCitizen": "invalid"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_payload_returns_422(self, client: TestClient):
        """An empty JSON object has no predictive signal - it would
        otherwise silently score an all-zero feature frame, so the
        schema-level guard rejects it with 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_minimal_payload_returns_200(self, client: TestClient):
        minimal = {
            "Gender": "Male",
            "tenure": 1,
            "MonthlyCharges": 50.0,
        }
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200

    def test_senior_citizen_typed_as_int(self, client: TestClient):
        payload = {**_VALID_RECORD, "SeniorCitizen": 1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_senior_citizen_with_float_returns_422(self, client: TestClient):
        payload = {**_VALID_RECORD, "SeniorCitizen": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# /predict/batch

class TestBatchPredictEndpoint:
    def test_batch_with_multiple_records(self, client: TestClient):
        payload = [_VALID_RECORD, _VALID_RECORD]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_returns_422(self, client: TestClient):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 422


# /generate_retention_script (mocked LLM Provider)

class TestRetentionScriptEndpoint:
    """All tests mock api.app.Groq to avoid hitting the live LLM provider API."""

    MOCKED_SCRIPT = (
        "- Lock the 12-month contract at the current price; waive early-termination "
        "fees for 60 days to remove the switching cost.\n"
        "- Open a satisfaction-recovery ticket within 24h; senior CSM calls, "
        "acknowledge the recent friction, attach a 5-10% MRR goodwill credit."
    )

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_valid_request_returns_script(self, mock_llm_cls, client: TestClient):
        mock_instance = _make_llm_mock(return_value=_mock_llm_response(self.MOCKED_SCRIPT))
        mock_llm_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={
                "risk_level": "High",
                "reasons": "Billing confusion, lack of usage.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        expected = "[Action Plan] " + self.MOCKED_SCRIPT
        assert data["script"] == expected
        assert "[Action Plan]" in data["script"]
        assert data["script"].startswith("[Action Plan]")
        assert "[Default Action Plan]" not in data["script"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_mock_was_called_with_correct_model(self, mock_llm_cls, client: TestClient):
        mock_instance = _make_llm_mock(return_value=_mock_llm_response(self.MOCKED_SCRIPT))
        mock_llm_cls.return_value = mock_instance
        client.post(
            "/generate_retention_script",
            json={"risk_level": "Low", "reasons": "No issues reported."},
        )
        mock_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == _expected_routed()["standard"]
        assert len(call_kwargs["messages"]) == 1
        assert "Low" in call_kwargs["messages"][0]["content"] or "low" in call_kwargs["messages"][0]["content"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_llm_exception_falls_back(self, mock_llm_cls, client: TestClient):
        mock_instance = _make_llm_mock(side_effect=Exception("API timeout"))
        mock_llm_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "Service outage."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["script"].startswith("[Default Action Plan]")
        assert "Audit the customer's actual usage" in data["script"]
        assert "satisfaction-recovery workflow" in data["script"]
        assert "[Action Plan]" not in data["script"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_llm_response_exactly_one_tag_prefix(self, mock_llm_cls, client: TestClient):
        mock_instance = _make_llm_mock(
            return_value=_mock_llm_response("Operational note on contract lock-in.")
        )
        mock_llm_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "Low", "reasons": "No issues."},
        )
        data = response.json()
        assert data["script"].count("[") == 1
        assert data["script"].count("]") == 1
        assert "[Action Plan]" in data["script"]
        assert data["script"].endswith("Operational note on contract lock-in.")

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_fallback_script_has_no_provider_model_tag(self, mock_llm_cls, client: TestClient):
        mock_instance = _make_llm_mock(side_effect=RuntimeError("Connection refused"))
        mock_llm_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "Network issues."},
        )
        data = response.json()
        assert data["script"].startswith("[Default Action Plan]")
        assert "[Action Plan]" not in data["script"]
        assert data["script"].count("[") == 1

    @patch("api.app.LLM_PROVIDER_API_KEY", "")
    @patch("api.app.Groq")
    def test_missing_api_key_returns_fallback(self, mock_llm_cls, client: TestClient):
        """With no LLM_PROVIDER_API_KEY the endpoint returns the default action plan
        AND logs a clear ``LLM_PROVIDER_KEY_MISSING`` warning. The LLM provider client
        must not be invoked (no API call without a key)."""
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "Network issues."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["script"].startswith("[Default Action Plan]")
        assert "[Action Plan]" not in data["script"]
        mock_llm_cls.assert_not_called()

    def test_missing_risk_level_returns_422(self, client: TestClient):
        response = client.post(
            "/generate_retention_script",
            json={"reasons": "Some reason."},
        )
        assert response.status_code == 422

    def test_missing_reasons_returns_422(self, client: TestClient):
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High"},
        )
        assert response.status_code == 422


# Round-Trip: /predict -> /generate_retention_script

class TestRoundTrip:
    """Verify that a prediction result can be fed into the retention script
    endpoint in a single mocked flow."""

    @patch("api.app.LLM_PROVIDER_API_KEY", "dummy")
    @patch("api.app.Groq")
    def test_predict_then_generate_script(self, mock_llm_cls, client: TestClient):
        MOCK_SCRIPT = "- Lock the 12-month contract at the current price."
        mock_instance = _make_llm_mock(return_value=_mock_llm_response(MOCK_SCRIPT))
        mock_llm_cls.return_value = mock_instance

        # Step 1: Get a prediction
        pred_resp = client.post("/predict", json=_VALID_RECORD)
        assert pred_resp.status_code == 200
        pred = pred_resp.json()
        assert "retention_risk" in pred
        assert "churn_probability" in pred

        # Step 2: Feed the risk level into the retention script endpoint
        script_resp = client.post(
            "/generate_retention_script",
            json={
                "risk_level": pred["retention_risk"],
                "reasons": (
                    f"Churn probability "
                    f"{(pred['churn_probability'] * 100):.1f}%. "
                    f"Contract: Month-to-Month. Tenure: 12 months."
                ),
            },
        )
        assert script_resp.status_code == 200
        script_data = script_resp.json()
        assert script_data["script"] == "[Action Plan] " + MOCK_SCRIPT

        # Step 3: Verify the mock was called with the correct risk level
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert pred["retention_risk"] in content


# / root

class TestRootEndpoint:
    def test_root_returns_message(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


# /llm/models + X-Provider-* header overrides

@pytest.fixture(autouse=True)
def _clear_routing_cache():
    """Keep model-routing tests hermetic - no cache leaks across tests."""
    from api import app as app_module

    app_module._routing_cache.clear()
    yield
    app_module._routing_cache.clear()


class TestModelRouting:
    """Unit tests for dynamic model discovery / tier routing."""

    def test_score_ranks_parameter_count(self):
        assert _model_intelligence_score("demo-70b") > _model_intelligence_score("demo-9b")
        assert _model_intelligence_score("demo-9b") > _model_intelligence_score("demo-3b-mini")

    def test_score_excludes_non_chat_models(self):
        assert _model_intelligence_score("whisper-large-v3") == float("-inf")
        assert _model_intelligence_score("guard-model") == float("-inf")
        assert _model_intelligence_score("demo-tts-1") == float("-inf")

    def test_score_handles_mixture_of_experts_sizes(self):
        # 8x7B ≈ 56B total parameters - must outrank a dense 9B model.
        assert _model_intelligence_score("mixtral-8x7b") > _model_intelligence_score("demo-9b")

    def test_select_routes_flagship_and_middle(self):
        routed = _select_routed_models(
            ["demo-70b-max", "demo-27b", "demo-9b", "demo-3b-mini"]
        )
        assert routed["high_capacity"] == "demo-70b-max"
        # Standard takes a middle rank, never the flagship or the mini.
        assert routed["standard"] in ("demo-27b", "demo-9b")

    def test_select_with_single_model_routes_it_for_both_tiers(self):
        routed = _select_routed_models(["demo-only-9b"])
        assert routed == {"standard": "demo-only-9b", "high_capacity": "demo-only-9b"}

    def test_select_with_no_eligible_models_returns_empty(self):
        assert _select_routed_models(["whisper-large-v3", "guard-x"]) == {}

    def test_routing_is_deterministic(self):
        ids = ["demo-27b", "demo-70b-max", "demo-9b"]
        assert _select_routed_models(ids) == _select_routed_models(list(reversed(ids)))


class TestLlmProviderOverride:
    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    def test_list_models_returns_tier_catalog(self, client: TestClient):
        """The catalog describes routing tiers only - concrete provider
        model ids must never leak into the UI-facing response."""
        response = client.get("/llm/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data and "default" in data
        assert "standard" in data["models"]
        assert "high_capacity" in data["models"]
        # No raw provider model ids anywhere in the payload.
        raw_ids = re.findall(r"\d+(?:\.\d+)?x\d+b|\d+b\b", json.dumps(data).lower())
        assert raw_ids == []

    @patch("api.app.Groq")
    def test_x_provider_key_header_overrides_env(
        self, mock_llm_cls, client: TestClient
    ):
        """A header-supplied key must win over the env-loaded key."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Header-key script.")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "header-key"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        # The Groq client must have been constructed with the header key,
        # NOT the env value.
        mock_llm_cls.assert_called_with(api_key="header-key")

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_x_provider_model_header_selects_high_capacity_tier(
        self, mock_llm_cls, client: TestClient
    ):
        """The X-Provider-Model header selects the high_capacity tier,
        resolved dynamically from the provider's model list."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("High-capacity model script.")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "key", "X-Provider-Model": "high_capacity"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == _expected_routed()["high_capacity"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_default_request_uses_standard_tier(
        self, mock_llm_cls, client: TestClient
    ):
        """Without an X-Provider-Model header, routine requests route to
        the standard (medium-intelligence) tier."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Standard script.")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "key"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == _expected_routed()["standard"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_discovery_failure_without_pin_returns_fallback(
        self, mock_llm_cls, client: TestClient
    ):
        """If model-list discovery fails and no tier is pinned, the
        endpoint degrades to the labelled fallback instead of guessing."""
        mock_llm_cls.return_value = _make_llm_mock(
            side_effect=RuntimeError("Connection refused")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "key"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        assert response.json()["script"].startswith("[Default Action Plan]")

    @patch.dict(os.environ, {"LLM_HIGH_CAPACITY_MODEL": "pinned-9b"})
    @patch("api.app.Groq")
    def test_env_pin_beats_discovery(self, mock_llm_cls, client: TestClient):
        """An operator-pinned tier id must win over dynamic discovery."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Pinned script.")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "key", "X-Provider-Model": "high_capacity"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "pinned-9b"

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_unknown_model_falls_back_to_default(
        self, mock_llm_cls, client: TestClient
    ):
        """An unrecognised model id must fall back to the default without raising."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Default model script.")
        )
        response = client.post(
            "/generate_retention_script",
            headers={"X-Provider-Key": "key", "X-Provider-Model": "nonexistent"},
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == _expected_routed()["standard"]

    @patch("api.app.LLM_PROVIDER_API_KEY", "")
    def test_no_key_no_header_returns_fallback(self, client: TestClient):
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "SatisfactionScore=2."},
        )
        assert response.status_code == 200
        assert response.json()["script"].startswith("[Default Action Plan]")

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_top_drivers_and_signals_woven_into_prompt(
        self, mock_llm_cls, client: TestClient
    ):
        """Optional top_drivers and risk_signals appear in the LLM prompt
        so the script is grounded in the actual SHAP output."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Grounded script.")
        )
        response = client.post(
            "/generate_retention_script",
            json={
                "risk_level": "High",
                "reasons": "SatisfactionScore=1, tenure=2mo.",
                "top_drivers": [
                    "SatisfactionScore (0.42)",
                    "Tenure_in_Months (0.18)",
                ],
                "risk_signals": ["Satisfaction-recovery outreach"],
                "probability_pct": 78.5,
            },
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert "78.5%" in content
        assert "SatisfactionScore (0.42)" in content
        assert "Satisfaction-recovery outreach" in content

    @patch("api.app.LLM_PROVIDER_API_KEY", "env-key")
    @patch("api.app.Groq")
    def test_optional_fields_can_be_omitted(
        self, mock_llm_cls, client: TestClient
    ):
        """The original two-field payload still works (back-compat)."""
        mock_llm_cls.return_value = _make_llm_mock(
            return_value=_mock_llm_response("Legacy script.")
        )
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "Low", "reasons": "Loyal customer."},
        )
        assert response.status_code == 200
        call_kwargs = mock_llm_cls.return_value.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert "Loyal customer" in content
        assert "Top Tree-SHAP drivers" not in content


class TestCorsOrigins:
    """CORS must be env-driven so production deployments can whitelist
    their own FE host without editing source."""

    def test_default_origins_present(self, client: TestClient):
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    @patch.dict("os.environ", {"CORS_ORIGINS": "https://churn.example.com,https://staging.example.com"})
    def test_env_override_extends_origins(self):
        import importlib

        from api import app as app_module

        importlib.reload(app_module)
        try:
            client = TestClient(app_module.app)
            response = client.options(
                "/predict",
                headers={
                    "Origin": "https://churn.example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.headers.get("access-control-allow-origin") == "https://churn.example.com"
        finally:
            importlib.reload(app_module)
