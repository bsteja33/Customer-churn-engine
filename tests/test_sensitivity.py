"""Input-sensitivity and integrity tests for the Churn Prediction API.

These tests prove the endpoints compute real, input-dependent outputs
(no hardcoded, cached, or cross-request-stale results) and that the
SHAP explainability layer is mathematically consistent with the model's
probabilities. The real model artifact (models/churn_model.pkl) is used;
all LLM provider calls are mocked.
"""

import json
import logging
import math
import pathlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _post_raw(client: TestClient, payload) -> object:
    """POST a payload using Python's own JSON encoder so NaN/Infinity
    reach the server (the TestClient's strict encoder would reject them
    client-side, before the API's allow_inf_nan=False guard runs)."""
    return client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from api.app import (  # noqa: E402
    JsonFormatter,
    _align_to_model,
    _discover_routed_models,
    _engineer_features,
    _select_routed_models,
    app,
    get_routed_models,
)
from src.explain import _compute_raw_contribs  # noqa: E402

# Two opposite customer profiles. Every assertion in this module is
# derived from what the live model does with them; nothing is hardcoded
# from a previous run.

HIGH_RISK_PROFILE = {
    "Gender": "Female",
    "SeniorCitizen": 1,
    "Partner": 0,
    "Dependents": 1,
    "tenure": 1,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": "Month-to-Month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharges": 118.75,
    "TotalCharges": 118.75,
    "Married": 0,
    "NumberOfDependents": 1,
    "NumberOfReferrals": 0,
    "SatisfactionScore": 1,
    "InternetType": "Fiber Optic",
    "Offer": "Offer E",
    "Age": 80,
    "AvgMonthlyGBDownload": 0,
    "AvgMonthlyLongDistanceCharges": 25.0,
    "CLTV": 2000,
    "Under30": 0,
    "UnlimitedData": 0,
    "StreamingMusic": 0,
    "ReferredAFriend": 0,
    "TotalRefunds": 50.0,
    "TotalExtraDataCharges": 0,
    "TotalLongDistanceCharges": 25.0,
    "TotalRevenue": 200.0,
}

LOW_RISK_PROFILE = {
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 72,
    "PhoneService": 1,
    "MultipleLines": 1,
    "InternetService": 1,
    "OnlineSecurity": 1,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": "Two Year",
    "PaperlessBilling": 0,
    "PaymentMethod": "Credit Card",
    "MonthlyCharges": 25.0,
    "TotalCharges": 1800.0,
    "Married": 1,
    "NumberOfDependents": 0,
    "NumberOfReferrals": 5,
    "SatisfactionScore": 5,
    "InternetType": "DSL",
    "Offer": "None",
    "Age": 35,
    "AvgMonthlyGBDownload": 10,
    "AvgMonthlyLongDistanceCharges": 5.0,
    "CLTV": 6500,
    "Under30": 0,
    "UnlimitedData": 1,
    "StreamingMusic": 1,
    "ReferredAFriend": 1,
    "TotalRefunds": 0.0,
    "TotalExtraDataCharges": 0,
    "TotalLongDistanceCharges": 5.0,
    "TotalRevenue": 4000.0,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _predict(client: TestClient, profile: dict) -> dict:
    response = client.post("/predict", json=profile)
    assert response.status_code == 200, response.text
    return response.json()


# Input sensitivity: outputs must track inputs


class TestInputSensitivity:
    def test_extreme_profiles_produce_strictly_ordered_scores(self, client):
        high = _predict(client, HIGH_RISK_PROFILE)
        low = _predict(client, LOW_RISK_PROFILE)
        assert 0.0 <= high["churn_probability"] <= 1.0
        assert 0.0 <= low["churn_probability"] <= 1.0
        assert high["churn_probability"] > low["churn_probability"] + 0.1

    def test_identical_input_is_deterministic_but_distinct_input_differs(self, client):
        first = _predict(client, HIGH_RISK_PROFILE)
        second = _predict(client, dict(HIGH_RISK_PROFILE))
        other = _predict(client, LOW_RISK_PROFILE)
        assert first == second
        assert first["churn_probability"] != other["churn_probability"]

    def test_score_varies_monotonically_with_satisfaction(self, client):
        base = dict(HIGH_RISK_PROFILE)
        scores = []
        for satisfaction in (1, 3, 5):
            base["SatisfactionScore"] = satisfaction
            scores.append(_predict(client, base)["churn_probability"])
        assert scores[0] > scores[2]
        assert len(set(scores)) == 3

    def test_score_varies_with_contract_term(self, client):
        base = dict(HIGH_RISK_PROFILE)
        month_to_month = _predict(client, base)["churn_probability"]
        base["Contract"] = "Two Year"
        two_year = _predict(client, base)["churn_probability"]
        assert month_to_month > two_year

    def test_full_payload_stream_never_collapses_to_one_value(self, client):
        # Empirically the served artifact has a knife-edge boundary
        # around Satisfaction Score 3-4 (tenure/contract deltas stay in
        # the 4th decimal), so the sweep varies that field. The assert
        # guards against a fully-collapsed (single-value) endpoint.
        base = dict(HIGH_RISK_PROFILE)
        probabilities = []
        for satisfaction in (1, 2, 3, 4, 5):
            base["SatisfactionScore"] = satisfaction
            probabilities.append(_predict(client, base)["churn_probability"])
        assert len(set(probabilities)) >= 3
        assert probabilities[0] > probabilities[-1]

    def test_prediction_flag_and_tier_agree_with_probability(self, client):
        for profile in (HIGH_RISK_PROFILE, LOW_RISK_PROFILE):
            data = _predict(client, profile)
            p = data["churn_probability"]
            assert data["prediction"] == int(p >= 0.5)
            if p >= 0.7:
                assert data["retention_risk"] == "High"
            elif p >= 0.4:
                assert data["retention_risk"] == "Medium"
            else:
                assert data["retention_risk"] == "Low"

    def test_batch_matches_individual_predictions_in_order(self, client):
        batch = client.post(
            "/predict/batch", json=[HIGH_RISK_PROFILE, LOW_RISK_PROFILE]
        )
        assert batch.status_code == 200, batch.text
        results = batch.json()["results"]
        single_high = _predict(client, HIGH_RISK_PROFILE)
        single_low = _predict(client, LOW_RISK_PROFILE)
        assert len(results) == 2
        assert results[0]["churn_probability"] == pytest.approx(
            single_high["churn_probability"], abs=1e-9
        )
        assert results[1]["churn_probability"] == pytest.approx(
            single_low["churn_probability"], abs=1e-9
        )

    def test_batch_high_risk_count_matches_results(self, client):
        data = client.post(
            "/predict/batch",
            json=[HIGH_RISK_PROFILE, LOW_RISK_PROFILE, LOW_RISK_PROFILE],
        ).json()
        counted = sum(1 for r in data["results"] if r["retention_risk"] == "High")
        assert data["high_risk_count"] == counted
        assert data["total_records"] == 3


# SHAP integrity: attributions must be real math, not decoration


class TestShapIntegrity:
    @pytest.fixture(scope="class")
    @staticmethod
    def engineered(client):
        pipeline = client.app.state.model
        expected = client.app.state.expected_features
        frames = {}
        for name, profile in (
            ("high", HIGH_RISK_PROFILE),
            ("low", LOW_RISK_PROFILE),
        ):
            df = _align_to_model(_engineer_features(pd.DataFrame([profile])), expected)
            frames[name] = (pipeline, df)
        return frames

    def test_contributions_sum_to_the_model_logit(self, engineered):
        pipeline, df = engineered["high"]
        raw = _compute_raw_contribs(pipeline, df)
        assert raw is not None
        bias = float(raw[0, -1])
        per_feature = raw[0, :-1]
        p = float(pipeline.predict_proba(df)[0][1])
        logit = math.log(p / (1.0 - p))
        assert per_feature.sum() + bias == pytest.approx(logit, abs=1e-6)

    def test_feature_importance_magnitudes_are_sorted_descending(self, client):
        items = _predict(client, HIGH_RISK_PROFILE)["feature_importance"]
        assert items, "expected a populated SHAP panel"
        magnitudes = [item["magnitude"] for item in items]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(m > 0 for m in magnitudes)
        assert all(item["direction"] in ("up", "down") for item in items)

    def test_satisfaction_contribution_moves_with_satisfaction(self, engineered):
        def _signed_contribution(pipeline, df):
            raw = _compute_raw_contribs(pipeline, df)
            column = df.columns.get_loc("Satisfaction_Score")
            return float(raw[0, column])

        high_pipeline, high_df = engineered["high"]
        low_pipeline, low_df = engineered["low"]
        low_satisfaction = _signed_contribution(high_pipeline, high_df)
        high_satisfaction = _signed_contribution(low_pipeline, low_df)
        assert low_satisfaction > high_satisfaction

    def test_top_drivers_differ_between_opposite_profiles(self, client):
        high = _predict(client, HIGH_RISK_PROFILE)["feature_importance"]
        low = _predict(client, LOW_RISK_PROFILE)["feature_importance"]
        top_high = [item["feature"] for item in high[:4]]
        top_low = [item["feature"] for item in low[:4]]
        assert top_high != top_low
        assert all(isinstance(item["value"], (int, float, str)) for item in high)


# Validation edge cases


class TestValidationEdgeCases:
    def test_nan_monthly_charges_is_rejected(self, client):
        payload = dict(HIGH_RISK_PROFILE)
        payload["MonthlyCharges"] = float("nan")
        response = _post_raw(client, payload)
        assert response.status_code == 422

    def test_infinity_total_charges_is_rejected(self, client):
        payload = dict(LOW_RISK_PROFILE)
        payload["TotalCharges"] = float("inf")
        response = _post_raw(client, payload)
        assert response.status_code == 422

    def test_wrong_type_tenure_is_rejected_in_strict_mode(self, client):
        payload = dict(HIGH_RISK_PROFILE)
        payload["tenure"] = "12"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_unknown_extra_field_is_rejected(self, client):
        payload = dict(HIGH_RISK_PROFILE)
        payload["SneakyExtraField"] = True
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_null_valued_fields_are_ignored_not_crashed(self, client):
        payload = {k: None for k in HIGH_RISK_PROFILE}
        payload["tenure"] = 12
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0.0 <= response.json()["churn_probability"] <= 1.0


# LLM grounding: prompts must track their structured input


class TestLLMGrounding:
    def test_distinct_payloads_produce_distinct_grounded_prompts(self, client):
        captured = []

        def _capture(**kwargs):
            captured.append(kwargs["messages"][0]["content"])
            msg = MagicMock()
            msg.content = "Personalized plan."
            return MagicMock(choices=[MagicMock(message=msg)])

        instance = MagicMock()
        instance.chat.completions.create.side_effect = _capture
        instance.models.list.return_value = [
            SimpleNamespace(id="demo-large-max-70b"),
            SimpleNamespace(id="demo-small-9b"),
        ]
        with patch("api.app.Groq", return_value=instance):
            first = client.post(
                "/generate_retention_script",
                headers={"X-Provider-Key": "gsk_test_grounding_a"},
                json={
                    "risk_level": "High",
                    "reasons": "Customer cited billing confusion.",
                    "probability_pct": 87.5,
                    "top_drivers": ["Contract: Month-to-Month", "Satisfaction Score"],
                },
            )
            second = client.post(
                "/generate_retention_script",
                headers={"X-Provider-Key": "gsk_test_grounding_a"},
                json={
                    "risk_level": "Low",
                    "reasons": "Customer renewed a two-year contract.",
                    "probability_pct": 4.2,
                    "top_drivers": ["Tenure in Months", "Contract: Two Year"],
                },
            )

        assert first.status_code == 200 and second.status_code == 200
        assert first.json()["script"] == "[Action Plan] Personalized plan."
        assert len(captured) == 2
        assert captured[0] != captured[1]
        assert "High" in captured[0] and "87.5%" in captured[0]
        assert "Contract: Month-to-Month" in captured[0]
        assert "Low" in captured[1] and "4.2%" in captured[1]
        assert "Tenure in Months" in captured[1]

    def test_empty_llm_content_falls_back_to_labelled_script(self, client):
        instance = MagicMock()
        instance.models.list.return_value = [SimpleNamespace(id="demo-small-9b")]
        msg = MagicMock()
        msg.content = "   "
        instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        with patch("api.app.Groq", return_value=instance):
            response = client.post(
                "/generate_retention_script",
                headers={"X-Provider-Key": "gsk_test_empty_b"},
                json={
                    "risk_level": "Medium",
                    "reasons": "Usage dropped over the last cycle.",
                    "probability_pct": 50.0,
                },
            )
        assert response.status_code == 200
        assert response.json()["script"].startswith("[Default Action Plan]")

    def test_different_llm_outputs_are_passed_through_verbatim(self, client):
        instance = MagicMock()
        instance.models.list.return_value = [SimpleNamespace(id="demo-small-9b")]

        responses = iter(["Offer 20% discount.", "Offer a free upgrade."])
        msg = MagicMock()

        def _create(**kwargs):
            msg.content = next(responses)
            return MagicMock(choices=[MagicMock(message=msg)])

        instance.chat.completions.create.side_effect = _create
        with patch("api.app.Groq", return_value=instance):
            first = client.post(
                "/generate_retention_script",
                headers={"X-Provider-Key": "gsk_test_verbatim_c"},
                json={"risk_level": "High", "reasons": "Billing complaints."},
            )
            second = client.post(
                "/generate_retention_script",
                headers={"X-Provider-Key": "gsk_test_verbatim_c"},
                json={"risk_level": "High", "reasons": "Billing complaints."},
            )
        assert first.json()["script"] == "[Action Plan] Offer 20% discount."
        assert second.json()["script"] == "[Action Plan] Offer a free upgrade."


# JSON log formatter: exception-safe structured logging


class TestJsonLogFormatter:
    def _record(self, msg, extra=None, exc=None):
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=1, msg=msg, args=(), exc_info=exc,
        )
        for key, value in (extra or {}).items():
            setattr(record, key, value)
        return record

    def test_emits_valid_json_with_extras_and_exception(self):
        formatter = JsonFormatter()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            record = self._record(
                "LLM_GENERATION_FAILED",
                extra={"model": "demo-small-9b", "attempt": 3},
                exc=sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "LLM_GENERATION_FAILED"
        assert parsed["model"] == "demo-small-9b"
        assert parsed["attempt"] == 3
        assert "boom" in parsed["exception"]

    def test_never_raises_on_hostile_extra_values(self):
        formatter = JsonFormatter()
        record = self._record("WEIRD", extra={"bad": {1, 2}})
        rendered = formatter.format(record)
        assert "WEIRD" in rendered


# Model discovery: real SDK payload shapes


class TestDiscoveryPayloadShapes:
    def test_paginated_data_attribute_is_read(self):
        paginated = SimpleNamespace(
            data=[
                SimpleNamespace(id="demo-large-max-70b"),
                SimpleNamespace(id="demo-mid-27b"),
                SimpleNamespace(id="demo-small-9b"),
            ]
        )
        with patch("api.app._get_llm_client") as get_client:
            get_client.return_value.models.list.return_value = paginated
            routed = _discover_routed_models("sk-shape-test")
        expected = _select_routed_models(
            ["demo-large-max-70b", "demo-mid-27b", "demo-small-9b"]
        )
        assert routed == expected

    def test_tuple_cursor_shape_is_supported_defensively(self):
        items = [
            SimpleNamespace(id="demo-large-max-70b"),
            SimpleNamespace(id="demo-small-9b"),
        ]
        listing = (items, "cursor-abc")
        with patch("api.app._get_llm_client") as get_client:
            get_client.return_value.models.list.return_value = listing
            routed = _discover_routed_models("sk-shape-test")
        expected = _select_routed_models(["demo-large-max-70b", "demo-small-9b"])
        assert routed == expected


class TestRoutingCache:
    key = "sk-cache-test"

    @staticmethod
    def _cache():
        # ``test_api.py`` calls importlib.reload(api.app), which
        # re-executes the module in place and re-binds module-level
        # names (including ``_routing_cache``) to fresh objects. A
        # top-level ``from api.app import _routing_cache`` would keep
        # pointing at the pre-reload dict, so resolve the attribute
        # through the live module on every access.
        import api.app as mod

        return mod._routing_cache

    def setup_method(self):
        self._cache().pop(self.key, None)

    def teardown_method(self):
        self._cache().pop(self.key, None)

    def test_fresh_cache_short_circuits_discovery(self):
        self._cache()[self.key] = {
            "models": {
                "standard": "demo-small-9b",
                "high_capacity": "demo-large-max-70b",
            },
            "expires": time.monotonic() + 60,
        }
        with patch("api.app._discover_routed_models") as discover:
            result = get_routed_models(self.key)
        assert result == {
            "standard": "demo-small-9b",
            "high_capacity": "demo-large-max-70b",
        }
        discover.assert_not_called()

    def test_stale_cache_is_served_when_discovery_fails(self):
        self._cache()[self.key] = {
            "models": {"standard": "demo-mid-27b", "high_capacity": "demo-mid-27b"},
            "expires": time.monotonic() - 1,
        }
        with patch(
            "api.app._discover_routed_models",
            side_effect=RuntimeError("provider down"),
        ):
            result = get_routed_models(self.key)
        assert result == {"standard": "demo-mid-27b", "high_capacity": "demo-mid-27b"}

    def test_discovery_failure_without_cache_returns_none(self):
        with patch(
            "api.app._discover_routed_models",
            side_effect=RuntimeError("provider down"),
        ):
            assert get_routed_models(self.key) is None
