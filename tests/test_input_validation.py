"""Regression tests for server-side semantic input validation and
deployment version reporting.

Contract (introduced with the drift-tracking / validation pass):

* ``/health`` reports the deployed commit as ``version``.
* ``/predict`` and ``/predict/batch`` return **422** for empty payloads,
  out-of-range semantics (SatisfactionScore outside 1-5, negative
  tenure/charges, ...), and NaN/Infinity.
* ``/predict`` returns **429** when its 10/minute limit is exceeded
  (limiter is force-enabled locally for exactly one test).

The real model artifact is used; no LLM calls are made.
"""

import json
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from api.app import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    """Lifespan-aware TestClient (loads the real model artifact)."""
    with TestClient(app) as c:
        yield c


# Valid minimal records exercising different field subsets.
VALID_MINIMAL = {"SatisfactionScore": 3, "tenure": 12, "MonthlyCharges": 70.0}
VALID_SINGLE_FIELD = {"SatisfactionScore": 5}


class TestHealthVersion:
    def test_health_reports_version_key(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        # Local runs resolve to "unknown" or a short sha; never blank.
        assert data["version"] == "unknown" or len(data["version"]) >= 4

    def test_health_reports_model_state(self, client):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)
        assert data["status"] in ("healthy", "degraded")


class TestEmptyPayloads:
    def test_empty_object_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_all_none_fields_returns_422(self, client):
        # Every field explicitly null is semantically identical to {}.
        response = client.post("/predict", json={"SatisfactionScore": None})
        assert response.status_code == 422

    def test_empty_batch_returns_422(self, client):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 422

    def test_batch_with_one_empty_record_returns_422(self, client):
        response = client.post("/predict/batch", json=[{}, VALID_MINIMAL])
        assert response.status_code == 422


class TestSemanticRanges:
    @pytest.mark.parametrize("score", [0, 6, -1, 99])
    def test_satisfaction_outside_1_to_5_returns_422(self, client, score):
        response = client.post("/predict", json={"SatisfactionScore": score})
        assert response.status_code == 422

    @pytest.mark.parametrize("score", [1, 5])
    def test_satisfaction_boundaries_return_200(self, client, score):
        response = client.post("/predict", json={"SatisfactionScore": score})
        assert response.status_code == 200

    @pytest.mark.parametrize("field", ["tenure", "MonthlyCharges", "TotalCharges",
                                       "TotalRevenue", "TotalRefunds", "CLTV",
                                       "Age", "NumberOfDependents",
                                       "NumberOfReferrals"])
    def test_negative_values_return_422(self, client, field):
        response = client.post("/predict", json={field: -1})
        assert response.status_code == 422

    def test_absurd_tenure_returns_422(self, client):
        response = client.post("/predict", json={"tenure": 60_000})
        assert response.status_code == 422

    def test_binary_flag_rejects_out_of_range(self, client):
        response = client.post("/predict", json={"SeniorCitizen": 2})
        assert response.status_code == 422

    def test_partial_payload_still_returns_200(self, client):
        """Partial payloads remain valid (zero-fill contract preserved)."""
        response = client.post("/predict", json=VALID_MINIMAL)
        assert response.status_code == 200
        assert response.json()["retention_risk"] in ("High", "Medium", "Low")

    def test_single_field_payload_returns_200(self, client):
        response = client.post("/predict", json=VALID_SINGLE_FIELD)
        assert response.status_code == 200


class TestNaNAndInfinity:
    @staticmethod
    def _post_raw(client, payload):
        # Python's json encoder emits NaN/Infinity literals that the
        # TestClient's strict encoder would reject client-side.
        return client.post(
            "/predict",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

    def test_nan_satisfaction_returns_422(self, client):
        response = self._post_raw(client, {"SatisfactionScore": float("nan")})
        assert response.status_code == 422

    def test_nan_charges_returns_422(self, client):
        response = self._post_raw(client, {"MonthlyCharges": float("nan")})
        assert response.status_code == 422

    def test_infinity_charges_returns_422(self, client):
        response = self._post_raw(client, {"MonthlyCharges": float("inf")})
        assert response.status_code == 422

    def test_valid_after_rejections(self, client):
        """Rate limiter is disabled by conftest; rejections must not
        poison subsequent valid requests (no cross-request state)."""
        assert client.post("/predict", json=VALID_MINIMAL).status_code == 200


class TestRateLimiting:
    def test_predict_returns_429_after_10_per_minute(self, client):
        """The /predict limit is 10/minute; the 11th request in a
        rolling minute must get 429. The limiter is force-enabled here
        because conftest disables it globally (production default is on)."""
        app.state.limiter.enabled = True
        try:
            statuses = [
                client.post("/predict", json=VALID_MINIMAL).status_code
                for _ in range(12)
            ]
            assert statuses.count(200) == 10, statuses
            assert statuses.count(429) == 2, statuses
            # Documented slowapi error shape.
            error = client.post("/predict", json=VALID_MINIMAL).json()
            assert "error" in error
        finally:
            app.state.limiter.enabled = False
