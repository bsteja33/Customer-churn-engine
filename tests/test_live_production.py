"""Live production round-trip tests against the deployed HF Space.

Skipped entirely unless RUN_LIVE_TESTS=1 is set. These never run in CI;
they are a manual verification that the deployed Space serves real,
input-dependent predictions and grounded LLM scripts end to end.
"""

import os
import sys
import pathlib

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS", "").strip().lower() != "1",
    reason="live production probe; set RUN_LIVE_TESTS=1 to run",
)

BASE_URL = os.environ.get(
    "LIVE_BASE_URL", "https://indianfincher-churn-engine.hf.space"
)

_HIGH_RISK = {
    "Contract": "Month-to-Month",
    "tenure": 1,
    "SatisfactionScore": 1,
    "MonthlyCharges": 118.75,
    "TotalCharges": 118.75,
    "InternetType": "Fiber Optic",
    "PaymentMethod": "Bank Withdrawal",
    "Offer": "Offer E",
    "Age": 80,
    "SeniorCitizen": 1,
    "Married": 0,
    "NumberOfDependents": 1,
    "TechSupport": 0,
    "OnlineSecurity": 0,
    "PaperlessBilling": 1,
    "Gender": "Female",
    "ReferredAFriend": 0,
    "TotalRefunds": 50.0,
    "TotalRevenue": 200.0,
}

_LOW_RISK = {
    "Contract": "Two Year",
    "tenure": 72,
    "SatisfactionScore": 5,
    "MonthlyCharges": 25.0,
    "TotalCharges": 1800.0,
    "InternetType": "DSL",
    "PaymentMethod": "Credit Card",
    "Offer": "None",
    "Age": 35,
    "SeniorCitizen": 0,
    "Married": 1,
    "NumberOfDependents": 0,
    "TechSupport": 1,
    "OnlineSecurity": 1,
    "PaperlessBilling": 0,
    "Gender": "Male",
    "ReferredAFriend": 1,
    "TotalRefunds": 0.0,
    "TotalRevenue": 4000.0,
}


@pytest.fixture(scope="module")
def live_client():
    try:
        import requests
    except ImportError:
        pytest.skip("requests not installed")
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"
    yield session
    session.close()


def test_live_health_is_healthy(live_client):
    response = live_client.get(f"{BASE_URL}/health", timeout=60)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_live_predictions_differ_for_opposite_profiles(live_client):
    high = live_client.post(f"{BASE_URL}/predict", json=_HIGH_RISK, timeout=60).json()
    low = live_client.post(f"{BASE_URL}/predict", json=_LOW_RISK, timeout=60).json()
    assert 0.0 <= high["churn_probability"] <= 1.0
    assert 0.0 <= low["churn_probability"] <= 1.0
    assert high["churn_probability"] > low["churn_probability"]
    assert high["retention_risk"] != low["retention_risk"]


def test_live_llm_script_is_generated_and_grounded(live_client):
    api_key = os.environ.get("LLM_PROVIDER_API_KEY", "")
    if not api_key:
        pytest.skip("LLM_PROVIDER_API_KEY not set; cannot probe the live LLM tier")
    response = live_client.post(
        f"{BASE_URL}/generate_retention_script",
        headers={"X-Provider-Key": api_key},
        json={
            "risk_level": "High",
            "probability_pct": 87.5,
            "top_drivers": ["Contract: Month-to-Month", "Satisfaction Score"],
        },
        timeout=60,
    )
    assert response.status_code == 200
    script = response.json()["script"]
    assert script.startswith("[Action Plan]")
    assert len(script) > len("[Action Plan] ")
