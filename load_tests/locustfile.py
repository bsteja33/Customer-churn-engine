import random
from locust import HttpUser, task, between


def _random_payload():
    genders = ["Male", "Female"]
    contracts = ["Month-to-Month", "One Year", "Two Year"]
    internet_types = ["DSL", "Fiber Optic", "Cable", "None"]
    payments = ["Bank Withdrawal", "Credit Card", "Mailed Check"]
    offers = ["None", "Offer A", "Offer B", "Offer C"]

    return {
        "Gender": random.choice(genders),
        "SeniorCitizen": random.randint(0, 1),
        "Partner": random.randint(0, 1),
        "Dependents": random.randint(0, 1),
        "tenure": random.randint(1, 72),
        "PhoneService": random.randint(0, 1),
        "MultipleLines": random.randint(0, 1),
        "InternetService": random.randint(0, 1),
        "OnlineSecurity": random.randint(0, 1),
        "OnlineBackup": random.randint(0, 1),
        "DeviceProtection": random.randint(0, 1),
        "TechSupport": random.randint(0, 1),
        "StreamingTV": random.randint(0, 1),
        "StreamingMovies": random.randint(0, 1),
        "Contract": random.choice(contracts),
        "PaperlessBilling": random.randint(0, 1),
        "PaymentMethod": random.choice(payments),
        "MonthlyCharges": round(random.uniform(20, 120), 2),
        "TotalCharges": round(random.uniform(100, 6000), 2),
        "Married": random.randint(0, 1),
        "NumberOfDependents": random.randint(0, 5),
        "NumberOfReferrals": random.randint(0, 10),
        "SatisfactionScore": random.randint(1, 5),
        "InternetType": random.choice(internet_types),
        "Offer": random.choice(offers),
        "Age": random.randint(18, 80),
        "AvgMonthlyGBDownload": random.randint(0, 500),
        "AvgMonthlyLongDistanceCharges": round(random.uniform(0, 50), 2),
        "CLTV": random.randint(1000, 8000),
        "Under30": random.randint(0, 1),
        "UnlimitedData": random.randint(0, 1),
        "StreamingMusic": random.randint(0, 1),
        "ReferredAFriend": random.randint(0, 1),
        "TotalRefunds": round(random.uniform(0, 50), 2),
        "TotalExtraDataCharges": random.randint(0, 20),
        "TotalLongDistanceCharges": round(random.uniform(0, 300), 2),
        "TotalRevenue": round(random.uniform(100, 6000), 2),
    }


class CustomerServiceAgent(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(1)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(3)
    def predict_churn(self):
        payload = _random_payload()
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code == 422:
                resp.failure(f"Validation error: {resp.text}")
            elif resp.status_code == 500:
                resp.failure(f"Server error: {resp.text}")
            elif resp.status_code != 200:
                resp.failure(f"Unexpected status: {resp.status_code}")
