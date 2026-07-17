"""
test_app.py
-----------
Basic test cases jo GitHub Actions CI pipeline mein run hongi.
Agar ye fail hui to CD (deployment) step bilkul nahi chalega -
yehi proper CI/CD ka principle hai.

Run locally:
    pytest -v
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict_valid_input():
    response = client.post("/predict", json={"size_sqft": 1500})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price_lakh" in body
    assert isinstance(body["predicted_price_lakh"], float)


def test_predict_invalid_input():
    # size_sqft manfi (negative) nahi ho sakta - validation check
    response = client.post("/predict", json={"size_sqft": -100})
    assert response.status_code == 422


def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
