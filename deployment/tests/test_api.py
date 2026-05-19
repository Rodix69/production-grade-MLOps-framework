"""
Tests for Telecom Churn Prediction API
Run with: pytest tests/test_api.py -v
"""
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ── Mock the model before importing app ──────────────────────────────────────
mock_sk_model = MagicMock()
mock_sk_model.feature_names_in_ = np.array(["tenure", "MonthlyCharges", "TotalCharges"])
mock_sk_model.predict_proba.return_value = np.array([[0.3, 0.7]])

mock_model = MagicMock()
mock_model.predict.return_value = np.array([1])
mock_model._model_impl.sklearn_model = mock_sk_model

with patch("mlflow.pyfunc.load_model", return_value=mock_model):
    from app import app

client = TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "expected_features" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_churn():
    payload = {
        "features": {
            "tenure": 2,
            "MonthlyCharges": 90.0,
            "TotalCharges": 180.0,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert "churn_status" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_missing_features():
    """Missing features should be filled with 0, not cause a crash."""
    payload = {"features": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "missing_features_filled" in response.json()


def test_predict_unknown_features_ignored():
    """Unknown features should be silently ignored."""
    payload = {
        "features": {
            "tenure": 5,
            "UNKNOWN_FEATURE_XYZ": 999,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_invalid_payload():
    """Non-dict features should return 422."""
    response = client.post("/predict", json={"features": "not_a_dict"})
    assert response.status_code == 422
