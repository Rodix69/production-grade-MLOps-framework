
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, Annotated
import pandas as pd
import mlflow.pyfunc
import logging
 
# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
 
# ── App & model ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.1",
    description="Predicts customer churn probability for a telecom provider.",
)
 
logger.info("Loading MLflow model...")
model = mlflow.pyfunc.load_model("model")
sk_model = model._model_impl.sklearn_model
EXPECTED_COLUMNS = list(sk_model.feature_names_in_)
logger.info(f"Model loaded. Expecting {len(EXPECTED_COLUMNS)} features.")
 
 
# ── Request schema ───────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
 
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "tenure": 12,
                        "MonthlyCharges": 65.5,
                        "TotalCharges": 786.0,
                    }
                }
            ]
        }
    }
 
 
# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def home():
    return {
        "message": "Telecom Churn Prediction API is running.",
        "version": "1.1",
        "expected_features": EXPECTED_COLUMNS,
        "total_features": len(EXPECTED_COLUMNS),
    }
 
 
@app.get("/health", tags=["Health"])
def health():
    """Liveness probe — used by Docker, Kubernetes, and load balancers."""
    return {"status": "healthy"}
 
 
@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe — confirms model is loaded and ready to serve."""
    if sk_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ready", "model": "telecom-churn-logreg"}
 
 
PREDICT_EXAMPLE = {
    "churn_customer": {
        "summary": "High-risk customer",
        "value": {
            "features": {
                "tenure": 2,
                "MonthlyCharges": 90.5,
                "TotalCharges": 181.0,
            }
        },
    },
    "stable_customer": {
        "summary": "Low-risk customer",
        "value": {
            "features": {
                "tenure": 48,
                "MonthlyCharges": 45.0,
                "TotalCharges": 2160.0,
            }
        },
    },
}
 
 
@app.post("/predict", tags=["Prediction"])
def predict(
    request: Annotated[PredictionRequest, Body(openapi_examples=PREDICT_EXAMPLE)],
):
    """
    Predict churn for a single customer.
 
    - Missing features are filled with **0** (logged as a warning).
    - Returns binary prediction, churn probability, and risk label.
    """
    try:
        # ── Input validation ─────────────────────────────────────────────────
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in request.features]
        unknown_cols = [c for c in request.features if c not in EXPECTED_COLUMNS]
 
        if missing_cols:
            logger.warning(f"Missing columns filled with 0: {missing_cols}")
        if unknown_cols:
            logger.warning(f"Unknown columns ignored: {unknown_cols}")
 
        # ── Build input dataframe ────────────────────────────────────────────
        data = {col: 0 for col in EXPECTED_COLUMNS}
        data.update({k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS})
        input_df = pd.DataFrame([data])[EXPECTED_COLUMNS]
 
        # ── Predict ──────────────────────────────────────────────────────────
        prediction = model.predict(input_df)
        proba = sk_model.predict_proba(input_df)[0][1]
 
        # ── Risk label ───────────────────────────────────────────────────────
        if proba >= 0.70:
            risk = "High"
        elif proba >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"
 
        result = {
            "prediction": int(prediction[0]),
            "churn_status": "Churn" if int(prediction[0]) == 1 else "No Churn",
            "churn_probability": round(float(proba), 4),
            "risk_level": risk,
            "missing_features_filled": missing_cols,
        }
 
        logger.info(f"Prediction result: {result}")
        return result
 
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))