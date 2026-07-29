from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Annotated
import pandas as pd
import mlflow.pyfunc
import logging
import sqlite3
import json
import time
import os
import psutil
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.2",
    description="Predicts customer churn probability for a telecom provider.",
)

# Model loading
logger.info("Loading MLflow model...")
model = mlflow.pyfunc.load_model("model")
sk_model = model._model_impl.sklearn_model
EXPECTED_COLUMNS = list(sk_model.feature_names_in_)
logger.info(f"Model loaded. Expecting {len(EXPECTED_COLUMNS)} features.")

# Prometheus metrics
REQUEST_COUNT     = Counter("prediction_requests_total", "Total prediction requests")
ERROR_COUNT       = Counter("prediction_errors_total", "Total prediction errors")
CHURN_COUNT       = Counter("churn_predictions_total", "Total churn predictions")
NO_CHURN_COUNT    = Counter("no_churn_predictions_total", "Total no-churn predictions")
LATENCY           = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
MISSING_FEATURES  = Histogram("missing_features_per_request", "Missing features per request", buckets=[0,1,2,5,10,15,20,27])
CHURN_PROBABILITY = Histogram("churn_probability", "Churn probability distribution", buckets=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

# SQLite setup
DB_PATH = os.environ.get("DB_PATH", "monitoring/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            prediction       INTEGER,
            churn_status     TEXT,
            churn_probability REAL,
            risk_level       TEXT,
            missing_count    INTEGER,
            latency_ms       REAL,
            model_version    TEXT,
            input_features   TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_to_db(timestamp, prediction, churn_status, churn_probability,
              risk_level, missing_count, latency_ms, input_features):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO predictions
            (timestamp, prediction, churn_status, churn_probability,
             risk_level, missing_count, latency_ms, model_version, input_features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, prediction, churn_status, churn_probability,
            risk_level, missing_count, latency_ms, "telecom-churn-logreg-v1",
            json.dumps(input_features)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB logging failed: {e}")


# Request schema
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "tenure_days": 60,
                        "calls_made": 120,
                        "sms_sent": 80,
                        "data_used": 5.5,
                    }
                }
            ]
        }
    }


PREDICT_EXAMPLE = {
    "churn_customer": {
        "summary": "High-risk customer",
        "value": {
            "features": {
                "tenure_days": 30,
                "calls_made": 200,
                "sms_sent": 100,
                "data_used": 9000.0,
                "engagement_score": 2000.0,
            }
        },
    },
    "stable_customer": {
        "summary": "Low-risk customer",
        "value": {
            "features": {
                "tenure_days": 2000,
                "calls_made": 40,
                "sms_sent": 20,
                "data_used": 3000.0,
                "engagement_score": 500.0,
            }
        },
    },
}


# Endpoints
@app.get("/", tags=["Info"])
def home():
    return {
        "message": "Telecom Churn Prediction API is running.",
        "version": "1.2",
        "expected_features": EXPECTED_COLUMNS,
        "total_features": len(EXPECTED_COLUMNS),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/ready", tags=["Health"])
def ready():
    if sk_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ready", "model": "telecom-churn-logreg"}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring", tags=["Monitoring"], response_class=HTMLResponse)
def monitoring_dashboard():
    """Simple HTML monitoring dashboard."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 1000", conn)
        conn.close()

        if df.empty:
            total = churn = no_churn = avg_prob = avg_latency = avg_missing = 0
            churn_rate = 0.0
        else:
            total       = len(df)
            churn       = int((df["prediction"] == 1).sum())
            no_churn    = int((df["prediction"] == 0).sum())
            avg_prob    = round(float(df["churn_probability"].mean()), 4)
            avg_latency = round(float(df["latency_ms"].mean()), 2)
            avg_missing = round(float(df["missing_count"].mean()), 1)
            churn_rate  = round(churn / total * 100, 1) if total > 0 else 0

        cpu    = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Churn API Monitoring</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
                h1 {{ color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 10px; }}
                h2 {{ color: #81d4fa; margin-top: 30px; }}
                .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
                .card {{ background: #16213e; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #0f3460; }}
                .card .value {{ font-size: 2em; font-weight: bold; color: #4fc3f7; }}
                .card .label {{ font-size: 0.9em; color: #aaa; margin-top: 5px; }}
                .alert {{ background: #b71c1c; padding: 10px 20px; border-radius: 8px; margin: 10px 0; }}
                .ok {{ background: #1b5e20; padding: 10px 20px; border-radius: 8px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th {{ background: #0f3460; padding: 10px; text-align: left; }}
                td {{ padding: 8px 10px; border-bottom: 1px solid #0f3460; font-size: 0.85em; }}
            </style>
        </head>
        <body>
            <h1>Telecom Churn API - Monitoring Dashboard</h1>
            <p style="color:#aaa">Auto-refreshes every 30 seconds | Model: telecom-churn-logreg-v1</p>

            <h2>Prediction Stats (Last 1000 requests)</h2>
            <div class="grid">
                <div class="card"><div class="value">{total}</div><div class="label">Total Predictions</div></div>
                <div class="card"><div class="value">{churn_rate}%</div><div class="label">Churn Rate</div></div>
                <div class="card"><div class="value">{avg_prob}</div><div class="label">Avg Churn Probability</div></div>
                <div class="card"><div class="value">{avg_latency}ms</div><div class="label">Avg Latency</div></div>
                <div class="card"><div class="value">{avg_missing}</div><div class="label">Avg Missing Features</div></div>
                <div class="card"><div class="value">{churn} / {no_churn}</div><div class="label">Churn / No Churn</div></div>
            </div>

            <h2>System Metrics</h2>
            <div class="grid">
                <div class="card"><div class="value">{cpu}%</div><div class="label">CPU Usage</div></div>
                <div class="card"><div class="value">{memory}%</div><div class="label">Memory Usage</div></div>
                <div class="card"><div class="value">v1.2</div><div class="label">API Version</div></div>
            </div>

            <h2>Drift Alerts</h2>
            {"<div class='alert'>WARNING: Churn rate " + str(churn_rate) + "% exceeds 30% threshold!</div>" if churn_rate > 30 else "<div class='ok'>Churn rate within normal range (&lt;30%)</div>"}
            {"<div class='alert'>WARNING: High avg latency " + str(avg_latency) + "ms exceeds 500ms threshold!</div>" if avg_latency > 500 else "<div class='ok'>Latency within normal range (&lt;500ms)</div>"}
            {"<div class='alert'>WARNING: High missing features avg " + str(avg_missing) + " exceeds 20 threshold!</div>" if avg_missing > 20 else "<div class='ok'>Feature completeness within normal range</div>"}
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>")


@app.post("/predict", tags=["Prediction"])
def predict(
    request: Annotated[PredictionRequest, Body(openapi_examples=PREDICT_EXAMPLE)],
):
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in request.features]
        unknown_cols = [c for c in request.features if c not in EXPECTED_COLUMNS]

        if missing_cols:
            logger.warning(f"Missing columns filled with 0: {missing_cols}")
        if unknown_cols:
            logger.warning(f"Unknown columns ignored: {unknown_cols}")

        data = {col: 0 for col in EXPECTED_COLUMNS}
        data.update({k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS})
        input_df = pd.DataFrame([data])[EXPECTED_COLUMNS]

        prediction = model.predict(input_df)
        proba = sk_model.predict_proba(input_df)[0][1]

        if proba >= 0.70:
            risk = "High"
        elif proba >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "prediction":            int(prediction[0]),
            "churn_status":          "Churn" if int(prediction[0]) == 1 else "No Churn",
            "churn_probability":     round(float(proba), 4),
            "risk_level":            risk,
            "missing_features_filled": missing_cols,
            "latency_ms":            round(latency_ms, 2),
        }

        # Prometheus metrics
        LATENCY.observe(latency_ms / 1000)
        CHURN_PROBABILITY.observe(float(proba))
        MISSING_FEATURES.observe(len(missing_cols))
        if int(prediction[0]) == 1:
            CHURN_COUNT.inc()
        else:
            NO_CHURN_COUNT.inc()

        # SQLite logging
        log_to_db(
            timestamp=datetime.utcnow().isoformat(),
            prediction=int(prediction[0]),
            churn_status="Churn" if int(prediction[0]) == 1 else "No Churn",
            churn_probability=round(float(proba), 4),
            risk_level=risk,
            missing_count=len(missing_cols),
            latency_ms=round(latency_ms, 2),
            input_features={k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS},
        )

        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
