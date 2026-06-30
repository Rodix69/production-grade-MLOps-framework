from fastapi import FastAPI, HTTPException, Body, Request
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
import uuid
import hashlib
import psutil
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.3",
    description="Predicts customer churn. Includes Phase 7 monitoring and Phase 9.4 audit logging.",
)

# Model
logger.info("Loading MLflow model...")
model    = mlflow.pyfunc.load_model("model")
sk_model = model._model_impl.sklearn_model
EXPECTED_COLUMNS = list(sk_model.feature_names_in_)
logger.info(f"Model loaded. Expecting {len(EXPECTED_COLUMNS)} features.")

# Prometheus metrics (Phase 7 - unchanged)
REQUEST_COUNT     = Counter("prediction_requests_total",    "Total prediction requests")
ERROR_COUNT       = Counter("prediction_errors_total",      "Total prediction errors")
CHURN_COUNT       = Counter("churn_predictions_total",      "Total churn predictions")
NO_CHURN_COUNT    = Counter("no_churn_predictions_total",   "Total no-churn predictions")
LATENCY           = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
MISSING_FEATURES  = Histogram("missing_features_per_request","Missing features per request", buckets=[0,1,2,5,10,15,20,27])
CHURN_PROBABILITY = Histogram("churn_probability",          "Churn probability distribution", buckets=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

# DB path
DB_PATH = os.environ.get("DB_PATH", "monitoring/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ── Phase 7: Predictions table (unchanged) ────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            prediction        INTEGER,
            churn_status      TEXT,
            churn_probability REAL,
            risk_level        TEXT,
            missing_count     INTEGER,
            latency_ms        REAL,
            model_version     TEXT,
            input_features    TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ── Phase 9.4: Audit tables ───────────────────────────────────────────────────
def init_audit_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id        TEXT NOT NULL UNIQUE,
            timestamp         TEXT NOT NULL,
            event_type        TEXT NOT NULL,
            model_version     TEXT,
            prediction        INTEGER,
            churn_probability REAL,
            risk_level        TEXT,
            compliance_flag   INTEGER DEFAULT 0,
            user_ip           TEXT,
            input_hash        TEXT,
            top_features      TEXT,
            reviewed          INTEGER DEFAULT 0,
            review_notes      TEXT,
            latency_ms        REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compliance_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            description   TEXT,
            actor         TEXT,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Audit log tables initialized.")

init_audit_db()


# ── Phase 7: DB logging ───────────────────────────────────────────────────────
def log_to_db(timestamp, prediction, churn_status, churn_probability,
              risk_level, missing_count, latency_ms, input_features):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO predictions
            (timestamp, prediction, churn_status, churn_probability,
             risk_level, missing_count, latency_ms, model_version, input_features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, prediction, churn_status, churn_probability,
              risk_level, missing_count, latency_ms,
              "telecom-churn-logreg-v1", json.dumps(input_features)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB logging failed: {e}")


# ── Phase 9.4: Audit helpers ──────────────────────────────────────────────────
def hash_input(features: dict) -> str:
    return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()


def get_top_features(input_df, n=3) -> str:
    try:
        impacts = {
            name: abs(float(coef) * float(val))
            for name, coef, val in zip(
                sk_model.feature_names_in_,
                sk_model.coef_[0],
                input_df.iloc[0].values
            )
        }
        top = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:n]
        return json.dumps({k: round(v, 4) for k, v in top})
    except Exception:
        return "{}"


def log_to_audit(request_id, timestamp, prediction, churn_probability,
                 risk_level, user_ip, input_features, input_hash,
                 top_features, latency_ms):
    try:
        compliance_flag = 1 if churn_probability >= 0.70 else 0
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO audit_log
            (request_id, timestamp, event_type, model_version, prediction,
             churn_probability, risk_level, compliance_flag, user_ip,
             input_hash, top_features, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (request_id, timestamp, "prediction", "telecom-churn-logreg-v1",
              prediction, churn_probability, risk_level, compliance_flag,
              user_ip, input_hash, top_features, latency_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log failed: {e}")


def log_compliance_event(event_type: str, description: str, actor: str = "system"):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO compliance_events
            (timestamp, event_type, description, actor, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), event_type, description, actor,
              "telecom-churn-logreg-v1"))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Compliance event log failed: {e}")

log_compliance_event("model_loaded", "Model telecom-churn-logreg-v1 loaded at startup")


# ── Request schema ────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_config = {
        "json_schema_extra": {"examples": [{"features": {"tenure_days": 60, "calls_made": 120}}]}
    }

PREDICT_EXAMPLE = {
    "churn_customer": {
        "summary": "High-risk customer",
        "value": {"features": {"tenure_days": 30, "calls_made": 200, "data_used": 9000.0}},
    },
    "stable_customer": {
        "summary": "Low-risk customer",
        "value": {"features": {"tenure_days": 2000, "calls_made": 40, "data_used": 3000.0}},
    },
}


# ── Phase 7 Endpoints (unchanged) ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def home():
    return {"message": "Telecom Churn API v1.3", "version": "1.3",
            "expected_features": EXPECTED_COLUMNS, "total_features": len(EXPECTED_COLUMNS)}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.get("/ready", tags=["Health"])
def ready():
    if sk_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ready", "model": "telecom-churn-logreg"}

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/monitoring", tags=["Monitoring"], response_class=HTMLResponse)
def monitoring_dashboard():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 1000", conn)
        conn.close()
        if df.empty:
            total = churn = no_churn = avg_prob = avg_latency = avg_missing = churn_rate = 0
        else:
            total = len(df); churn = int((df["prediction"]==1).sum())
            no_churn = int((df["prediction"]==0).sum())
            avg_prob = round(float(df["churn_probability"].mean()), 4)
            avg_latency = round(float(df["latency_ms"].mean()), 2)
            avg_missing = round(float(df["missing_count"].mean()), 1)
            churn_rate = round(churn/total*100, 1) if total > 0 else 0
        cpu = psutil.cpu_percent(); memory = psutil.virtual_memory().percent
        html = f"""<!DOCTYPE html><html><head><title>Churn API Monitoring</title>
        <meta http-equiv="refresh" content="30">
        <style>body{{font-family:Arial;background:#1a1a2e;color:#eee;padding:20px}}
        h1{{color:#4fc3f7;border-bottom:2px solid #4fc3f7;padding-bottom:10px}}
        h2{{color:#81d4fa;margin-top:30px}}
        .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}}
        .card{{background:#16213e;border-radius:8px;padding:20px;text-align:center;border:1px solid #0f3460}}
        .card .value{{font-size:2em;font-weight:bold;color:#4fc3f7}}
        .card .label{{font-size:.9em;color:#aaa;margin-top:5px}}
        .alert{{background:#b71c1c;padding:10px 20px;border-radius:8px;margin:10px 0}}
        .ok{{background:#1b5e20;padding:10px 20px;border-radius:8px;margin:10px 0}}</style></head>
        <body><h1>Telecom Churn API - Monitoring Dashboard</h1>
        <p style="color:#aaa">Auto-refreshes every 30s | Model: telecom-churn-logreg-v1</p>
        <h2>Prediction Stats</h2>
        <div class="grid">
        <div class="card"><div class="value">{total}</div><div class="label">Total Predictions</div></div>
        <div class="card"><div class="value">{churn_rate}%</div><div class="label">Churn Rate</div></div>
        <div class="card"><div class="value">{avg_prob}</div><div class="label">Avg Churn Probability</div></div>
        <div class="card"><div class="value">{avg_latency}ms</div><div class="label">Avg Latency</div></div>
        <div class="card"><div class="value">{avg_missing}</div><div class="label">Avg Missing Features</div></div>
        <div class="card"><div class="value">{churn}/{no_churn}</div><div class="label">Churn/No Churn</div></div>
        </div>
        <h2>System</h2>
        <div class="grid">
        <div class="card"><div class="value">{cpu}%</div><div class="label">CPU</div></div>
        <div class="card"><div class="value">{memory}%</div><div class="label">Memory</div></div>
        <div class="card"><div class="value">v1.3</div><div class="label">API Version</div></div>
        </div>
        <h2>Drift Alerts</h2>
        {"<div class='alert'>WARNING: Churn rate "+str(churn_rate)+"% exceeds 30%</div>" if churn_rate>30 else "<div class='ok'>Churn rate within normal range</div>"}
        {"<div class='alert'>WARNING: High latency "+str(avg_latency)+"ms</div>" if avg_latency>500 else "<div class='ok'>Latency within normal range</div>"}
        {"<div class='alert'>WARNING: Avg missing features "+str(avg_missing)+"</div>" if avg_missing>20 else "<div class='ok'>Feature completeness OK</div>"}
        </body></html>"""
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>")


# ── Phase 9.4 Audit Endpoints ─────────────────────────────────────────────────
@app.get("/audit/logs", tags=["Audit"])
def get_audit_logs(limit: int = 100):
    """Return recent audit log entries — Phase 9.4."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return {"total": len(df), "logs": df.to_dict(orient="records")}

@app.get("/audit/compliance-events", tags=["Audit"])
def get_compliance_events(limit: int = 100):
    """Return compliance events (model loads, overrides, access)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM compliance_events ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return {"total": len(df), "events": df.to_dict(orient="records")}

@app.get("/audit/high-risk", tags=["Audit"])
def get_high_risk_predictions(limit: int = 100):
    """Return high-risk predictions (probability >= 0.70) for compliance review."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM audit_log WHERE compliance_flag=1 ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return {"total_high_risk": len(df), "predictions": df.to_dict(orient="records")}

@app.post("/audit/review/{request_id}", tags=["Audit"])
def review_prediction(request_id: str, notes: str = ""):
    """Mark a prediction as reviewed by a human auditor."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE audit_log SET reviewed=1, review_notes=? WHERE request_id=?",
        (notes, request_id)
    )
    affected = conn.total_changes
    conn.commit()
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
    log_compliance_event("human_review", f"Prediction {request_id} reviewed: {notes}", actor="auditor")
    return {"status": "reviewed", "request_id": request_id, "notes": notes}


# ── /predict endpoint (Phase 7 + Phase 9.4 combined) ─────────────────────────
@app.post("/predict", tags=["Prediction"])
def predict(
    request: Annotated[PredictionRequest, Body(openapi_examples=PREDICT_EXAMPLE)],
    http_request: Request = None,
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

        prediction   = model.predict(input_df)
        proba        = sk_model.predict_proba(input_df)[0][1]
        risk         = "High" if proba >= 0.70 else ("Medium" if proba >= 0.40 else "Low")
        latency_ms   = (time.time() - start_time) * 1000

        # Phase 9.4: Audit fields
        request_id   = str(uuid.uuid4())
        clean_input  = {k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS}
        input_hash   = hash_input(clean_input)
        top_features = get_top_features(input_df)
        user_ip      = http_request.client.host if http_request else "unknown"

        result = {
            "prediction":              int(prediction[0]),
            "churn_status":            "Churn" if int(prediction[0]) == 1 else "No Churn",
            "churn_probability":       round(float(proba), 4),
            "risk_level":              risk,
            "missing_features_filled": missing_cols,
            "latency_ms":              round(latency_ms, 2),
            # Phase 9.4 fields
            "request_id":              request_id,
            "input_hash":              input_hash,
            "audit_logged":            True,
        }

        # Phase 7: Prometheus
        LATENCY.observe(latency_ms / 1000)
        CHURN_PROBABILITY.observe(float(proba))
        MISSING_FEATURES.observe(len(missing_cols))
        if int(prediction[0]) == 1:
            CHURN_COUNT.inc()
        else:
            NO_CHURN_COUNT.inc()

        # Phase 7: SQLite predictions table
        log_to_db(
            timestamp=datetime.utcnow().isoformat(),
            prediction=int(prediction[0]),
            churn_status="Churn" if int(prediction[0]) == 1 else "No Churn",
            churn_probability=round(float(proba), 4),
            risk_level=risk,
            missing_count=len(missing_cols),
            latency_ms=round(latency_ms, 2),
            input_features=clean_input,
        )

        # Phase 9.4: Audit log table
        log_to_audit(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            prediction=int(prediction[0]),
            churn_probability=round(float(proba), 4),
            risk_level=risk,
            user_ip=user_ip,
            input_features=clean_input,
            input_hash=input_hash,
            top_features=top_features,
            latency_ms=round(latency_ms, 2),
        )

        logger.info(f"Prediction: {result}")
        return result

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
