"""
Phase 9.4 - Audit Log Patch
Add these to your existing app.py.
Phase 7 predictions table is untouched.
"""

# ── ADD to imports section ────────────────────────────────────────────────────
import uuid
import hashlib
# (fastapi Request is also needed — add to existing fastapi import line)

# ── ADD after existing init_db() ─────────────────────────────────────────────
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


def hash_input(features: dict) -> str:
    """SHA-256 hash of input — proves inputs were not tampered with."""
    return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()


def get_top_features(input_df, sk_model, n=3) -> str:
    """Top N features by impact (coefficient x value)."""
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
        """, (
            request_id, timestamp, "prediction",
            "telecom-churn-logreg-v1", prediction,
            churn_probability, risk_level, compliance_flag,
            user_ip, input_hash, top_features, latency_ms
        ))
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


# Log model startup event
log_compliance_event("model_loaded", "Model telecom-churn-logreg-v1 loaded at startup")


# ── ADD these new endpoints to app.py ─────────────────────────────────────────

@app.get("/audit/logs", tags=["Audit"])
def get_audit_logs(limit: int = 100):
    """Return recent audit log entries — Step 9.4."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return {"total": len(df), "logs": df.to_dict(orient="records")}


@app.get("/audit/compliance-events", tags=["Audit"])
def get_compliance_events(limit: int = 100):
    """Return compliance events (model changes, access, overrides)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM compliance_events ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return {"total": len(df), "events": df.to_dict(orient="records")}


@app.get("/audit/high-risk", tags=["Audit"])
def get_high_risk_predictions(limit: int = 100):
    """Return high-risk predictions (churn_probability >= 0.70) for compliance review."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM audit_log WHERE compliance_flag = 1 ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return {"total_high_risk": len(df), "predictions": df.to_dict(orient="records")}


@app.post("/audit/review/{request_id}", tags=["Audit"])
def review_prediction(request_id: str, notes: str = ""):
    """Mark a prediction as reviewed by a human auditor."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE audit_log SET reviewed = 1, review_notes = ? WHERE request_id = ?",
        (notes, request_id)
    )
    affected = conn.total_changes
    conn.commit()
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
    log_compliance_event("human_review", f"Prediction {request_id} reviewed: {notes}", actor="auditor")
    return {"status": "reviewed", "request_id": request_id, "notes": notes}


# ── ADD inside existing /predict endpoint, after result dict is built ─────────
# (these lines slot in just before "logger.info(f'Prediction result: {result}')")

"""
request_id   = str(uuid.uuid4())
input_hash   = hash_input({k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS})
top_features = get_top_features(input_df, sk_model)
user_ip      = "api-client"

log_to_audit(
    request_id=request_id,
    timestamp=datetime.utcnow().isoformat(),
    prediction=int(prediction[0]),
    churn_probability=round(float(proba), 4),
    risk_level=risk,
    user_ip=user_ip,
    input_features={k: v for k, v in request.features.items() if k in EXPECTED_COLUMNS},
    input_hash=input_hash,
    top_features=top_features,
    latency_ms=round(latency_ms, 2),
)

result["request_id"]   = request_id
result["input_hash"]   = input_hash
result["audit_logged"] = True
"""
