# Phase 9.4: Audit Logs

## Overview
Implements tamper-evident audit logging for all prediction requests.
Extends the existing Phase 7 `predictions.db` with two new tables.
All existing Phase 7 functionality is completely unchanged.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Full combined app (Phase 7 + Phase 9.4) — replace existing |
| `app_audit_patch.py` | Just the audit additions — reference for what changed |
| `audit/audit_report.py` | Generates HTML compliance audit report |
| `README.md` | This file |

---

## New Endpoints (Phase 9.4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/audit/logs` | All audit log entries |
| GET | `/audit/high-risk` | High-risk predictions (probability >= 0.70) |
| GET | `/audit/compliance-events` | Model load, review, override events |
| POST | `/audit/review/{request_id}` | Mark prediction as human reviewed |

---

## New DB Tables (same predictions.db)

### audit_log
| Field | Description |
|-------|-------------|
| request_id | Unique UUID per prediction |
| timestamp | UTC timestamp |
| event_type | Always "prediction" |
| model_version | telecom-churn-logreg-v1 |
| prediction | 0 or 1 |
| churn_probability | 0.0 to 1.0 |
| risk_level | High / Medium / Low |
| compliance_flag | 1 if probability >= 0.70 |
| user_ip | Client IP address |
| input_hash | SHA-256 of input features |
| top_features | Top 3 influential features (JSON) |
| reviewed | 0 = pending, 1 = reviewed |
| review_notes | Human reviewer notes |
| latency_ms | Response time |

### compliance_events
| Field | Description |
|-------|-------------|
| event_type | model_loaded, human_review, etc. |
| description | Human readable event description |
| actor | system or auditor |
| model_version | Model that was active |

---

## Setup and Usage

### 1. Replace app.py
```
Copy app.py to deployment/app.py
```

### 2. Start the API
```cmd
cd C:\Users\a\Desktop\intership\MLOPS\deployment
C:\Users\a\Desktop\intership\MLOPS\myenv\Scripts\uvicorn.exe app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Make predictions (audit logged automatically)
```cmd
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"tenure_days\": 30, \"calls_made\": 200}}"
```

### 4. View audit logs
```
http://localhost:8000/audit/logs
http://localhost:8000/audit/high-risk
http://localhost:8000/audit/compliance-events
```

### 5. Mark a prediction as reviewed
```cmd
curl -X POST "http://localhost:8000/audit/review/YOUR-REQUEST-ID?notes=Reviewed+and+approved"
```

### 6. Generate audit report
```cmd
cd C:\Users\a\Desktop\intership\MLOPS\deployment
C:\Users\a\Desktop\intership\MLOPS\myenv\Scripts\python.exe audit/audit_report.py
```

---

## What Each Prediction Response Now Returns
```json
{
  "prediction": 1,
  "churn_status": "Churn",
  "churn_probability": 0.9823,
  "risk_level": "High",
  "missing_features_filled": [...],
  "latency_ms": 12.4,
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "input_hash": "sha256:a3f5...",
  "audit_logged": true
}
```
