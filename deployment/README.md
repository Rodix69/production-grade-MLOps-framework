# Phase 7: Monitoring and Observability

## Overview
Full monitoring stack for the Telecom Churn Prediction API covering system metrics, model performance, data drift, prediction logging, alerting, and dashboards.

---

## What Was Built

### 7.1 System Metrics
- CPU and memory tracked via `psutil`
- Request latency tracked via Prometheus `Histogram`
- Metrics exposed at `/metrics` endpoint (Prometheus format)
- Grafana dashboard connected to Prometheus

### 7.2 Model Performance Monitoring
- **Data drift**: Evidently AI compares training vs production feature distributions
- **Prediction drift**: Churn rate in production vs training baseline (15%)
- **Concept drift**: Tracked via prediction probability distribution shifts

### 7.3 Prediction Logging
- Every `/predict` call logged to SQLite database
- Fields: timestamp, prediction, churn_probability, risk_level, missing_count, latency_ms, model_version, input_features

### 7.4 Alerting
- Drift alert: >20% features drifted
- Prediction drift alert: churn rate shifts by >10%
- Latency alert: avg latency >500ms
- Data quality alert: avg missing features >20/27
- All alerts written to `monitoring/alerts.log`

### 7.5 Dashboards
- `/monitoring` endpoint: live HTML dashboard (auto-refreshes every 30s)
- Grafana: `http://localhost:3000` (admin/admin)
- Prometheus: `http://localhost:9090`

---

## File Structure

```
deployment/
    app.py                          Updated API with logging + metrics
    Dockerfile                      Updated with monitoring folder
    requirements.txt                Added prometheus-client, evidently, psutil
    monitoring/
        drift_detection.py          Evidently drift report script
        retrain_trigger.py          Retraining trigger checks
        prometheus.yml              Prometheus scrape config
        docker-compose.yml          Prometheus + Grafana stack
        predictions.db              SQLite prediction log (auto-created)
        drift_log.csv               Drift history log (auto-created)
        alerts.log                  Alert history log (auto-created)
        reports/                    Evidently HTML reports (auto-created)
```

---

## Setup and Usage

### 1. Run the API locally
```bash
cd deployment
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Make some predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure_days": 30, "calls_made": 200, "data_used": 9000}}'
```

### 3. View live monitoring dashboard
```
http://localhost:8000/monitoring
```

### 4. View Prometheus metrics
```
http://localhost:8000/metrics
```

### 5. Start Grafana + Prometheus
```bash
cd deployment/monitoring
docker-compose up -d
```
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### 6. Run drift detection
```bash
cd deployment
python monitoring/drift_detection.py
```

### 7. Run retraining trigger check
```bash
cd deployment
python monitoring/retrain_trigger.py
```

---

## Alerting Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Data drift share | >20% | Retrain model |
| Prediction drift | >10% shift | Retrain model |
| Avg latency | >500ms | Investigate API |
| Missing features | >20/27 avg | Check data pipeline |

---

## Deliverables

- [x] System metrics (CPU, memory, latency, request count, error rate)
- [x] Data drift detection (Evidently AI)
- [x] Prediction drift monitoring
- [x] Prediction logging to SQLite
- [x] Live monitoring dashboard (/monitoring endpoint)
- [x] Prometheus metrics endpoint (/metrics)
- [x] Grafana + Prometheus docker-compose stack
- [x] Alerting rules (4 thresholds)
- [x] Retraining trigger script
- [x] Drift HTML reports
