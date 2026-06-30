# Phase 9: Governance and Compliance

This phase implements end-to-end governance for the telecom churn prediction system: model lineage tracking, explainability, a staged approval workflow, tamper-evident audit logging, and data privacy controls.

| Sub-phase | Requirement | Owner | Status |
|-----------|-------------|-------|--------|
| 9.1 | Model lineage (data → features → model → deployment) | Pipeline team | ✅ Complete |
| 9.2 | Model explainability (SHAP) | Pipeline team | ✅ Complete |
| 9.3 | Approval workflow (Dev → Staging → Production) | Pipeline team | ✅ Complete |
| 9.4 | Audit logs for predictions | Serving team | ✅ Complete |
| 9.5 | Data privacy (anonymization, RBAC, encryption) | Pipeline team | ✅ Complete |

---

## 9.1 — Model Lineage

`phase5_automation/lineage_tracker.py`

Every successful training run records a complete lineage chain linking:

- **Data sources** — train/val/test parquet paths, row counts, file modification times
- **Feature definitions** — raw features, engineered features, and dropped PII/ID columns
- **Model** — algorithm, hyperparameters, run ID, val/test AUC, MLflow URL
- **Deployment** — registry name, version, stage, promotion timestamp

Lineage records are also written back as MLflow run tags (`lineage_id`, `model_version`, `trigger`, `train_rows`, `feature_count`) so every MLflow run is traceable without leaving the MLflow UI.

```bash
cd phase5_automation
python lineage_tracker.py latest      # most recent lineage report
python lineage_tracker.py history     # all recorded versions
python lineage_tracker.py version 7   # lineage for a specific model version
```

Records persist in `phase5_automation/lineage_registry.json`.

---

## 9.2 — Model Explainability

`phase5_automation/explainability.py`

Generates SHAP-based explainability artifacts for the Production model:

- **Feature importance bar chart** — top 15 features by mean |SHAP value|
- **SHAP summary plot** — distribution of feature impact across samples
- **Dependence plot** — relationship between the top feature and model output
- **Single-prediction explanation** — per-customer feature contributions

All plots and a JSON summary are saved to `outputs/explainability/` and logged as MLflow artifacts on the run.

```bash
python explainability.py
```

Current top churn drivers (by SHAP importance): `sms_sent`, `avg_sms_per_day`, `calls_made`, `calls_intensity`, `calls_vs_partner`.

---

## 9.3 — Approval Workflow

`phase5_automation/approval_workflow.py`

Implements staged promotion gates in the MLflow Model Registry:

| Stage | Min AUC | Max Inference | Max Memory |
|-------|---------|---------------|------------|
| Staging | 0.78 | 500ms | 500MB |
| Production | 0.80 | 500ms | 500MB |

A model can only move to the next stage if it passes the gate for that stage. Every promotion attempt (approved or rejected) is recorded with a timestamp, approver, and the metrics evaluated.

```bash
python approval_workflow.py status              # current registry state
python approval_workflow.py gates                # view gate thresholds
python approval_workflow.py history               # full approval/rejection log
python approval_workflow.py promote --version 7 --stage Production
```

Decision history persists in `phase5_automation/approval_workflow.json`.

---

## 9.4 — Audit Logs

`deployment/audit/` (serving layer)

Every prediction made through the live API is logged to a tamper-evident audit trail, separate from the existing Phase 7 monitoring table.

### What gets logged

Each `/predict` call writes a row to a SQLite `audit_log` table containing:

- `request_id` — unique UUID per request
- `input_hash` — SHA-256 hash of the input features (proves the input wasn't altered after the fact)
- `model_version`, `prediction`, `churn_probability`, `risk_level`
- `compliance_flag` — automatically set to `1` for high-risk predictions (probability ≥ 0.70)
- `top_features` — top 3 features driving that specific prediction
- `reviewed` / `review_notes` — human sign-off fields for regulated-industry review
- `user_ip`, `latency_ms`

A second `compliance_events` table logs system-level events (model load, human reviews, overrides).

### New API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/audit/logs` | GET | Recent audit log entries |
| `/audit/compliance-events` | GET | Model loads, reviews, and other compliance events |
| `/audit/high-risk` | GET | Predictions flagged for compliance review (probability ≥ 0.70) |
| `/audit/review/{request_id}` | POST | Mark a prediction as human-reviewed with notes |

The existing `/predict` response now also returns `request_id`, `input_hash`, and `audit_logged: true`, so any caller can reference their specific prediction in the audit trail.

### Compliance report

`audit/audit_report.py` generates a standalone HTML report summarizing total predictions, high-risk count, review status, and average latency, alongside a full audit log table and compliance event timeline.

```bash
cd deployment
python audit/audit_report.py
```

Output is saved to `monitoring/reports/audit_report_<timestamp>.html`.

### Design notes

- The existing Phase 7 `predictions` table is untouched — audit logging is fully additive.
- `input_hash` makes the log tamper-evident: any change to the original input would change the hash, so logged hashes can be cross-checked against re-submitted inputs during an investigation.
- `compliance_flag` and the `/audit/high-risk` endpoint exist specifically to support regulated-industry review workflows (e.g., finance, telecom churn decisions affecting customer treatment).

---

## 9.5 — Data Privacy

`phase5_automation/data_privacy.py`

### PII classification

| Sensitivity | Columns | Anonymization strategy |
|-------------|---------|------------------------|
| High | `customer_id`, `pincode` | SHA-256 hashing |
| Medium | `gender`, `age`, `date_of_registration`, `estimated_salary` | Generalization (age bands, salary quartiles, year only) |
| Low | `city`, `state`, `telecom_partner` | Suppression (`REDACTED`) |
| Non-sensitive | `calls_made`, `sms_sent`, `data_used`, `tenure_days`, `num_dependents`, `churn` | No transformation |

All high/medium/low sensitivity columns are dropped before model training — only behavioral, non-sensitive features are used (see Phase 5 `create_features`).

### Role-based access control (RBAC)

| Role | Access |
|------|--------|
| `data_scientist` | Anonymized training data, model metrics, SHAP reports |
| `ml_engineer` | Full pipeline, model registry, deployment |
| `data_engineer` | Raw data for pipeline maintenance |
| `auditor` | Read-only access to audit logs and lineage records |
| `business_analyst` | Aggregated metrics and reports only |

```bash
python data_privacy.py report     # generate compliance report (JSON)
python data_privacy.py demo       # anonymization demo on sample rows
python data_privacy.py rbac       # print RBAC policy
```

Compliance report is saved to `outputs/privacy/privacy_report.json`.

---

## How It All Connects

```
Training run (Phase 5/8)
        │
        ▼
  lineage_tracker.py  ──► lineage_registry.json + MLflow tags        (9.1)
        │
        ▼
  explainability.py   ──► SHAP plots + JSON, logged to MLflow         (9.2)
        │
        ▼
  approval_workflow.py ──► Staging/Production gate check + history    (9.3)
        │
        ▼
  Model promoted to Production in MLflow Registry
        │
        ▼
  FastAPI /predict (deployment/app.py)
        │
        ▼
  audit_log + compliance_events tables ──► /audit/* endpoints          (9.4)
        │
        ▼
  audit_report.py ──► HTML compliance report

  data_privacy.py runs independently of the above —
  applied during feature engineering and available on demand          (9.5)
```

## Running Everything

```bash
# Pipeline-side governance (lineage, explainability, approval, privacy)
cd phase5_automation
python main_flow.py                 # runs pipeline, auto-records lineage + privacy report
python lineage_tracker.py latest
python explainability.py
python approval_workflow.py status
python data_privacy.py report

# Serving-side audit logs
cd deployment
python audit/audit_report.py
```
