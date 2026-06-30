# Production-Grade MLOps Framework

End-to-end MLOps implementation for a telecom customer churn prediction system, built across 10 phases following production ML engineering practices. Each phase is developed and submitted as its own branch.

## Project Overview

The model predicts customer churn from telecom usage data. The project demonstrates the full MLOps lifecycle: data engineering, feature engineering, experiment tracking, pipeline automation, deployment, monitoring, continuous retraining, and governance.

### Production Model (Phase 4 / Phase 6)

The model actually serving live traffic is a **Logistic Regression baseline**, registered in MLflow as `telecom-churn-logreg`. An **XGBoost (Optuna-tuned)** candidate was trained and evaluated against it but did not clear the promotion gate, so it remains in **Staging**, not Production.

| Model | AUC | F1 | Recall | Precision | Stage |
|-------|-----|----|--------|-----------|-------|
| Logistic Regression (baseline) | 0.8302 | 0.4727 | 0.7652 | 0.3420 | **Production** |
| XGBoost (Optuna-tuned) | 0.8296 | 0.4809 | 0.7356 | 0.3572 | Staging |

XGBoost's AUC came in 0.0005 below the baseline (promotion rule: candidate AUC must exceed baseline AUC), so Logistic Regression remained champion. This comparison and the promotion decision are implemented in the Phase 4 notebook and logged to MLflow.

### Reference Pipeline Model (Phase 5 / 8 / 9)

The branches for Phase 5, 8, and 9 train and register a separate **Random Forest classifier** (test AUC: 0.8291). This model is **not** the one deployed in production -- it exists to demonstrate pipeline automation, continuous retraining, and governance tooling (lineage, explainability, approval workflow, audit logs) end-to-end on a working architecture. The Production model story above (Logistic Regression vs XGBoost) is the actual deployed system.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Version control | Git, DVC |
| Experiment tracking | MLflow |
| Pipeline orchestration | Prefect |
| Containerization | Docker |
| Deployment | FastAPI, Railway |
| Monitoring | Evidently AI, Prometheus, custom SQLite audit logs |
| Model registry | MLflow Model Registry |
| Explainability | SHAP |

## Phase Branches

Each phase is implemented and submitted independently on its own branch. `main` is intentionally kept minimal -- it does not contain merged code from the phase branches.

| Phase | Description | Branch | Status |
|-------|-------------|--------|--------|
| 1 | Problem Definition and Feasibility | -- | Complete |
| 2 | Data Engineering and Versioning | -- | Complete |
| 3 | Feature Engineering and Store | -- | Partial |
| 4 | Model Development and Experiment Tracking | -- | Complete |
| 5 | Pipeline Automation (CI/CD for ML) | [`phase5-pipeline-automation`](https://github.com/Rodix69/production-grade-MLOps-framework/tree/phase5-pipeline-automation) | Complete |
| 6 | Model Deployment | [`deployment` (teammate)](https://github.com/Rodix69/production-grade-MLOps-framework/tree/main/deployment) | Complete -- live on Railway |
| 7 | Monitoring and Observability | teammate branch | In progress |
| 8 | Continuous Training and Retraining | [`phase8-continuous-training`](https://github.com/Rodix69/production-grade-MLOps-framework/tree/phase8-continuous-training) | Complete |
| 9 | Governance and Compliance | [`phase9-governance-compliance`](https://github.com/Rodix69/production-grade-MLOps-framework/tree/phase9-governance-compliance) | Complete |
| 10 | Maintenance and Iteration | not started | Not started |

## How to Review a Phase

Each phase branch contains:
- The code for that phase
- A `README.md` documenting what was built, how it works, and how to run it
- An open pull request into `main` showing the full diff for that phase's work

Open the relevant branch link above, or check the [open pull requests](https://github.com/Rodix69/production-grade-MLOps-framework/pulls) for a side-by-side diff view.

## Key Results

| Metric | Production (LogReg) | Staged (XGBoost) | Reference Pipeline (Random Forest, Phase 5/8/9) |
|--------|---------------------|-------------------|---------------------------------------------------|
| Test AUC | 0.8302 | 0.8296 | 0.8291 |
| Inference time | -- | -- | ~85ms |
| Model size | -- | -- | 1.80 MB |
| Validation gate (Phase 5/8/9) | -- | -- | AUC >= 0.80, latency <= 500ms, memory <= 500MB |

## Repository Structure (per phase branch)

```
phase5-pipeline-automation/
  phase5_automation/
    main_flow.py              Prefect orchestration entry point
    pipeline_steps.py         ingest, features, train, validate, register
    monitoring_flow.py        Evidently drift detection
    prefect.yaml               daily 6am cron schedule
    docker-compose.yml         MLflow + Prefect + Prometheus + Grafana

phase8-continuous-training/
  phase5_automation/
    retrain_manager.py         volume trigger, version tracking, rollback
    simulate_new_data.py       simulate new data arriving

phase9-governance-compliance/
  phase5_automation/
    lineage_tracker.py         end-to-end lineage recording
    explainability.py          SHAP feature importance reports
    approval_workflow.py       Dev -> Staging -> Production gates
    data_privacy.py            PII anonymization, RBAC, compliance report
  deployment/
    audit/                     tamper-evident prediction audit logging
```

## Contributors

- Rohit -- Phases 5, 8, 9, 10
- Teammate (forked) -- Phases 6, 7, 9.4
