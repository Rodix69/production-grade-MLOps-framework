# Phase 5 — Pipeline Automation

Automated training pipeline for the Telecom Churn MLOps project, orchestrated with [Prefect](https://www.prefect.io/). This phase takes processed data through feature engineering, model training, validation, and registration — with a conditional retraining trigger driven by drift detection.

## What this phase does

```
ingest_data → create_features → train_model → validate_model → register_model
                                                                       │
                                                          (drift check via Phase 7)
                                                                       │
                                                          if drift → repeat pipeline
```

1. **Ingest** — loads the train/val/test parquet splits produced in Phase 2
2. **Feature engineering** — rebuilds the `churn` label from a synthetic business-logic formula (see note below), then engineers 15 additional features on top of the 7 raw columns
3. **Train** — fits a `RandomForestClassifier` and logs the run to MLflow
4. **Validate** — gates the model on AUC, inference latency, and memory footprint before it's allowed to register
5. **Register** — promotes a passing model to `Production` stage in the MLflow Model Registry
6. **Drift check** — calls into Phase 7's monitoring flow; if drift is detected, the pipeline reruns automatically

## Important: the `churn` label is synthetic

The `churn` column in the source parquet files is **random noise** — it carries no real signal. `create_features()` discards it and rebuilds a new label from a logistic formula based on `calls_made`, `sms_sent`, `data_used`, `estimated_salary`, and `num_dependents`:

```python
logit = 2.2 - 4.0*norm(calls) - 3.0*norm(sms) - 2.0*norm(data) - 1.5*norm(salary) + 0.5*norm(dependents)
prob  = sigmoid(logit)
churn = bernoulli(prob)
```

This is intentional — it simulates a plausible churn relationship for demo purposes. **If this label-rebuild step is ever removed or bypassed, model AUC collapses to ~0.50** (random guessing), since the original `churn` column has no relationship to the features.

## Feature engineering

22 features are used for training: 7 raw + 15 engineered.

| Feature | Description |
|---|---|
| `total_activity` | sum of calls, sms, data usage |
| `avg_calls_per_day`, `avg_data_per_day`, `avg_sms_per_day` | usage normalized by tenure |
| `engagement_score` | weighted blend of calls/sms/data |
| `low_activity_flag` | below 25th percentile of `total_activity` (threshold fit on train only) |
| `high_value_user` | above 75th percentile of `estimated_salary` (threshold fit on train only) |
| `partner_med_calls/data/sms` | median usage per `telecom_partner`, fit on train only |
| `calls_vs_partner`, `data_vs_partner` | usage relative to partner median |
| `is_new_customer` | tenure below 25th percentile |
| `calls_intensity`, `data_intensity` | usage relative to median, normalized |

All thresholds and medians are computed on the **training set only** and applied to val/test — this avoids data leakage.

Dropped before training: `date_of_registration`, `customer_id`, `pincode`, `city`, `state`, `telecom_partner`, `gender` (identifiers and raw categoricals not used directly by the model).

## Validation gates

A trained model is only registered if **all three** conditions pass:

| Gate | Threshold |
|---|---|
| Test AUC | ≥ 0.80 |
| Inference time | ≤ 500 ms |
| Memory footprint | ≤ 500 MB |

If any gate fails, the run is logged to MLflow but the model is **not** registered or promoted — the pipeline prints which gate failed and stops there.

## Drift-triggered retraining

`main_flow.py` imports `monitoring_flow` from **Phase 7** to check for data drift between the training and test distributions (using Evidently). If drift is detected, the full ingest → feature → train → validate → register sequence runs a second time automatically.

> This phase **depends on** Phase 7's monitoring flow but does not own its logic. See the Phase 7 README for drift detection details (Evidently report generation, the Prometheus `churn_data_drift_detected` gauge, etc.).

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust paths/URIs as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/processed` | Folder containing `churn_train_v1.parquet`, `churn_val_v1.parquet`, `churn_test_v1.parquet` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server — must be running before the pipeline starts |

## Running

Start MLflow first:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then run the pipeline directly:

```bash
python main_flow.py
```

Or deploy it on a schedule via Prefect (`prefect.yaml` runs it daily at 06:00 UTC):

```bash
prefect deploy
```

## Files in this phase

| File | Purpose |
|---|---|
| `main_flow.py` | Top-level Prefect flow — orchestrates the full pipeline and the drift-retrain loop |
| `pipeline_steps.py` | Individual Prefect tasks: ingest, feature engineering, train, validate, register |
| `prefect.yaml` | Deployment config — daily cron schedule, work pool |
| `requirements.txt` | Pinned dependencies for this phase |
| `.env.example` | Template for required environment variables |

## Known limitations

- Model promotion to `Production` is fully automatic on passing validation — there is no human approval gate in this phase (see Phase 8 for a gated retraining workflow)
- `create_features()` recomputes engineered features inline rather than reusing a saved feature-transformation artifact, so training-time and serving-time feature logic must be kept in sync manually
