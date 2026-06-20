# Phase 8 — Continuous Training

Extends Phase 5's training pipeline with automated retraining triggers, model version tracking, and rollback capability. The core train/validate/register logic is unchanged from Phase 5 — this phase adds the decision layer that decides *when* to retrain and *how to recover* if a new model underperforms.

## What this phase adds on top of Phase 5

| Capability | File |
|---|---|
| 8.1 — Retraining triggers (drift, data volume) | `main_flow.py`, `retrain_manager.py` |
| 8.2 — Scheduled retraining | `main_flow.py` (always runs once per invocation) |
| 8.4 — Version tracking (which model is live) | `retrain_manager.py` |
| 8.5 — Rollback to previous version | `retrain_manager.py` |

## Pipeline flow

```
main_flow()
  │
  ├─ show current live model (get_current_live_version)
  │
  ├─ run_training_pipeline(trigger="scheduled")   ← always runs
  │
  ├─ monitoring_flow()  [Phase 7]                  ← drift check
  │     └─ if drift → run_training_pipeline(trigger="drift")
  │
  └─ check_data_volume_trigger()                   ← volume check
        └─ if growth ≥ 5% → run_training_pipeline(trigger="volume")
```

Every call to `run_training_pipeline()` runs the full Phase 5 sequence (ingest → feature engineering → train → validate → register), then — if the model passed validation — records the new version and updates the row-count baseline used by the volume trigger.

## Retraining triggers

**Scheduled** — runs once every time `main_flow()` executes (daily, per the Prefect deployment schedule).

**Drift-based** — delegates to Phase 7's `monitoring_flow()`, which compares the training and test distributions with Evidently. If drift is detected, the pipeline reruns.

**Data volume** — `check_data_volume_trigger()` compares the current row count in `churn_train_v1.parquet` against the row count recorded after the last successful training run. If the data has grown by **5% or more**, retraining is triggered.

> Drift detection logic itself belongs to Phase 7 — see the Phase 7 README for how the Evidently report and Prometheus drift gauge work. This phase only consumes that signal.

## Version tracking (8.4)

Every successful training run is recorded in a JSON registry (`VERSION_REGISTRY_PATH`, default `registry/version_registry.json`):

```json
{
  "current_live":  { "run_id": "...", "model_version": "7", "test_auc": 0.829, "trigger": "scheduled", "registered_at": "..." },
  "previous_live": { ... },
  "version_history": [ ... ]
}
```

- `current_live` — the model version currently promoted to `Production` in MLflow
- `previous_live` — the version before that, kept specifically to support rollback
- `version_history` — full append-only log of every version ever registered, with its trigger and AUC

Check the current state from the command line:
```bash
python retrain_manager.py status
python retrain_manager.py history
```

## Rollback (8.5)

If a newly promoted model underperforms in production, roll back to the previous version:

```bash
python retrain_manager.py rollback
```

This archives the current `Production` model in MLflow (moves it to `Archived`) and restores the previous version to `Production`, updating the registry to match. Rollback only supports **one level back** — there's no "redo" if you roll back twice in a row.

## Simulating new data (for testing the volume trigger)

`simulate_new_data.py` appends synthetic rows to `churn_train_v1.parquet` to simulate new customer data arriving, so the volume trigger can be tested without waiting for real data growth.

```bash
python simulate_new_data.py --rows 10000
```

**This file mutates `churn_train_v1.parquet` in place.** A timestamped backup is created automatically before any changes are written (`data/processed/backups/`). To undo a simulation run:

```bash
python simulate_new_data.py --restore-latest
```

The synthetic `churn` label is rebuilt using a logistic formula similar to (but not identical to) the one in `pipeline_steps.py` — this means repeated simulation runs will slightly shift the overall churn rate over time. Restore from backup between test runs if you need a clean baseline.

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `.env.example` below documents the required variables, but these scripts do **not** call `load_dotenv()` — a `.env` file alone will not be picked up. Set the variables directly in your shell before running (see below).

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/processed` | Folder with the train/val/test parquet files; also where `simulate_new_data.py` stores backups |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server |
| `VERSION_REGISTRY_PATH` | `registry/version_registry.json` | Where the live/previous/history version registry is stored |

## Running

Start MLflow first, then set `DATA_DIR` to wherever your parquet files actually live and run the pipeline.

**Windows (PowerShell):**
```powershell
$env:DATA_DIR = "C:/path/to/your/data/processed"
python main_flow.py
```

**macOS / Linux:**
```bash
export DATA_DIR="/path/to/your/data/processed"
python main_flow.py
```

If `DATA_DIR` is not set, it defaults to a relative `data/processed` folder — which only works if you run the script from a directory that actually has that subfolder.

## Files in this phase

| File | Purpose |
|---|---|
| `main_flow.py` | Orchestrates scheduled run + drift trigger + volume trigger |
| `pipeline_steps.py` | Shared with Phase 5 — ingest, features, train, validate, register |
| `retrain_manager.py` | Volume trigger, version tracking, rollback |
| `simulate_new_data.py` | Generates synthetic rows to test the volume trigger, with auto-backup |
| `debug_check.py` | Dev/diagnostic script — feature correlations, AUC sanity checks. Not part of the automated pipeline. |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Template for required environment variables |

## Known limitations

- Rollback supports only one level of history — rolling back twice loses the ability to "redo"
- `simulate_new_data.py`'s churn-generation formula has a different intercept than `pipeline_steps.py`'s, so repeated simulation runs gradually drift the synthetic churn rate
- `debug_check.py` is a development scratch file, not a maintained pipeline component — kept for reference only
- `approval_workflow.py` is present in the shared `phase5_automation` folder but belongs to **Phase 9** (governance), not this phase — see the Phase 9 README
