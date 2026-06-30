"""
lineage_tracker.py — Phase 9.1
Tracks end-to-end model lineage: data → features → model → deployment.
Every training run logs a complete lineage record linking all artifacts.

Usage:
    from lineage_tracker import record_lineage, get_lineage, print_lineage_report
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────
LINEAGE_FILE = "C:/phase 5 pipeline MLOps/phase5_automation/lineage_registry.json"
BASE         = "C:/phase 5 pipeline MLOps/phase2_model"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── 9.1 Record Lineage ───────────────────────────────────────────────────────

def record_lineage(
    run_id:        str,
    model_version: str,
    feature_cols:  list,
    train_rows:    int,
    val_auc:       float,
    test_auc:      float,
    trigger:       str = "scheduled",
) -> dict:
    """
    Record complete lineage for a training run.
    Links data sources → feature definitions → model run → deployment version.
    """
    # Compute data fingerprints
    train_path = f"{BASE}/churn_train_v1.parquet"
    val_path   = f"{BASE}/churn_val_v1.parquet"
    test_path  = f"{BASE}/churn_test_v1.parquet"

    data_sources = {
        "train": {
            "path":     train_path,
            "rows":     train_rows,
            "modified": _file_modified_time(train_path),
            "size_kb":  _file_size_kb(train_path),
        },
        "val": {
            "path":     val_path,
            "rows":     _count_rows(val_path),
            "modified": _file_modified_time(val_path),
            "size_kb":  _file_size_kb(val_path),
        },
        "test": {
            "path":     test_path,
            "rows":     _count_rows(test_path),
            "modified": _file_modified_time(test_path),
            "size_kb":  _file_size_kb(test_path),
        },
    }

    # Feature definitions
    feature_definitions = {
        "raw_features": [
            "age", "num_dependents", "estimated_salary",
            "calls_made", "sms_sent", "data_used", "tenure_days"
        ],
        "engineered_features": [
            "total_activity", "avg_data_per_day", "avg_calls_per_day",
            "avg_sms_per_day", "engagement_score", "low_activity_flag",
            "high_value_user", "partner_med_calls", "partner_med_data",
            "partner_med_sms", "calls_vs_partner", "data_vs_partner",
            "is_new_customer", "calls_intensity", "data_intensity"
        ],
        "dropped_columns": [
            "date_of_registration", "customer_id", "pincode",
            "city", "state", "telecom_partner", "gender"
        ],
        "total_features_used": len(feature_cols),
        "feature_list":        feature_cols,
    }

    # Model info from MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        run_info = client.get_run(run_id)
        model_params = run_info.data.params
    except Exception:
        model_params = {}

    lineage_record = {
        "lineage_id":          f"lineage_{run_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "created_at":          datetime.now().isoformat(),
        "trigger":             trigger,
        "data_sources":        data_sources,
        "feature_definitions": feature_definitions,
        "model": {
            "run_id":        run_id,
            "model_version": str(model_version),
            "algorithm":     "RandomForestClassifier",
            "parameters":    model_params,
            "metrics": {
                "val_auc":  round(val_auc, 4),
                "test_auc": round(test_auc, 4),
            },
            "mlflow_url": f"{MLFLOW_URI}/#/experiments/1/runs/{run_id}",
        },
        "deployment": {
            "registry_name":  "telecom_churn_champion",
            "model_version":  str(model_version),
            "stage":          "Production",
            "promoted_at":    datetime.now().isoformat(),
        },
    }

    # Log lineage tags back to MLflow run
    try:
        client.set_tag(run_id, "lineage_id",     lineage_record["lineage_id"])
        client.set_tag(run_id, "model_version",  str(model_version))
        client.set_tag(run_id, "trigger",        trigger)
        client.set_tag(run_id, "train_rows",     str(train_rows))
        client.set_tag(run_id, "feature_count",  str(len(feature_cols)))
    except Exception as e:
        print(f"Warning: Could not tag MLflow run: {e}")

    # Save to local registry
    registry = _load_registry()
    if "lineage_records" not in registry:
        registry["lineage_records"] = []
    registry["lineage_records"].append(lineage_record)
    _save_registry(registry)

    print(f"Lineage recorded: {lineage_record['lineage_id']}")
    return lineage_record


def get_lineage(run_id: str = None, model_version: str = None) -> dict:
    """Retrieve lineage record by run_id or model_version."""
    registry = _load_registry()
    records  = registry.get("lineage_records", [])

    if run_id:
        for r in records:
            if r["model"]["run_id"] == run_id:
                return r
    if model_version:
        for r in records:
            if r["model"]["model_version"] == str(model_version):
                return r

    return records[-1] if records else {}


def print_lineage_report(run_id: str = None, model_version: str = None) -> None:
    """Print a human-readable lineage report."""
    record = get_lineage(run_id, model_version)
    if not record:
        print("No lineage record found.")
        return

    print("\n" + "=" * 60)
    print("MODEL LINEAGE REPORT")
    print("=" * 60)
    print(f"Lineage ID  : {record.get('lineage_id')}")
    print(f"Created At  : {record.get('created_at', '')[:19]}")
    print(f"Trigger     : {record.get('trigger')}")

    print("\n--- DATA SOURCES ---")
    for split, info in record.get("data_sources", {}).items():
        print(f"  {split:6}: {info['rows']:>7} rows | "
              f"{info['size_kb']:>6.0f} KB | "
              f"modified {info['modified']}")

    print("\n--- FEATURES ---")
    fd = record.get("feature_definitions", {})
    print(f"  Raw features        : {len(fd.get('raw_features', []))}")
    print(f"  Engineered features : {len(fd.get('engineered_features', []))}")
    print(f"  Total used          : {fd.get('total_features_used', 0)}")
    print(f"  Dropped (PII/ID)    : {fd.get('dropped_columns', [])}")

    print("\n--- MODEL ---")
    m = record.get("model", {})
    print(f"  Algorithm    : {m.get('algorithm')}")
    print(f"  Run ID       : {m.get('run_id')}")
    print(f"  Version      : v{m.get('model_version')}")
    print(f"  Val AUC      : {m.get('metrics', {}).get('val_auc')}")
    print(f"  Test AUC     : {m.get('metrics', {}).get('test_auc')}")
    print(f"  MLflow URL   : {m.get('mlflow_url')}")

    print("\n--- DEPLOYMENT ---")
    d = record.get("deployment", {})
    print(f"  Registry     : {d.get('registry_name')}")
    print(f"  Version      : v{d.get('model_version')}")
    print(f"  Stage        : {d.get('stage')}")
    print(f"  Promoted At  : {d.get('promoted_at', '')[:19]}")
    print("=" * 60)


def get_full_lineage_history() -> None:
    """Print summary of all lineage records."""
    registry = _load_registry()
    records  = registry.get("lineage_records", [])

    print(f"\nLineage History ({len(records)} records):")
    print(f"{'Version':<10} {'AUC':<8} {'Trigger':<12} {'Train Rows':<12} {'Created At'}")
    print("-" * 65)
    for r in records:
        print(f"  v{r['model']['model_version']:<8} "
              f"{r['model']['metrics']['test_auc']:<8} "
              f"{r['trigger']:<12} "
              f"{r['data_sources']['train']['rows']:<12} "
              f"{r['created_at'][:19]}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _file_modified_time(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def _file_size_kb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024
    except Exception:
        return 0.0


def _count_rows(path: str) -> int:
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _load_registry() -> dict:
    path = Path(LINEAGE_FILE)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(data: dict) -> None:
    path = Path(LINEAGE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Standalone usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "latest"

    if cmd == "latest":
        print_lineage_report()
    elif cmd == "history":
        get_full_lineage_history()
    elif cmd == "run":
        print_lineage_report(run_id=sys.argv[2])
    elif cmd == "version":
        print_lineage_report(model_version=sys.argv[2])
    else:
        print("Usage: python lineage_tracker.py [latest|history|run <run_id>|version <v>]")
