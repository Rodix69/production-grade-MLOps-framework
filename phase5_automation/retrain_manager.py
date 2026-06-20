"""
retrain_manager.py — Phase 8
Handles all continuous training and retraining logic:
  - 8.1 New data volume trigger
  - 8.4 Version tracking — which model is currently live
  - 8.5 Rollback capability — revert to previous model version

Usage:
    from retrain_manager import (
        check_data_volume_trigger,
        save_version_record,
        get_current_live_version,
        rollback_to_previous_version,
    )
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────
BASE             = "C:/phase 5 pipeline MLOps/phase2_model"
TRAIN_PATH       = f"{BASE}/churn_train_v1.parquet"
VERSION_FILE     = "C:/phase 5 pipeline MLOps/phase3_automation/version_registry.json"
VOLUME_THRESHOLD = 0.05   # retrain if data grows by more than 5%
MODEL_NAME       = "telecom_churn_champion"
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── 8.1 Data Volume Trigger ──────────────────────────────────────────────────

def check_data_volume_trigger() -> tuple[bool, int, int]:
    """
    Compare current row count against last recorded count.
    Returns (should_retrain, current_rows, last_rows).
    Triggers retraining if data has grown by more than VOLUME_THRESHOLD.
    """
    current_rows = len(pd.read_parquet(TRAIN_PATH))
    last_rows    = _load_last_row_count()

    if last_rows == 0:
        print(f"No previous row count found — recording {current_rows} rows as baseline")
        _save_last_row_count(current_rows)
        return False, current_rows, 0

    growth = (current_rows - last_rows) / last_rows
    print(f"Data volume check:")
    print(f"  Last training rows : {last_rows}")
    print(f"  Current rows       : {current_rows}")
    print(f"  Growth             : +{growth * 100:.2f}%")
    print(f"  Threshold          : {VOLUME_THRESHOLD * 100:.0f}%")

    if growth >= VOLUME_THRESHOLD:
        print(f"  => TRIGGER: Data grew by {growth*100:.1f}% — retraining required")
        return True, current_rows, last_rows
    else:
        print(f"  => OK: Growth below threshold — no retraining needed")
        return False, current_rows, last_rows


def update_row_count_after_training() -> None:
    """Call this after a successful training run to update the baseline."""
    current_rows = len(pd.read_parquet(TRAIN_PATH))
    _save_last_row_count(current_rows)
    print(f"Row count baseline updated to {current_rows}")


def _load_last_row_count() -> int:
    registry = _load_registry()
    return registry.get("last_training_row_count", 0)


def _save_last_row_count(count: int) -> None:
    registry = _load_registry()
    registry["last_training_row_count"] = count
    _save_registry(registry)


# ─── 8.4 Version Tracking ─────────────────────────────────────────────────────

def save_version_record(run_id: str, model_version: str, test_auc: float,
                        trigger: str = "scheduled") -> None:
    """
    Record which model version is currently live.
    trigger: 'scheduled' | 'drift' | 'volume' | 'manual'
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    registry = _load_registry()

    # Move current live to previous before updating
    if "current_live" in registry and registry["current_live"]:
        registry["previous_live"] = registry["current_live"].copy()

    registry["current_live"] = {
        "run_id":        run_id,
        "model_version": str(model_version),
        "test_auc":      round(test_auc, 4),
        "registered_at": datetime.now().isoformat(),
        "trigger":       trigger,
    }

    # Keep a full history of all versions
    if "version_history" not in registry:
        registry["version_history"] = []
    registry["version_history"].append(registry["current_live"].copy())

    _save_registry(registry)

    print(f"Version record saved:")
    print(f"  Model version : {model_version}")
    print(f"  Run ID        : {run_id}")
    print(f"  Test AUC      : {test_auc:.4f}")
    print(f"  Trigger       : {trigger}")
    print(f"  Registered at : {registry['current_live']['registered_at']}")


def get_current_live_version() -> dict:
    """Return info about the currently live model version."""
    registry = _load_registry()
    current  = registry.get("current_live", {})

    if not current:
        print("No live model version recorded yet")
        return {}

    print(f"Current live model:")
    print(f"  Model version : {current.get('model_version')}")
    print(f"  Run ID        : {current.get('run_id')}")
    print(f"  Test AUC      : {current.get('test_auc')}")
    print(f"  Trigger       : {current.get('trigger')}")
    print(f"  Registered at : {current.get('registered_at')}")
    return current


def get_version_history() -> list:
    """Return full history of all registered model versions."""
    registry = _load_registry()
    history  = registry.get("version_history", [])

    print(f"\nVersion history ({len(history)} versions):")
    print(f"{'Version':<10} {'AUC':<8} {'Trigger':<12} {'Registered At'}")
    print("-" * 60)
    for v in history:
        print(f"  v{v.get('model_version'):<8} "
              f"{v.get('test_auc'):<8} "
              f"{v.get('trigger'):<12} "
              f"{v.get('registered_at', '')[:19]}")
    return history


# ─── 8.5 Rollback ─────────────────────────────────────────────────────────────

def rollback_to_previous_version() -> bool:
    """
    Roll back production to the previous model version in MLflow.
    Returns True if rollback succeeded, False otherwise.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    registry = _load_registry()

    previous = registry.get("previous_live", {})
    current  = registry.get("current_live", {})

    if not previous:
        print("No previous version available — cannot roll back")
        return False

    prev_version = previous.get("model_version")
    curr_version = current.get("model_version")

    print(f"Rolling back:")
    print(f"  Current version  : v{curr_version} (AUC {current.get('test_auc')})")
    print(f"  Previous version : v{prev_version} (AUC {previous.get('test_auc')})")

    client = mlflow.tracking.MlflowClient()

    try:
        # Archive the current production version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=curr_version,
            stage="Archived"
        )
        print(f"  Archived v{curr_version}")

        # Restore previous version to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prev_version,
            stage="Production"
        )
        print(f"  Restored v{prev_version} to Production")

        # Swap current and previous in registry
        registry["current_live"]  = previous.copy()
        registry["current_live"]["registered_at"] = datetime.now().isoformat()
        registry["current_live"]["trigger"]       = "rollback"
        registry["previous_live"] = current.copy()

        # Log rollback in history
        if "version_history" not in registry:
            registry["version_history"] = []
        registry["version_history"].append({
            **previous,
            "registered_at": registry["current_live"]["registered_at"],
            "trigger":       "rollback",
        })

        _save_registry(registry)
        print(f"Rollback complete — v{prev_version} is now live")
        return True

    except Exception as e:
        print(f"Rollback failed: {e}")
        return False


def get_rollback_info() -> dict:
    """Show what version is available for rollback."""
    registry = _load_registry()
    previous = registry.get("previous_live", {})

    if not previous:
        print("No previous version available for rollback")
        return {}

    print(f"Available rollback target:")
    print(f"  Model version : v{previous.get('model_version')}")
    print(f"  Test AUC      : {previous.get('test_auc')}")
    print(f"  Registered at : {previous.get('registered_at', '')[:19]}")
    return previous


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _load_registry() -> dict:
    path = Path(VERSION_FILE)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(data: dict) -> None:
    path = Path(VERSION_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Standalone usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        get_current_live_version()
        print()
        get_rollback_info()

    elif cmd == "history":
        get_version_history()

    elif cmd == "rollback":
        rollback_to_previous_version()

    elif cmd == "check-volume":
        check_data_volume_trigger()

    else:
        print("Usage: python retrain_manager.py [status|history|rollback|check-volume]")
