"""
Phase 7 - Retraining Trigger
Checks drift log and prediction stats to decide if retraining is needed.

Usage:
    python monitoring/retrain_trigger.py
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

DB_PATH        = os.environ.get("DB_PATH", "monitoring/predictions.db")
DRIFT_LOG      = "monitoring/drift_log.csv"
ALERT_LOG      = "monitoring/alerts.log"

DRIFT_THRESHOLD        = 0.20
CHURN_RATE_BASELINE    = 0.15
CHURN_RATE_THRESHOLD   = 0.10
LATENCY_THRESHOLD_MS   = 500
MIN_PREDICTIONS        = 10


def log_alert(message: str):
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {message}"
    print(f"  ALERT: {line}")
    with open(ALERT_LOG, "a") as f:
        f.write(line + "\n")


def check_drift_threshold() -> bool:
    print("Checking drift threshold...")
    if not os.path.exists(DRIFT_LOG):
        print("  No drift log found. Run drift_detection.py first.")
        return False
    df = pd.read_csv(DRIFT_LOG)
    if df.empty:
        return False
    latest = df.iloc[-1]
    drift_share = float(latest["drift_share"])
    print(f"  Latest drift share: {drift_share:.2%}")
    if drift_share > DRIFT_THRESHOLD:
        log_alert(f"DATA DRIFT: {drift_share:.2%} drifted (threshold: {DRIFT_THRESHOLD:.2%}). Retraining recommended.")
        return True
    print(f"  OK: Drift within threshold")
    return False


def check_prediction_drift() -> bool:
    print("\nChecking prediction drift...")
    if not os.path.exists(DB_PATH):
        print("  No predictions DB found.")
        return False
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT prediction FROM predictions", conn)
    conn.close()
    if len(df) < MIN_PREDICTIONS:
        print(f"  Not enough predictions yet ({len(df)}/{MIN_PREDICTIONS}).")
        return False
    churn_rate = df["prediction"].mean()
    drift = abs(churn_rate - CHURN_RATE_BASELINE)
    print(f"  Baseline: {CHURN_RATE_BASELINE:.2%} | Production: {churn_rate:.2%} | Drift: {drift:.2%}")
    if drift > CHURN_RATE_THRESHOLD:
        log_alert(f"PREDICTION DRIFT: Churn rate shifted from {CHURN_RATE_BASELINE:.2%} to {churn_rate:.2%}.")
        return True
    print(f"  OK: Prediction drift within threshold")
    return False


def check_latency() -> bool:
    print("\nChecking latency...")
    if not os.path.exists(DB_PATH):
        print("  No predictions DB found.")
        return False
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT latency_ms FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    if df.empty:
        return False
    avg_latency = df["latency_ms"].mean()
    print(f"  Avg latency (last 100): {avg_latency:.1f}ms")
    if avg_latency > LATENCY_THRESHOLD_MS:
        log_alert(f"LATENCY SPIKE: {avg_latency:.1f}ms exceeds {LATENCY_THRESHOLD_MS}ms threshold.")
        return True
    print(f"  OK: Latency within threshold")
    return False


def check_missing_features() -> bool:
    print("\nChecking feature completeness...")
    if not os.path.exists(DB_PATH):
        print("  No predictions DB found.")
        return False
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT missing_count FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    if df.empty:
        return False
    avg_missing = df["missing_count"].mean()
    print(f"  Avg missing features (last 100): {avg_missing:.1f}/27")
    if avg_missing > 20:
        log_alert(f"DATA QUALITY: Avg {avg_missing:.1f}/27 features missing per request.")
        return True
    print(f"  OK: Feature completeness acceptable")
    return False


def print_summary(triggers: dict):
    print("\n" + "=" * 60)
    print("RETRAINING TRIGGER SUMMARY")
    print("=" * 60)
    any_triggered = any(triggers.values())
    for check, triggered in triggers.items():
        status = "TRIGGERED" if triggered else "OK"
        print(f"  {check:<30} {status}")
    print()
    if any_triggered:
        print("ACTION REQUIRED: Retrain the model with fresh data.")
        print(f"Alert log: {ALERT_LOG}")
    else:
        print("Model is healthy. No retraining needed.")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 7 - Retraining Trigger Check")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    triggers = {
        "Data Drift":       check_drift_threshold(),
        "Prediction Drift": check_prediction_drift(),
        "Latency Spike":    check_latency(),
        "Missing Features": check_missing_features(),
    }
    print_summary(triggers)
