"""
Phase 7 - Drift Detection
Compares training data vs logged predictions using Evidently AI.

Usage:
    python monitoring/drift_detection.py
"""

import pandas as pd
import sqlite3
import os
import json
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

DB_PATH    = os.environ.get("DB_PATH", "monitoring/predictions.db")
TRAIN_PATH = "../feature_store/data/churn_train_v1.parquet"
REPORT_DIR = "monitoring/reports"
DRIFT_LOG  = "monitoring/drift_log.csv"
DRIFT_THRESHOLD = 0.20

NUMERICAL_FEATURES = [
    "age", "num_dependents", "estimated_salary",
    "calls_made", "sms_sent", "data_used", "tenure_days",
    "total_activity", "avg_data_per_day", "avg_calls_per_day",
    "avg_sms_per_day", "engagement_score", "low_activity_flag",
    "high_value_user", "partner_med_calls", "partner_med_data",
    "partner_med_sms", "calls_vs_partner", "data_vs_partner",
    "is_new_customer", "calls_intensity", "data_intensity",
]


def load_reference_data() -> pd.DataFrame:
    print("Loading reference (training) data...")
    df = pd.read_parquet(TRAIN_PATH)
    available = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df = df[available].sample(min(5000, len(df)), random_state=42)
    print(f"  Reference data: {df.shape}")
    return df


def load_production_data() -> pd.DataFrame:
    print("Loading production prediction logs...")
    if not os.path.exists(DB_PATH):
        print(f"  No DB found at {DB_PATH}.")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    logs = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 1000", conn)
    conn.close()

    if logs.empty:
        print("  No predictions logged yet.")
        return pd.DataFrame()

    records = []
    for _, row in logs.iterrows():
        try:
            features = json.loads(row["input_features"])
            records.append(features)
        except Exception:
            continue

    df = pd.DataFrame(records)
    available = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df = df[available] if available else df
    print(f"  Production data: {df.shape}")
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    print("\nRunning Evidently drift analysis...")
    os.makedirs(REPORT_DIR, exist_ok=True)

    common_cols = [c for c in reference.columns if c in current.columns]
    ref = reference[common_cols].reset_index(drop=True)
    cur = current[common_cols].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])

    # run() returns a Snapshot object in evidently 0.7.x
    snapshot = report.run(reference_data=ref, current_data=cur)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORT_DIR}/drift_report_{timestamp}.html"
    snapshot.save_html(report_path)
    print(f"  Drift report saved: {report_path}")

    result  = snapshot.dict()
    metrics = result.get("metrics", [])

    drift_share  = 0
    drifted_cols = 0
    for m in metrics:
        r = m.get("result", {})
        if "drift_share" in r:
            drift_share  = r["drift_share"]
            drifted_cols = r.get("number_of_drifted_columns", 0)
            break

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "drift_share":  round(drift_share, 4),
        "drifted_cols": drifted_cols,
        "total_cols":   len(common_cols),
        "alert":        drift_share > DRIFT_THRESHOLD,
        "report_path":  report_path,
    }
    return summary


def log_drift_result(summary: dict):
    os.makedirs(os.path.dirname(DRIFT_LOG), exist_ok=True)
    df = pd.DataFrame([summary])
    if os.path.exists(DRIFT_LOG):
        df.to_csv(DRIFT_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(DRIFT_LOG, index=False)
    print(f"  Drift log updated: {DRIFT_LOG}")


def check_prediction_drift():
    print("\nChecking prediction drift...")
    if not os.path.exists(DB_PATH):
        print("  No DB found.")
        return
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT prediction FROM predictions", conn)
    conn.close()
    if len(df) < 5:
        print(f"  Not enough predictions yet ({len(df)}).")
        return
    churn_rate = df["prediction"].mean()
    baseline   = 0.15
    print(f"  Training churn rate:    {baseline:.2%}")
    print(f"  Production churn rate:  {churn_rate:.2%}")
    print(f"  Drift:                  {abs(churn_rate - baseline):.2%}")
    if abs(churn_rate - baseline) > 0.10:
        print("  ALERT: Prediction drift detected!")
    else:
        print("  OK: Prediction drift within normal range")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 7 - Drift Detection Report")
    print("=" * 60)

    reference = load_reference_data()
    current   = load_production_data()

    if current.empty:
        print("\nNo production data available yet.")
        check_prediction_drift()
    else:
        summary = run_drift_report(reference, current)
        log_drift_result(summary)
        check_prediction_drift()

        print("\n" + "=" * 60)
        print("DRIFT SUMMARY")
        print("=" * 60)
        print(f"Drift share:     {summary['drift_share']:.2%}")
        print(f"Drifted columns: {summary['drifted_cols']} / {summary['total_cols']}")
        print(f"Alert:           {'YES - RETRAINING RECOMMENDED' if summary['alert'] else 'NO - Model is healthy'}")
        print(f"Report:          {summary['report_path']}")