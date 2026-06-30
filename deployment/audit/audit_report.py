"""
Phase 9.4 - Audit Report Generator
Generates a compliance audit report from the audit_log table.

Usage:
    python audit/audit_report.py
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

DB_PATH     = os.environ.get("DB_PATH", "monitoring/predictions.db")
REPORT_DIR  = "monitoring/reports"


def load_audit_logs() -> pd.DataFrame:
    print("Loading audit logs...")
    if not os.path.exists(DB_PATH):
        print(f"  No DB found at {DB_PATH}.")
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM audit_log ORDER BY id", conn)
    conn.close()
    print(f"  Loaded {len(df)} audit records")
    return df


def load_compliance_events() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM compliance_events ORDER BY id", conn)
    conn.close()
    return df


def generate_report(audit_df: pd.DataFrame, events_df: pd.DataFrame):
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORT_DIR}/audit_report_{timestamp}.html"

    if audit_df.empty:
        total = churn = no_churn = high_risk = reviewed = 0
        avg_prob = avg_latency = 0.0
    else:
        total      = len(audit_df)
        churn      = int((audit_df["prediction"] == 1).sum())
        no_churn   = int((audit_df["prediction"] == 0).sum())
        high_risk  = int(audit_df["compliance_flag"].sum())
        reviewed   = int(audit_df["reviewed"].sum())
        avg_prob   = round(float(audit_df["churn_probability"].mean()), 4)
        avg_latency= round(float(audit_df["latency_ms"].mean()), 2)

    rows_html = ""
    if not audit_df.empty:
        for _, row in audit_df.iterrows():
            flag_color = "#b71c1c" if row["compliance_flag"] == 1 else "#1b5e20"
            flag_text  = "HIGH RISK" if row["compliance_flag"] == 1 else "Normal"
            reviewed_text = "Yes" if row["reviewed"] == 1 else "No"
            rows_html += f"""
            <tr>
                <td>{row['request_id'][:8]}...</td>
                <td>{row['timestamp'][:19]}</td>
                <td>{'Churn' if row['prediction']==1 else 'No Churn'}</td>
                <td>{row['churn_probability']}</td>
                <td>{row['risk_level']}</td>
                <td style="color:{flag_color};font-weight:bold">{flag_text}</td>
                <td>{row['latency_ms']}ms</td>
                <td>{reviewed_text}</td>
            </tr>"""

    events_html = ""
    if not events_df.empty:
        for _, row in events_df.iterrows():
            events_html += f"""
            <tr>
                <td>{row['timestamp'][:19]}</td>
                <td><b>{row['event_type']}</b></td>
                <td>{row['description']}</td>
                <td>{row['actor']}</td>
                <td>{row['model_version']}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Phase 9.4 — Audit Log Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; color: #333; }}
        h1 {{ color: #1F4E79; border-bottom: 3px solid #2E75B6; padding-bottom: 10px; }}
        h2 {{ color: #2E75B6; margin-top: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card .value {{ font-size: 2em; font-weight: bold; color: #1F4E79; }}
        .card .label {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #1F4E79; color: white; padding: 12px 10px; text-align: left; font-size: 0.85em; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; font-size: 0.82em; }}
        tr:hover {{ background: #f0f7ff; }}
        .alert-box {{ background: #fff3e0; border-left: 4px solid #f57c00; padding: 12px 20px; margin: 15px 0; border-radius: 4px; }}
        .info-box {{ background: #e3f2fd; border-left: 4px solid #1976d2; padding: 12px 20px; margin: 15px 0; border-radius: 4px; }}
        .meta {{ background: white; padding: 15px 20px; border-radius: 8px; margin: 15px 0; border: 1px solid #ddd; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Phase 9.4 — Prediction Audit Log Report</h1>
    <div class="meta">
        <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")} &nbsp;|&nbsp;
        <b>Model:</b> telecom-churn-logreg-v1 &nbsp;|&nbsp;
        <b>Project:</b> Telecom Churn MLOps &nbsp;|&nbsp;
        <b>Phase:</b> 9.4 — Audit Logs
    </div>

    <div class="info-box">
        This audit report provides a tamper-evident trail of all prediction requests,
        including input hashes, risk classifications, and human review status.
        Required for compliance in regulated industries.
    </div>

    <h2>Summary Statistics</h2>
    <div class="grid">
        <div class="card"><div class="value">{total}</div><div class="label">Total Predictions Audited</div></div>
        <div class="card"><div class="value" style="color:#b71c1c">{high_risk}</div><div class="label">High Risk Predictions</div></div>
        <div class="card"><div class="value" style="color:#1b5e20">{reviewed}</div><div class="label">Human Reviewed</div></div>
        <div class="card"><div class="value">{avg_prob}</div><div class="label">Avg Churn Probability</div></div>
        <div class="card"><div class="value">{churn}</div><div class="label">Churn Predictions</div></div>
        <div class="card"><div class="value">{no_churn}</div><div class="label">No Churn Predictions</div></div>
        <div class="card"><div class="value">{avg_latency}ms</div><div class="label">Avg Latency</div></div>
        <div class="card"><div class="value">{total - reviewed}</div><div class="label">Pending Review</div></div>
    </div>

    {"<div class='alert-box'><b>COMPLIANCE ALERT:</b> " + str(high_risk) + " high-risk predictions require human review.</div>" if high_risk > reviewed else ""}

    <h2>Prediction Audit Log</h2>
    <table>
        <thead>
            <tr>
                <th>Request ID</th><th>Timestamp</th><th>Prediction</th>
                <th>Probability</th><th>Risk Level</th><th>Compliance</th>
                <th>Latency</th><th>Reviewed</th>
            </tr>
        </thead>
        <tbody>{rows_html if rows_html else '<tr><td colspan="8" style="text-align:center;color:#999">No predictions logged yet</td></tr>'}</tbody>
    </table>

    <h2>Compliance Events</h2>
    <table>
        <thead>
            <tr><th>Timestamp</th><th>Event Type</th><th>Description</th><th>Actor</th><th>Model Version</th></tr>
        </thead>
        <tbody>{events_html if events_html else '<tr><td colspan="5" style="text-align:center;color:#999">No compliance events logged yet</td></tr>'}</tbody>
    </table>
</body>
</html>"""

    with open(report_path, "w") as f:
        f.write(html)
    print(f"  Audit report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 9.4 - Audit Report Generator")
    print("=" * 60)
    audit_df   = load_audit_logs()
    events_df  = load_compliance_events()
    report_path = generate_report(audit_df, events_df)
    print(f"\nReport: {report_path}")
    print("Open in browser to view the full audit trail.")
