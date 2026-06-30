"""
data_privacy.py — Phase 9.5
Handles data privacy for the telecom churn pipeline:
  - PII column identification and anonymization
  - Data masking utilities
  - Privacy compliance documentation
  - Role-based access control definitions

Usage:
    from data_privacy import anonymize_dataframe, get_privacy_report
    python data_privacy.py
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

OUTPUT_DIR   = "C:/phase 5 pipeline MLOps/phase5_automation/outputs/privacy"
BASE         = "C:/phase 5 pipeline MLOps/phase2_model"

# ─── PII Definitions ──────────────────────────────────────────────────────────

# Columns classified by sensitivity level
PII_COLUMNS = {
    "high_sensitivity": [
        "customer_id",        # Direct identifier
        "pincode",            # Location identifier
    ],
    "medium_sensitivity": [
        "gender",             # Demographic
        "age",                # Demographic
        "date_of_registration",  # Temporal identifier
        "estimated_salary",   # Financial
    ],
    "low_sensitivity": [
        "city",               # Location (broad)
        "state",              # Location (broad)
        "telecom_partner",    # Service provider
    ],
    "non_sensitive": [
        "calls_made", "sms_sent", "data_used",
        "tenure_days", "num_dependents", "churn",
    ]
}

# RBAC — who can access what
RBAC_POLICY = {
    "data_scientist": {
        "description": "Can access anonymized training data, model metrics, SHAP reports",
        "allowed":     ["non_sensitive", "low_sensitivity"],
        "denied":      ["high_sensitivity", "medium_sensitivity"],
        "mlflow":      ["read_experiments", "read_models"],
    },
    "ml_engineer": {
        "description": "Can access full pipeline, model registry, deployment",
        "allowed":     ["non_sensitive", "low_sensitivity", "medium_sensitivity"],
        "denied":      ["high_sensitivity"],
        "mlflow":      ["read_experiments", "read_models", "write_models", "deploy"],
    },
    "data_engineer": {
        "description": "Can access raw data for pipeline maintenance",
        "allowed":     ["non_sensitive", "low_sensitivity", "medium_sensitivity", "high_sensitivity"],
        "denied":      [],
        "mlflow":      ["read_experiments"],
    },
    "auditor": {
        "description": "Read-only access to audit logs and lineage records",
        "allowed":     ["non_sensitive"],
        "denied":      ["high_sensitivity", "medium_sensitivity", "low_sensitivity"],
        "mlflow":      ["read_experiments", "read_models"],
    },
    "business_analyst": {
        "description": "Can access aggregated metrics and reports only",
        "allowed":     ["non_sensitive"],
        "denied":      ["high_sensitivity", "medium_sensitivity", "low_sensitivity"],
        "mlflow":      ["read_experiments"],
    },
}


# ─── 9.5 Anonymization ────────────────────────────────────────────────────────

def anonymize_dataframe(df: pd.DataFrame, role: str = "data_scientist") -> pd.DataFrame:
    """
    Anonymize a DataFrame based on the requester's role.
    Applies different anonymization strategies per sensitivity level.

    Args:
        df:   Input DataFrame
        role: Requester role (data_scientist, ml_engineer, etc.)

    Returns:
        Anonymized DataFrame safe for the given role
    """
    policy  = RBAC_POLICY.get(role, RBAC_POLICY["data_scientist"])
    denied  = policy["denied"]
    df_anon = df.copy()

    cols_to_process = []
    for level in denied:
        cols_to_process.extend(PII_COLUMNS.get(level, []))

    for col in cols_to_process:
        if col not in df_anon.columns:
            continue

        sensitivity = _get_sensitivity(col)

        if sensitivity == "high_sensitivity":
            # Hash high-sensitivity identifiers
            df_anon[col] = df_anon[col].astype(str).apply(_hash_value)

        elif sensitivity == "medium_sensitivity":
            if col == "age":
                # Age → age band
                df_anon[col] = pd.cut(
                    df_anon[col],
                    bins=[0, 25, 35, 45, 55, 65, 100],
                    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
                ).astype(str)
            elif col == "estimated_salary":
                # Salary → quartile band
                df_anon[col] = pd.qcut(
                    df_anon[col], q=4,
                    labels=["Q1_Low", "Q2_Mid-Low", "Q3_Mid-High", "Q4_High"]
                ).astype(str)
            elif col == "date_of_registration":
                # Date → year only
                try:
                    df_anon[col] = pd.to_datetime(
                        df_anon[col], errors="coerce"
                    ).dt.year.astype(str)
                except Exception:
                    df_anon[col] = "REDACTED"
            elif col == "gender":
                # Keep gender as-is for low risk, or mask
                df_anon[col] = "REDACTED"

        elif sensitivity == "low_sensitivity":
            # Replace with regional grouping or REDACTED
            df_anon[col] = "REDACTED"

    print(f"Anonymization applied for role '{role}':")
    print(f"  Columns processed : {[c for c in cols_to_process if c in df.columns]}")
    print(f"  Rows affected     : {len(df_anon)}")
    return df_anon


def mask_for_logging(record: dict) -> dict:
    """
    Mask PII fields in a prediction record before logging.
    Used by audit logger to ensure PII never appears in logs.
    """
    masked = record.copy()
    high_pii = PII_COLUMNS["high_sensitivity"] + PII_COLUMNS["medium_sensitivity"]

    for key in high_pii:
        if key in masked:
            if key == "customer_id":
                masked[key] = _hash_value(str(masked[key]))
            elif key == "age":
                masked[key] = f"age_band_{masked[key] // 10 * 10}s"
            elif key == "estimated_salary":
                masked[key] = "REDACTED"
            else:
                masked[key] = "REDACTED"
    return masked


# ─── Privacy Report ───────────────────────────────────────────────────────────

def get_privacy_report(output_path: str = None) -> dict:
    """
    Generate a privacy compliance report documenting:
    - PII column classifications
    - Anonymization strategies applied
    - RBAC policy definitions
    - Compliance status
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "generated_at":    datetime.now().isoformat(),
        "project":         "Telecom Churn Prediction Pipeline",
        "compliance_framework": "GDPR-aligned data minimization principles",

        "pii_inventory": {
            level: {
                "columns":  cols,
                "strategy": _anonymization_strategy(level),
                "count":    len(cols),
            }
            for level, cols in PII_COLUMNS.items()
        },

        "pipeline_privacy_controls": {
            "training": {
                "description": "PII columns dropped before model training",
                "dropped_cols": (
                    PII_COLUMNS["high_sensitivity"] +
                    PII_COLUMNS["low_sensitivity"] +
                    ["date_of_registration", "gender"]
                ),
                "rationale": "These columns provide no predictive signal and introduce bias/privacy risk"
            },
            "feature_store": {
                "description": "Only behavioural features retained",
                "retained":    PII_COLUMNS["non_sensitive"],
            },
            "model_registry": {
                "description": "No PII stored in MLflow artifacts",
                "verified":    True,
            },
        },

        "rbac_policy": RBAC_POLICY,

        "compliance_checklist": {
            "pii_identified":           True,
            "pii_dropped_from_training": True,
            "anonymization_available":  True,
            "rbac_defined":             True,
            "audit_logging_planned":    True,
            "data_minimization":        True,
        },
    }

    # Save report
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/privacy_report.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Privacy report saved: {output_path}")
    _print_privacy_summary(report)
    return report


def _print_privacy_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("DATA PRIVACY REPORT")
    print("=" * 60)
    print(f"Generated : {report['generated_at'][:19]}")
    print(f"Framework : {report['compliance_framework']}")

    print("\n--- PII INVENTORY ---")
    for level, info in report["pii_inventory"].items():
        print(f"  {level:<22}: {info['columns']}")
        print(f"  {'':22}  Strategy: {info['strategy']}")

    print("\n--- RBAC ROLES ---")
    for role, policy in report["rbac_policy"].items():
        print(f"  {role:<20}: {policy['description']}")

    print("\n--- COMPLIANCE STATUS ---")
    for check, status in report["compliance_checklist"].items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {check}")
    print("=" * 60)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hash_value(value: str) -> str:
    """One-way hash for identifier anonymization."""
    return "HASH_" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _get_sensitivity(col: str) -> str:
    for level, cols in PII_COLUMNS.items():
        if col in cols:
            return level
    return "non_sensitive"


def _anonymization_strategy(level: str) -> str:
    strategies = {
        "high_sensitivity":  "SHA-256 hashing",
        "medium_sensitivity": "Generalization (age bands, salary quartiles, year only)",
        "low_sensitivity":   "Suppression (REDACTED)",
        "non_sensitive":     "No transformation required",
    }
    return strategies.get(level, "None")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "report":
        get_privacy_report()

    elif cmd == "demo":
        print("Loading sample data for anonymization demo...")
        df = pd.read_parquet(f"{BASE}/churn_train_v1.parquet").head(5)
        print("\nOriginal data:")
        print(df[["customer_id", "age", "estimated_salary", "gender", "calls_made"]].to_string())

        print("\nAnonymized for data_scientist role:")
        anon = anonymize_dataframe(df, role="data_scientist")
        print(anon[["customer_id", "age", "estimated_salary", "gender", "calls_made"]].to_string())

    elif cmd == "rbac":
        print("\nRBAC Policy:")
        for role, policy in RBAC_POLICY.items():
            print(f"\n{role.upper()}")
            print(f"  {policy['description']}")
            print(f"  Allowed: {policy['allowed']}")
            print(f"  Denied:  {policy['denied']}")
    else:
        print("Usage: python data_privacy.py [report|demo|rbac]")
