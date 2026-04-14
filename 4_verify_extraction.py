"""
Step 4 — Verify Extraction Output
===================================
Run after 3_extract_from_api.py to confirm everything
looks right before handing data to Phase 3.

Usage:
    python 4_verify_extraction.py
"""

import json, logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_PATH   = Path("data/raw/telecom_churn_raw_v1.parquet")
TRAIN_PATH = Path("data/processed/churn_train_v1.parquet")
VAL_PATH   = Path("data/processed/churn_val_v1.parquet")
TEST_PATH  = Path("data/processed/churn_test_v1.parquet")
META_PATH  = Path("data/raw/extraction_meta_v1.json")


def load_and_check():
    log.info("Loading all output files...")

    with open(META_PATH) as f:
        meta = json.load(f)

    df    = pd.read_parquet(RAW_PATH)
    train = pd.read_parquet(TRAIN_PATH)
    val   = pd.read_parquet(VAL_PATH)
    test  = pd.read_parquet(TEST_PATH)

    print("\n" + "="*60)
    print("EXTRACTION VERIFICATION REPORT")
    print("="*60)

    print(f"\n{'Source':<12} {'Rows':>8} {'Churn%':>8} {'Date from':<14} {'Date to':<14}")
    print("-"*60)
    for name, d in [("Full",train.append(val).append(test) if False else df),
                    ("Train", train), ("Val", val), ("Test", test)]:
        print(
            f"{name:<12} "
            f"{len(d):>8,} "
            f"{d['churn'].mean():>7.1%} "
            f"{str(d['date_of_registration'].min().date()):<14} "
            f"{str(d['date_of_registration'].max().date()):<14}"
        )

    print(f"\n── Column summary ──────────────────────────────────────")
    print(df.dtypes.to_string())

    print(f"\n── Null counts ─────────────────────────────────────────")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No nulls found in any column.")
    else:
        print(nulls[nulls > 0].to_string())

    print(f"\n── Numeric ranges ──────────────────────────────────────")
    num_cols = ["age","estimated_salary","calls_made",
                "sms_sent","data_used","tenure_days"]
    print(df[num_cols].describe().round(1).to_string())

    print(f"\n── Telecom partner distribution ────────────────────────")
    print(df["telecom_partner"].value_counts().to_string())

    print(f"\n── Churn by partner ────────────────────────────────────")
    print(df.groupby("telecom_partner")["churn"].mean().round(3).to_string())

    print(f"\n── Churn by gender ─────────────────────────────────────")
    print(df.groupby("gender")["churn"].mean().round(3).to_string())

    print(f"\n── Top 5 states by customer count ──────────────────────")
    print(df["state"].value_counts().head(5).to_string())

    print(f"\n── Metadata summary ────────────────────────────────────")
    print(f"  Version:         {meta['version']}")
    print(f"  Extracted at:    {meta['extraction_date']}")
    print(f"  SHA256 (raw):    {meta['sha256_raw'][:32]}...")
    print(f"  N features:      {meta['n_features']}")
    print(f"  Feature cols:    {meta['feature_cols']}")

    print(f"\n── Validation rule results ─────────────────────────────")
    for rule, passed in meta["validation_results"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {rule}")

    all_passed = all(meta["validation_results"].values())
    print("\n" + "="*60)
    if all_passed:
        print("ALL CHECKS PASSED — ready for Phase 3 (Feature Engineering)")
    else:
        failed = [k for k,v in meta["validation_results"].items() if not v]
        print(f"FAILED RULES: {failed}")
        print("Fix issues before proceeding to Phase 3.")
    print("="*60 + "\n")


if __name__ == "__main__":
    load_and_check()
