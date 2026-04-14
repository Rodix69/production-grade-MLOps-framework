"""
Step 3 — Data Scientist Extraction Script
==========================================
Calls the FastAPI REST endpoints, paginates through all records,
cleans, validates, splits, and saves versioned Parquet files
ready for Phase 3 (Feature Engineering).

Usage:
    pip install requests pandas pyarrow python-dotenv tqdm mlflow great-expectations
    python 3_extract_from_api.py

Output:
    data/raw/telecom_churn_raw_v1.parquet     ← full merged dataset
    data/processed/churn_train_v1.parquet     ← 70% time-ordered train
    data/processed/churn_val_v1.parquet       ← 15% validation
    data/processed/churn_test_v1.parquet      ← 15% test (locked)
    data/raw/extraction_meta_v1.json          ← provenance log
"""

import os, json, logging, hashlib, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────
BASE_URL   = f"http://{os.getenv('API_HOST','localhost')}:{os.getenv('API_PORT',8000)}/api/v1"
API_KEY    = os.getenv("API_KEY")
PAGE_SIZE  = int(os.getenv("PAGE_SIZE", 1000))
OUTPUT_DIR = Path(os.getenv("DATA_OUTPUT_DIR", "data/raw"))
PROC_DIR   = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

assert API_KEY, "API_KEY not set in .env"


# ─────────────────────────────────────────────────────────────
# 1. Build robust HTTP session
# ─────────────────────────────────────────────────────────────

def build_session() -> requests.Session:
    """Session with retry, backoff, and auth header baked in."""
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=2,                            # 2, 4, 8, 16, 32 sec waits
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)

    session.headers.update({
        "X-API-Key":    API_KEY,
        "Accept":       "application/json",
        "Content-Type": "application/json",
    })
    return session


def api_get(session: requests.Session, endpoint: str,
            params: dict = None, timeout: int = 60) -> dict:
    """Single GET call with rate-limit awareness and error handling."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = session.get(url, params=params, timeout=timeout)

        # Honour rate limits if server sends them
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 9999))
        if remaining < 5:
            wait = int(resp.headers.get("X-RateLimit-Reset", 30))
            log.warning(f"Rate limit low ({remaining} left). Sleeping {wait}s...")
            time.sleep(wait)

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        log.error(f"Timeout on {url}")
        raise
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        raise


# ─────────────────────────────────────────────────────────────
# 2. Health check
# ─────────────────────────────────────────────────────────────

def check_health(session: requests.Session):
    """Verify API is up before starting a long extraction."""
    url = BASE_URL.replace("/api/v1", "/health")
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    log.info(f"API health: {data}")
    if data.get("status") != "ok":
        raise RuntimeError(f"API not healthy: {data}")


# ─────────────────────────────────────────────────────────────
# 3. Paginated full extraction
# ─────────────────────────────────────────────────────────────

def extract_all_pages(session: requests.Session) -> list[dict]:
    """
    Call /customers/export/all in a page loop until has_next = False.
    Returns list of raw record dicts.
    """
    all_records = []
    page = 1

    # First call to get total
    first = api_get(session, "customers/export/all",
                    params={"page": 1, "limit": PAGE_SIZE})
    total_pages = first["total_pages"]
    total_rows  = first["total"]
    log.info(f"Total records: {total_rows:,} across {total_pages} pages "
             f"(page_size={PAGE_SIZE})")

    all_records.extend(first["data"])

    with tqdm(total=total_rows, initial=len(first["data"]),
              desc="Extracting via API", unit="rows") as pbar:
        while page < total_pages:
            page += 1
            resp = api_get(session, "customers/export/all",
                           params={"page": page, "limit": PAGE_SIZE})
            batch = resp["data"]
            all_records.extend(batch)
            pbar.update(len(batch))

            if not resp.get("has_next", False):
                break

    log.info(f"Extraction complete: {len(all_records):,} records")
    return all_records


# ─────────────────────────────────────────────────────────────
# 4. API stats check
# ─────────────────────────────────────────────────────────────

def get_api_stats(session: requests.Session) -> dict:
    """Pull server-side stats for post-extraction validation."""
    return api_get(session, "stats")


# ─────────────────────────────────────────────────────────────
# 5. Clean and type-cast
# ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all type casts and data quality fixes.
    Mirrors exactly what 1_load_to_postgres.py did —
    so training and serving feature logic stay in sync.
    """
    log.info("Cleaning and type-casting...")

    # Dates
    df["date_of_registration"] = pd.to_datetime(
        df["date_of_registration"], errors="coerce"
    )

    # Numerics (API returns them correctly typed but recast for safety)
    int_cols = ["customer_id", "age", "pincode", "num_dependents",
                "estimated_salary", "calls_made", "sms_sent", "data_used", "churn"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Strings — strip whitespace
    str_cols = ["telecom_partner", "gender", "state", "city"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Clamp negatives (defensive — should already be clean from DB)
    for col in ["calls_made", "sms_sent", "data_used"]:
        neg = (df[col] < 0).sum()
        if neg > 0:
            log.warning(f"{col}: {neg} negatives clamped to 0")
        df[col] = df[col].clip(lower=0)

    # Derived feature: tenure in days from registration to today
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    df["tenure_days"] = (today - df["date_of_registration"]).dt.days
    df["tenure_days"] = df["tenure_days"].clip(lower=0)

    log.info(f"Clean shape: {df.shape}")
    log.info(f"Churn rate:  {df['churn'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────
# 6. Validate
# ─────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, api_stats: dict) -> dict:
    """
    Run assertion-based validation rules.
    Returns a dict of results — all must pass before saving.
    """
    log.info("Running validation rules...")
    results = {}

    def check(name: str, passed: bool, detail: str = ""):
        results[name] = {"passed": passed, "detail": detail}
        status = "PASS" if passed else "FAIL"
        log.info(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # Row count matches API stats
    api_total = int(api_stats["total_customers"])
    check("row_count_matches_api",
          len(df) == api_total,
          f"extracted={len(df)}, api={api_total}")

    # No duplicate customer IDs
    dupes = df["customer_id"].duplicated().sum()
    check("no_duplicate_customer_ids",
          dupes == 0, f"{dupes} duplicates found")

    # Churn rate in expected range (15–25% from our EDA)
    churn_rate = df["churn"].mean()
    check("churn_rate_in_range",
          0.10 <= churn_rate <= 0.30,
          f"churn_rate={churn_rate:.1%}")

    # No nulls in critical columns
    critical = ["customer_id", "churn", "telecom_partner",
                "age", "estimated_salary"]
    for col in critical:
        null_count = df[col].isnull().sum()
        check(f"no_nulls_{col}",
              null_count == 0, f"{null_count} nulls")

    # Age range
    check("age_in_range",
          df["age"].between(0, 120).all(),
          f"min={df['age'].min()}, max={df['age'].max()}")

    # Salary positive
    check("salary_positive",
          (df["estimated_salary"] > 0).all(),
          f"min={df['estimated_salary'].min()}")

    # Churn binary
    check("churn_binary",
          df["churn"].isin([0, 1]).all())

    # Known partners
    expected_partners = {"Reliance Jio", "Vodafone", "BSNL", "Airtel"}
    actual_partners   = set(df["telecom_partner"].unique())
    check("valid_telecom_partners",
          actual_partners.issubset(expected_partners),
          f"found={actual_partners}")

    # No negatives in usage cols (after clamping)
    for col in ["calls_made", "sms_sent", "data_used"]:
        check(f"no_negatives_{col}",
              (df[col] >= 0).all())
        
    # Outlier bounds (from EDA)
    for col, upper in [("calls_made", 5000), ("data_used", 10000), ("estimated_salary", 5000000)]:
        pct_extreme = (df[col] > upper).mean()
        check(f"outlier_pct_{col}_under_1pct", pct_extreme < 0.01,
            f"{pct_extreme:.2%} above {upper}")

    failed = [k for k, v in results.items() if not v["passed"]]
    if failed:
        log.error(f"VALIDATION FAILED: {failed}")
        raise ValueError(f"Validation failed for: {failed}")

    log.info(f"All {len(results)} validation rules passed.")
    return results


# ─────────────────────────────────────────────────────────────
# 7. Time-aware train / val / test split
# ─────────────────────────────────────────────────────────────

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by date_of_registration (time-ordered).
    70% train / 15% val / 15% test — no shuffle.
    """
    log.info("Splitting by registration date (time-aware)...")
    df_sorted = df.sort_values("date_of_registration").reset_index(drop=True)

    n         = len(df_sorted)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df_sorted.iloc[:train_end].copy()
    val   = df_sorted.iloc[train_end:val_end].copy()
    test  = df_sorted.iloc[val_end:].copy()

    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        log.info(
            f"  {name}: {len(split_df):>7,} rows | "
            f"date {split_df['date_of_registration'].min().date()} → "
            f"{split_df['date_of_registration'].max().date()} | "
            f"churn {split_df['churn'].mean():.1%}"
        )

    # Sanity: no temporal overlap
    assert train["date_of_registration"].max() <= val["date_of_registration"].min(), \
        "Temporal overlap between train and val!"
    assert val["date_of_registration"].max() <= test["date_of_registration"].min(), \
        "Temporal overlap between val and test!"

    return train, val, test


# ─────────────────────────────────────────────────────────────
# 8. Save and fingerprint
# ─────────────────────────────────────────────────────────────

def sha256_of(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values
    ).hexdigest()


def save_all(df: pd.DataFrame,
             train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
             validation_results: dict, api_stats: dict) -> dict:
    """Save all Parquet files and write extraction metadata JSON."""
    raw_path   = OUTPUT_DIR / "telecom_churn_raw_v1.parquet"
    train_path = PROC_DIR   / "churn_train_v1.parquet"
    val_path   = PROC_DIR   / "churn_val_v1.parquet"
    test_path  = PROC_DIR   / "churn_test_v1.parquet"

    df.to_parquet(raw_path,   index=False)
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path,     index=False)
    test.to_parquet(test_path,   index=False)
    log.info(f"Saved: {raw_path}, {train_path}, {val_path}, {test_path}")

    feature_cols = [c for c in df.columns
                    if c not in ["customer_id", "churn", "date_of_registration"]]

    meta = {
        "version":              "v1.0",
        "extraction_date":      datetime.now(timezone.utc).isoformat(),
        "api_base_url":         BASE_URL,
        "page_size_used":       PAGE_SIZE,
        "total_rows":           len(df),
        "train_rows":           len(train),
        "val_rows":             len(val),
        "test_rows":            len(test),
        "n_features":           len(feature_cols),
        "feature_cols":         feature_cols,
        "churn_rate_overall":   round(float(df["churn"].mean()), 4),
        "churn_rate_train":     round(float(train["churn"].mean()), 4),
        "churn_rate_val":       round(float(val["churn"].mean()), 4),
        "churn_rate_test":      round(float(test["churn"].mean()), 4),
        "telecom_partners":     sorted(df["telecom_partner"].unique().tolist()),
        "states_count":         df["state"].nunique(),
        "sha256_raw":           sha256_of(df),
        "sha256_train":         sha256_of(train),
        "sha256_val":           sha256_of(val),
        "sha256_test":          sha256_of(test),
        "api_stats":            api_stats,
        "validation_results":   {k: v["passed"] for k, v in validation_results.items()},
        "output_paths": {
            "raw":   str(raw_path),
            "train": str(train_path),
            "val":   str(val_path),
            "test":  str(test_path),
        }
    }

    meta_path = OUTPUT_DIR / "extraction_meta_v1.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Metadata saved: {meta_path}")

    return meta


# ─────────────────────────────────────────────────────────────
# 9. Log to MLflow
# ─────────────────────────────────────────────────────────────

def log_to_mlflow(meta: dict):
    try:
        import mlflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("telecom-churn-phase2")

        with mlflow.start_run(run_name="api_extraction_v1"):
            # Params
            mlflow.log_param("data_version",    meta["version"])
            mlflow.log_param("api_base_url",    meta["api_base_url"])
            mlflow.log_param("page_size",       meta["page_size_used"])
            mlflow.log_param("extraction_date", meta["extraction_date"])
            mlflow.log_param("n_features",      meta["n_features"])
            mlflow.log_param("sha256_raw",      meta["sha256_raw"][:16])

            # Metrics
            mlflow.log_metric("total_rows",         meta["total_rows"])
            mlflow.log_metric("train_rows",         meta["train_rows"])
            mlflow.log_metric("val_rows",           meta["val_rows"])
            mlflow.log_metric("test_rows",          meta["test_rows"])
            mlflow.log_metric("churn_rate_overall", meta["churn_rate_overall"])
            mlflow.log_metric("churn_rate_train",   meta["churn_rate_train"])
            mlflow.log_metric("churn_rate_val",     meta["churn_rate_val"])
            mlflow.log_metric("churn_rate_test",    meta["churn_rate_test"])
            mlflow.log_metric("n_states",           meta["states_count"])
            mlflow.log_metric("n_partners",
                              len(meta["telecom_partners"]))

            # Artifacts
            mlflow.log_artifact(
                str(OUTPUT_DIR / "extraction_meta_v1.json"))
            mlflow.log_artifact(
                str(OUTPUT_DIR / "telecom_churn_raw_v1.parquet"))

            # Tags
            mlflow.set_tag("phase",   "2.1_extraction")
            mlflow.set_tag("source",  "REST API → PostgreSQL")
            mlflow.set_tag("status",  "validated")
            mlflow.set_tag("engineer", os.getenv("USER", "ds"))

        log.info("MLflow run logged successfully.")
    except Exception as e:
        log.warning(f"MLflow logging skipped (not blocking): {e}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Phase 2.1 — Telecom Churn API Extraction")
    log.info("=" * 60)

    session = build_session()

    # 1. Health check
    check_health(session)

    # 2. Get server-side stats (for post-validation)
    api_stats = get_api_stats(session)
    log.info(f"API stats: {json.dumps(api_stats, indent=2, default=str)}")

    # 3. Extract all records via paginated API
    records = extract_all_pages(session)

    # 4. Build DataFrame
    df = pd.DataFrame(records)
    log.info(f"Raw DataFrame shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")

    # 5. Clean and type-cast
    df = clean(df)

    # 6. Validate
    validation_results = validate(df, api_stats)

    # 7. Split
    train, val, test = split(df)

    # 8. Save
    meta = save_all(df, train, val, test, validation_results, api_stats)
    meta["dvc_tracked"] = True
    meta["data_version_tag"] = "phase2-data-v1"
    meta["data_quality"] = {
    "pct_nulls": float(df.isnull().mean().mean()),
    "pct_negative_clamped": {
        col: float((df[col] == 0).mean()) for col in ["calls_made","sms_sent","data_used"]
    },
    "outlier_pct_calls": float((df["calls_made"] > 5000).mean()),
    "outlier_pct_data":  float((df["data_used"]  > 10000).mean()),
}
    
    # 9. Log to MLflow
    log_to_mlflow(meta)

    log.info("=" * 60)
    log.info("Phase 2.1 COMPLETE")
    log.info(f"  Raw dataset:  {meta['output_paths']['raw']}")
    log.info(f"  Train:        {meta['output_paths']['train']} ({meta['train_rows']:,} rows)")
    log.info(f"  Val:          {meta['output_paths']['val']}   ({meta['val_rows']:,} rows)")
    log.info(f"  Test:         {meta['output_paths']['test']}  ({meta['test_rows']:,} rows)")
    log.info(f"  Churn rate:   {meta['churn_rate_overall']:.1%}")
    log.info(f"  SHA256:       {meta['sha256_raw'][:24]}...")
    log.info("Next step: run 4_verify_extraction.py, then move to Phase 3.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
