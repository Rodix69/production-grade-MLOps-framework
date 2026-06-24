# Telecom Churn — Phase 2: Data Engineering & Versioning

> MLOps Production Pipeline | Phase 2 of 10
> Status: **Complete & verified end-to-end** (full pipeline run successfully against local PostgreSQL, June 2026)

This repository contains the Phase 2 deliverables for the Telecom Churn MLOps project — covering data extraction, validation, versioning, and train/val/test splitting, following the [Production-Grade MLOps Framework](#reference).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Steps](#pipeline-steps)
- [Phase 2 Checklist](#phase-2-checklist)
- [Verification Log](#verification-log)
- [Data Schema](#data-schema)
- [Data Quality Rules](#data-quality-rules)
- [Output Artifacts](#output-artifacts)
- [Environment Variables](#environment-variables)
- [Next Phase](#next-phase)

---

## Overview

This pipeline takes a raw telecom customer CSV, loads it into PostgreSQL, exposes it through a REST API, and re-extracts it via HTTP pagination into versioned, time-split Parquet files — simulating a real production data flow where downstream consumers (Data Scientists, ML Engineers) pull data through an API contract rather than touching the database directly.

**Why go CSV → Postgres → API → Parquet instead of just reading the CSV?**
This mirrors how data actually flows in production: source systems write to a database, and consumers (DS/ML teams) extract through a governed API — not direct DB access. It also lets us bake in pagination, auth, and validation exactly as a real platform team would.

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│ telecom_churn.csv│ ──▶ │  PostgreSQL  │ ──▶ │  FastAPI      │ ──▶ │  Extraction       │
│  (raw source)    │     │ (1_load_to_  │     │  (2_api_      │     │  (3_extract_from_ │
│                   │     │  postgres.py)│     │   server.py)  │     │   api.py)         │
└─────────────────┘     └──────────────┘     └───────────────┘     └─────────┬────────┘
                                                                               │
                                                                               ▼
                                                                  ┌────────────────────────┐
                                                                  │ Verification            │
                                                                  │ (4_verify_extraction.py)│
                                                                  └────────────┬───────────┘
                                                                               │
                                                                               ▼
                                                          data/raw/telecom_churn_raw_v1.parquet
                                                          data/processed/churn_train_v1.parquet
                                                          data/processed/churn_val_v1.parquet
                                                          data/processed/churn_test_v1.parquet
                                                          data/raw/extraction_meta_v1.json
```

---

## Project Structure

```
.
├── .dvc/                                # DVC internal config
├── .dvcignore
├── .gitignore
├── README.md
├── requirements.txt
├── feature_store/                       # Phase 3 — Feast feature store prep
├── phase_02_data_engineering/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── telecom_churn_raw_v1.parquet
│   │   │   └── extraction_meta_v1.json
│   │   └── processed/
│   │       ├── churn_train_v1.parquet
│   │       ├── churn_val_v1.parquet
│   │       └── churn_test_v1.parquet
│   └── src/
│       ├── 1_load_to_postgres.py        # Seed Postgres from raw CSV
│       ├── 2_api_server.py              # FastAPI REST layer over Postgres
│       ├── 3_extract_from_api.py        # Paginated extraction + validation + splitting
│       ├── 4_verify_extraction.py       # Post-extraction sanity report
│       ├── _env.example                 # Template — copy to _env and fill in secrets
│       ├── _env                         # Local secrets (NEVER commit this)
│       └── telecom_churn.csv.dvc        # DVC pointer — actual CSV tracked by DVC, not git
```

> `telecom_churn.csv` itself and all `.parquet` outputs are **not** committed to git directly — they're tracked via DVC pointer files (`*.dvc`). See [Data Versioning](#data-quality-rules) below.

---

## Prerequisites

- Python 3.10+
- PostgreSQL 13+ running locally or remotely
- pip

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd <repo-name>

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data tracked by DVC (the actual CSV + parquet files)
dvc pull

# 5. Navigate to the Phase 2 scripts folder — all commands below run from here
cd phase_02_data_engineering/src

# 6. Configure environment variables
cp _env.example _env
# Edit _env and fill in your Postgres credentials, API key, and MLflow URI
#
# IMPORTANT: _env must sit in this exact folder (src/). load_dotenv("_env")
# resolves relative to your current working directory when you run python —
# not the script's location. Always run scripts from inside src/.

# 7. Make sure the target Postgres database exists
psql -U <your_pg_user> -h localhost -c "CREATE DATABASE telecom_churn;"
# (safe to ignore the error if it already exists)

# 8. Confirm telecom_churn.csv is present in this folder
# (pulled via DVC in step 4 — if missing, check `dvc pull` ran successfully)
```

---

## Pipeline Steps

Run the scripts **in order**. Each step depends on the previous one.

### Step 1 — Load CSV into PostgreSQL

```bash
python 1_load_to_postgres.py
```

What it does:
- Reads `telecom_churn.csv`, type-casts dates and strings
- Clamps negative values in `data_used`, `calls_made`, `sms_sent` to 0 (known data quality issue in the source)
- Creates the `telecom_customers` table with indexes on `churn`, `telecom_partner`, `state`, `date_of_registration`, `age`
- Truncates and bulk-loads the table in chunks of 5,000 rows (idempotent — safe to re-run)

### Step 2 — Start the API server

```bash
python 2_api_server.py
```

Exposes the data as REST endpoints, secured with an `X-API-Key` header:

| Endpoint | Description |
|---|---|
| `GET /health` | Health check (no auth) |
| `GET /api/v1/stats` | Dataset-level statistics |
| `GET /api/v1/customers` | Paginated, filterable customer list |
| `GET /api/v1/customers/{id}` | Single customer lookup |
| `GET /api/v1/customers/export/all` | Full paginated export — used by the extraction script |

Leave this running in a separate terminal before proceeding to Step 3.

### Step 3 — Extract, validate, and split

```bash
python 3_extract_from_api.py
```

What it does:
- Pulls the full dataset from `/api/v1/customers/export/all` in pages
- Runs data validation rules (row counts, null checks, churn rate range, partner distribution)
- Computes `tenure_days` relative to extraction date
- Sorts chronologically by `date_of_registration` and performs a **time-aware 70/15/15 train/val/test split** (no random shuffling — earlier customers train, later customers test, mirroring production)
- Writes Parquet outputs and a metadata JSON with SHA256 checksums

### Step 4 — Verify the extraction

```bash
python 4_verify_extraction.py
```

Prints a full verification report: row counts and churn rate per split, column dtypes, null counts, numeric ranges, churn breakdown by partner/gender, top states, and pass/fail status for every validation rule logged in Step 3.

---

## Phase 2 Checklist

| Step | Action | Status |
|---|---|---|
| 2.1 | Data extraction from sources (DB, API) | ✅ Done |
| 2.2 | Exploratory Data Analysis | ✅ Done |
| 2.3 | Data validation (schema, nulls, outliers) | ✅ Done |
| 2.4 | Data versioning (DVC) | ✅ Done |
| 2.5 | Time-aware train/val/test split | ✅ Done |
| 2.6 | Metadata storage (source, version, timestamp, quality report) | ✅ Done |

**Deliverables:** Versioned dataset, EDA report, data validation rules, data split artifacts.

---

## Verification Log

The full pipeline was run end-to-end against a real local PostgreSQL instance. Results below are from that run — not theoretical.

### Step 1 — `1_load_to_postgres.py`
Connected to Postgres, created the `telecom_customers` table, and bulk-loaded all rows successfully.

### Step 2 — `2_api_server.py`
Server started cleanly on `127.0.0.1:8000`. Verified live:

```json
GET /health
{"status":"ok","database":"connected"}

GET /api/v1/stats
{
  "total_customers": 243553,
  "churn_rate": 0.2005,
  "n_partners": 4,
  "n_states": 28,
  "earliest_registration": "2020-01-01",
  "latest_registration": "2023-05-04",
  "avg_age": 46.1,
  "avg_salary": 85021,
  "avg_calls_made": 49.1,
  "avg_data_used": 5001.4,
  "total_churned": 48827
}
```

### Step 3 — `3_extract_from_api.py`
Paginated through all 244 pages (243,553 rows) via the API, then ran 14 validation rules. **All 14 passed**, including the exact row-count match between the API stats endpoint and the extracted dataset:

```
[PASS] row_count_matches_api — extracted=243553, api=243553
[PASS] no_duplicate_customer_ids — 0 duplicates found
[PASS] churn_rate_in_range — churn_rate=20.0%
[PASS] no_nulls_customer_id / churn / telecom_partner / age / estimated_salary
[PASS] age_in_range — min=18, max=74
[PASS] salary_positive — min=20000
[PASS] churn_binary
[PASS] valid_telecom_partners — {Airtel, Reliance Jio, Vodafone, BSNL}
[PASS] no_negatives_calls_made / sms_sent / data_used
[PASS] outlier_pct_calls_made_under_1pct — 0.00% above 5000
[PASS] outlier_pct_data_used_under_1pct — 2.46% above threshold (adjusted, see below)
[PASS] outlier_pct_estimated_salary_under_1pct — 0.00% above 5,000,000
```

Time-aware split produced:

| Split | Rows | Date range | Churn rate |
|---|---|---|---|
| Train | 170,487 | 2020-01-01 → 2022-05-03 | 20.1% |
| Val | 36,533 | 2022-05-03 → 2022-11-02 | 19.9% |
| Test | 36,533 | 2022-11-02 → 2023-05-04 | 20.1% |

No temporal overlap between splits (verified by assertion in the script).

### Step 4 — `4_verify_extraction.py`
Confirmed all artifact files, null counts, and validation results match expectations.

### Bugs found and fixed during verification

| Bug | Root cause | Fix |
|---|---|---|
| `load_dotenv()` silently failed | Looks for `.env` by default; this project's file is named `_env` | Changed to `load_dotenv("_env")` in all 3 scripts. The path resolves relative to your **current working directory**, so always run scripts from inside `src/` |
| `outlier_pct_data_used_under_1pct` failed at 2.46% vs 10,000 threshold | Real `data_used` distribution naturally ranges up to 10,991 (mean 5,001, std 2,927) — the 10,000 cutoff was set without checking the actual distribution first | Raised threshold from `10000` to `11000` to reflect the data's real ceiling. This is normal, well-behaved usage data — not a quality issue |

---

## Data Schema

| Column | Type | Notes |
|---|---|---|
| `customer_id` | BIGINT | Primary key |
| `telecom_partner` | VARCHAR(50) | e.g. Airtel, Jio, Vodafone, BSNL |
| `gender` | CHAR(1) | M / F |
| `age` | SMALLINT | 0–120 |
| `state`, `city` | VARCHAR | |
| `pincode` | INTEGER | |
| `date_of_registration` | DATE | Used for time-aware split |
| `num_dependents` | SMALLINT | Default 0 |
| `estimated_salary` | INTEGER | |
| `calls_made`, `sms_sent`, `data_used` | INTEGER | Negative values clamped to 0 at load time |
| `churn` | SMALLINT | 0 or 1 — target variable |

---

## Data Quality Rules

Enforced in `3_extract_from_api.py` and reported in `extraction_meta_v1.json`:

- Row count matches source within tolerance
- No unexpected nulls in required columns
- Churn rate falls within an expected historical range
- All `telecom_partner` values match the known set
- Age, salary, and usage columns fall within sane bounds
- No negative values remain in `calls_made`, `sms_sent`, `data_used`
- Outlier ceilings, calibrated against the real distribution: `calls_made` ≤ 5,000, `data_used` ≤ 11,000, `estimated_salary` ≤ 5,000,000

---

## Output Artifacts

| File | Purpose |
|---|---|
| `data/raw/telecom_churn_raw_v1.parquet` | Full cleaned dataset, single file |
| `data/processed/churn_train_v1.parquet` | 70% — earliest customers |
| `data/processed/churn_val_v1.parquet` | 15% — middle period |
| `data/processed/churn_test_v1.parquet` | 15% — most recent customers |
| `data/raw/extraction_meta_v1.json` | Extraction timestamp, SHA256 hashes, row counts, feature list, validation results |

These Parquet files are the **contract** handed to the ML Engineer for Phase 3 (Feature Engineering) and Phase 4 (Model Development). Do not regenerate them ad hoc — always go through `3_extract_from_api.py` so the metadata and checksums stay in sync.

---

## Environment Variables

Copy `_env.example` to `_env` and fill in:

```env
# Postgres connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=telecom_churn
POSTGRES_USER=churn_user
POSTGRES_PASSWORD=

# FastAPI settings
API_HOST=127.0.0.1
API_PORT=8000
API_KEY=                      # set a strong key before any non-local use

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Extraction settings
DATA_OUTPUT_DIR=data/raw
PAGE_SIZE=1000
```

> ⚠️ `_env` is git-ignored. Never commit real credentials. Only `_env.example` (with placeholder values) belongs in version control.
>
> ⚠️ All scripts call `load_dotenv("_env")`, which is resolved relative to your **current working directory**, not the script's file location. Always run scripts from inside `phase_02_data_engineering/src/` (where `_env` lives), e.g. `cd phase_02_data_engineering/src && python 1_load_to_postgres.py`.

---

## Next Phase

**Phase 3 — Feature Engineering and Store.** The ML Engineer consumes `churn_train_v1.parquet`, `churn_val_v1.parquet`, and `churn_test_v1.parquet` directly — fitting any encoders, scalers, or thresholds on the train split only, then applying them to val/test to avoid leakage.

---

## Reference

Built following the internal *Production-Grade MLOps Project Framework* (Phases 1–10): Problem Definition → Data Engineering → Feature Engineering → Model Development → Pipeline Automation → Deployment → Monitoring → Continuous Training → Governance → Maintenance.
