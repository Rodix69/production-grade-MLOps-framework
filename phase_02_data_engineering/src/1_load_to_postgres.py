"""
Step 1 — Load telecom_churn.csv into PostgreSQL
================================================
Run this ONCE to seed your database from the raw CSV.

Usage:
    pip install pandas sqlalchemy psycopg2-binary python-dotenv tqdm
    python 1_load_to_postgres.py
"""

import os, logging
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# ── Connection ────────────────────────────────────────────────
DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)

CSV_PATH = r"C:\Mlops framework phase 1\telecom_churn.csv"   # adjust path if needed


def load_csv() -> pd.DataFrame:
    """Read and type-cast the raw CSV."""
    log.info(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    log.info(f"Raw shape: {df.shape}")

    # ── Type casting ─────────────────────────────────────────
    df["date_of_registration"] = pd.to_datetime(
        df["date_of_registration"], errors="coerce"
    )
    df["gender"]           = df["gender"].astype(str).str.strip().str.upper()
    df["telecom_partner"]  = df["telecom_partner"].astype(str).str.strip()
    df["state"]            = df["state"].astype(str).str.strip()
    df["city"]             = df["city"].astype(str).str.strip()

    # ── Clamp negative values (data quality issue in source) ──
    # data_used, calls_made, sms_sent have negatives — floor at 0
    for col in ["data_used", "calls_made", "sms_sent"]:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            log.warning(f"{col}: {neg_count} negative values → clamped to 0")
        df[col] = df[col].clip(lower=0)

    log.info(f"Churn rate: {df['churn'].mean():.1%}")
    log.info(f"Telecom partners: {df['telecom_partner'].unique().tolist()}")
    return df


def create_schema(engine):
    """Create the telecom_customers table if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS telecom_customers (
        customer_id          BIGINT PRIMARY KEY,
        telecom_partner      VARCHAR(50)  NOT NULL,
        gender               CHAR(1)      NOT NULL,
        age                  SMALLINT     NOT NULL CHECK (age BETWEEN 0 AND 120),
        state                VARCHAR(100) NOT NULL,
        city                 VARCHAR(100) NOT NULL,
        pincode              INTEGER      NOT NULL,
        date_of_registration DATE,
        num_dependents       SMALLINT     NOT NULL DEFAULT 0,
        estimated_salary     INTEGER      NOT NULL,
        calls_made           INTEGER      NOT NULL DEFAULT 0,
        sms_sent             INTEGER      NOT NULL DEFAULT 0,
        data_used            INTEGER      NOT NULL DEFAULT 0,
        churn                SMALLINT     NOT NULL CHECK (churn IN (0,1)),
        created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    );

    -- Indexes for common query patterns
    CREATE INDEX IF NOT EXISTS idx_churn        ON telecom_customers(churn);
    CREATE INDEX IF NOT EXISTS idx_partner      ON telecom_customers(telecom_partner);
    CREATE INDEX IF NOT EXISTS idx_state        ON telecom_customers(state);
    CREATE INDEX IF NOT EXISTS idx_reg_date     ON telecom_customers(date_of_registration);
    CREATE INDEX IF NOT EXISTS idx_age          ON telecom_customers(age);
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    log.info("Schema created / verified.")


def load_to_postgres(df: pd.DataFrame, engine):
    """
    Bulk-load DataFrame into Postgres in chunks.
    Uses if_exists='append' so re-runs are safe after truncation.
    """
    # Truncate first to make this idempotent
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE telecom_customers RESTART IDENTITY"))
        conn.commit()
    log.info("Table truncated — loading fresh data...")

    chunk_size = 5000
    n_chunks   = (len(df) // chunk_size) + 1

    for i, chunk_start in enumerate(
        tqdm(range(0, len(df), chunk_size),
             total=n_chunks, desc="Loading chunks")
    ):
        chunk = df.iloc[chunk_start : chunk_start + chunk_size]
        chunk.to_sql(
            name="telecom_customers",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",         # faster bulk insert
        )

    # Verify
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*), AVG(churn::float) FROM telecom_customers")
        ).fetchone()
    log.info(f"Loaded {result[0]:,} rows | churn rate: {result[1]:.1%}")


def main():
    log.info("Connecting to Postgres...")
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Connection OK.")

    df = load_csv()
    create_schema(engine)
    load_to_postgres(df, engine)

    log.info("Done. Run 2_api_server.py next.")


if __name__ == "__main__":
    main()
