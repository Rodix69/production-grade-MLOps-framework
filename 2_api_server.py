"""
Step 2 — FastAPI REST API over PostgreSQL
==========================================
Exposes your telecom_customers table as paginated REST endpoints.

Usage:
    pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv
    python 2_api_server.py

Endpoints:
    GET /health
    GET /api/v1/customers          — paginated customer list
    GET /api/v1/customers/{id}     — single customer
    GET /api/v1/stats              — dataset-level statistics
    GET /api/v1/customers/export   — full export for ML (large, paginated)
"""

import os, logging
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import create_engine, text
import uvicorn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# ── Config ────────────────────────────────────────────────────
DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)
API_KEY      = os.getenv("API_KEY", "change_this_key")
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", 8000))
DEFAULT_PAGE = int(os.getenv("PAGE_SIZE", 1000))

# ── Database engine ───────────────────────────────────────────
engine = create_engine(
    DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn Data API",
    description="REST API exposing telecom customer data for ML extraction",
    version="1.0.0",
)

# ── API Key auth ──────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — no auth required."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/v1/stats")
def get_stats(api_key: str = Depends(verify_api_key)):
    """
    Dataset-level statistics — useful for the DS to verify
    they received the full dataset after extraction.
    """
    sql = """
        SELECT
            COUNT(*)                                    AS total_customers,
            ROUND(AVG(churn::float)::numeric, 4)        AS churn_rate,
            COUNT(DISTINCT telecom_partner)             AS n_partners,
            COUNT(DISTINCT state)                       AS n_states,
            MIN(date_of_registration)                   AS earliest_registration,
            MAX(date_of_registration)                   AS latest_registration,
            ROUND(AVG(age)::numeric, 1)                 AS avg_age,
            ROUND(AVG(estimated_salary)::numeric, 0)    AS avg_salary,
            ROUND(AVG(calls_made)::numeric, 1)          AS avg_calls_made,
            ROUND(AVG(data_used)::numeric, 1)           AS avg_data_used,
            SUM(churn)                                  AS total_churned
        FROM telecom_customers
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().fetchone()
    return dict(row)


@app.get("/api/v1/customers")
def list_customers(
    # Pagination
    page:     int = Query(1,    ge=1,    description="Page number (1-based)"),
    limit:    int = Query(DEFAULT_PAGE, ge=1, le=5000, description="Records per page"),
    # Filters
    churn:              Optional[int] = Query(None, ge=0, le=1),
    telecom_partner:    Optional[str] = Query(None),
    state:              Optional[str] = Query(None),
    gender:             Optional[str] = Query(None),
    min_age:            Optional[int] = Query(None),
    max_age:            Optional[int] = Query(None),
    # Sorting
    order_by: str = Query("customer_id", description="Column to sort by"),
    desc:     bool = Query(False,        description="Sort descending"),
    api_key:  str  = Depends(verify_api_key),
):
    """
    Paginated list of customers with optional filtering.
    Used by the DS extraction script to pull all data in pages.
    """
    allowed_order_cols = {
        "customer_id", "age", "estimated_salary",
        "calls_made", "data_used", "date_of_registration", "churn"
    }
    if order_by not in allowed_order_cols:
        raise HTTPException(status_code=400,
                            detail=f"order_by must be one of {allowed_order_cols}")

    # Build WHERE clause dynamically
    filters = []
    params  = {"limit": limit, "offset": (page - 1) * limit}

    if churn is not None:
        filters.append("churn = :churn")
        params["churn"] = churn
    if telecom_partner:
        filters.append("telecom_partner ILIKE :partner")
        params["partner"] = f"%{telecom_partner}%"
    if state:
        filters.append("state ILIKE :state")
        params["state"] = f"%{state}%"
    if gender:
        filters.append("gender = :gender")
        params["gender"] = gender.upper()
    if min_age is not None:
        filters.append("age >= :min_age")
        params["min_age"] = min_age
    if max_age is not None:
        filters.append("age <= :max_age")
        params["max_age"] = max_age

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""
    direction    = "DESC" if desc else "ASC"

    count_sql = f"SELECT COUNT(*) FROM telecom_customers {where_clause}"
    data_sql  = f"""
        SELECT
            customer_id, telecom_partner, gender, age,
            state, city, pincode,
            TO_CHAR(date_of_registration, 'YYYY-MM-DD') AS date_of_registration,
            num_dependents, estimated_salary,
            calls_made, sms_sent, data_used, churn
        FROM telecom_customers
        {where_clause}
        ORDER BY {order_by} {direction}
        LIMIT :limit OFFSET :offset
    """

    with engine.connect() as conn:
        total    = conn.execute(text(count_sql), params).scalar()
        rows     = conn.execute(text(data_sql),  params).mappings().fetchall()

    total_pages = (total + limit - 1) // limit

    return {
        "data":        [dict(r) for r in rows],
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "has_next":    page < total_pages,
    }


@app.get("/api/v1/customers/{customer_id}")
def get_customer(
    customer_id: int,
    api_key: str = Depends(verify_api_key),
):
    """Single customer lookup by ID."""
    sql = """
        SELECT
            customer_id, telecom_partner, gender, age,
            state, city, pincode,
            TO_CHAR(date_of_registration, 'YYYY-MM-DD') AS date_of_registration,
            num_dependents, estimated_salary,
            calls_made, sms_sent, data_used, churn
        FROM telecom_customers
        WHERE customer_id = :cid
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"cid": customer_id}).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"Customer {customer_id} not found")
    return dict(row)


@app.get("/api/v1/customers/export/all")
def export_all(
    page:  int = Query(1,    ge=1),
    limit: int = Query(DEFAULT_PAGE, ge=1, le=5000),
    order_by: str  = Query("customer_id"),
    api_key:  str  = Depends(verify_api_key),
):
    """
    Full dataset export — no filters, optimised for ML extraction.
    The DS extraction script calls this endpoint in a pagination loop.
    """
    allowed = {"customer_id", "date_of_registration", "age"}
    if order_by not in allowed:
        order_by = "customer_id"

    sql = f"""
        SELECT
            customer_id, telecom_partner, gender, age,
            state, city, pincode,
            TO_CHAR(date_of_registration, 'YYYY-MM-DD') AS date_of_registration,
            num_dependents, estimated_salary,
            calls_made, sms_sent, data_used, churn
        FROM telecom_customers
        ORDER BY {order_by} ASC
        LIMIT :limit OFFSET :offset
    """
    params = {"limit": limit, "offset": (page - 1) * limit}

    count_sql = "SELECT COUNT(*) FROM telecom_customers"

    with engine.connect() as conn:
        total = conn.execute(text(count_sql)).scalar()
        rows  = conn.execute(text(sql), params).mappings().fetchall()

    total_pages = (total + limit - 1) // limit

    return {
        "data":        [dict(r) for r in rows],
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "has_next":    page < total_pages,
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Starting API on {API_HOST}:{API_PORT}")
    uvicorn.run("2_api_server:app", host=API_HOST, port=API_PORT, reload=False)
