from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64, String

# ── Entity ────────────────────────────────────────────────────────────────────
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Unique customer identifier",
)

# ── Data Source ───────────────────────────────────────────────────────────────
train_source = FileSource(
    name="churn_train_source",
    path="data/churn_train_v1.parquet",
    timestamp_field="event_timestamp",
)

# ── Feature Views ─────────────────────────────────────────────────────────────

# 1. Raw customer demographics
customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="telecom_partner",  dtype=String),
        Field(name="gender",           dtype=String),
        Field(name="age",              dtype=Int64),
        Field(name="state",            dtype=String),
        Field(name="city",             dtype=String),
        Field(name="pincode",          dtype=Int64),
        Field(name="num_dependents",   dtype=Int64),
        Field(name="estimated_salary", dtype=Float64),
    ],
    source=train_source,
    tags={"phase": "3", "type": "demographics"},
)

# 2. Raw usage features
customer_usage = FeatureView(
    name="customer_usage",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="calls_made",  dtype=Int64),
        Field(name="sms_sent",    dtype=Int64),
        Field(name="data_used",   dtype=Float64),
        Field(name="tenure_days", dtype=Int64),
    ],
    source=train_source,
    tags={"phase": "3", "type": "usage"},
)

# 3. Engineered features
customer_engineered = FeatureView(
    name="customer_engineered",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="total_activity",    dtype=Float64),
        Field(name="avg_data_per_day",  dtype=Float64),
        Field(name="avg_calls_per_day", dtype=Float64),
        Field(name="avg_sms_per_day",   dtype=Float64),
        Field(name="engagement_score",  dtype=Float64),
        Field(name="low_activity_flag", dtype=Int64),
        Field(name="high_value_user",   dtype=Int64),
        Field(name="partner_med_calls", dtype=Float64),
        Field(name="partner_med_data",  dtype=Float64),
        Field(name="partner_med_sms",   dtype=Float64),
        Field(name="calls_vs_partner",  dtype=Float64),
        Field(name="data_vs_partner",   dtype=Float64),
        Field(name="is_new_customer",   dtype=Int64),
        Field(name="calls_intensity",   dtype=Float64),
        Field(name="data_intensity",    dtype=Float64),
    ],
    source=train_source,
    tags={"phase": "3", "type": "engineered"},
)