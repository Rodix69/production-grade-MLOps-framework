from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType
from datetime import timedelta

# Entity
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    value_type=ValueType.INT64
)

# Data source
churn_source = FileSource(
    path="../../churn_train_v1.parquet",
    timestamp_field="event_timestamp",
)

# Feature View
churn_features = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="num_dependents", dtype=Int64),
        Field(name="estimated_salary", dtype=Float32),
        Field(name="calls_made", dtype=Int64),
        Field(name="sms_sent", dtype=Int64),
        Field(name="data_used", dtype=Float32),
        Field(name="telecom_partner", dtype=String),
        Field(name="gender", dtype=String),
        Field(name="state", dtype=String),
        Field(name="city", dtype=String),
    ],
    online=True,
    source=churn_source,
)