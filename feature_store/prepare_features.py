import pandas as pd

df = pd.read_parquet("data/raw/telecom_churn_raw_v1.parquet")

# Add timestamp (required by Feast)
df["event_timestamp"] = pd.to_datetime(df["date_of_registration"])

# Save parquet
df.to_parquet("feast_demo_source.parquet", index=False)

print("Parquet created")