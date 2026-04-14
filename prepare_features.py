import pandas as pd

df = pd.read_csv("telecom_churn.csv")

# Add timestamp (required by Feast)
df["event_timestamp"] = pd.Timestamp.now()

# Save parquet
df.to_parquet("churn_train_v1.parquet", index=False)

print("Parquet created")