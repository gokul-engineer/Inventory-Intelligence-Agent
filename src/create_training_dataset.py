import pandas as pd

from utils import PROCESSED_DATA

print("=" * 70)
print("CREATING TRAINING DATASET")
print("=" * 70)

df = pd.read_parquet(
    PROCESSED_DATA / "m5_processed.parquet"
)

print(f"Original Rows : {len(df):,}")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Remove missing prices
df = df.dropna(subset=["sell_price"])

print(f"Rows after removing missing prices : {len(df):,}")

# Sort
df = df.sort_values(
    ["item_id", "store_id", "date"]
).reset_index(drop=True)

print("Saving...")

df.to_parquet(
    PROCESSED_DATA / "training_dataset.parquet",
    index=False
)

print("Done!")

print(df.head())