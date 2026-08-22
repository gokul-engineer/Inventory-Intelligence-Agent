import pandas as pd

from utils import PROCESSED_DATA

df = pd.read_parquet(
    PROCESSED_DATA / "m5_processed.parquet"
)

missing = df[df["sell_price"].isna()]

print("=" * 70)
print("PRICE INVESTIGATION")
print("=" * 70)

print(f"Rows with Missing Price : {len(missing):,}")

print("\nFirst Date")

print(missing["date"].min())

print("\nLast Date")

print(missing["date"].max())

print("\nStores")

print(missing["store_id"].value_counts())

print("\nCategories")

print(missing["cat_id"].value_counts())