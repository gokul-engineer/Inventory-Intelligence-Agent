import pandas as pd

from utils import PROCESSED_DATA

df = pd.read_parquet(
    PROCESSED_DATA / "m5_processed.parquet"
)

print("="*70)
print("DATA VALIDATION")
print("="*70)

print(f"Rows                 : {len(df):,}")
print(f"Columns              : {len(df.columns)}")

print()

print(f"Missing Prices       : {df['sell_price'].isna().sum():,}")

print(f"Missing Sales        : {df['sales'].isna().sum():,}")

print(f"Duplicate Rows       : {df.duplicated().sum():,}")

print(f"Negative Sales       : {(df['sales']<0).sum():,}")

print(f"Negative Prices      : {(df['sell_price']<0).sum():,}")

print()

print("Price Coverage (%)")

coverage = (
    100
    - df["sell_price"].isna().mean()*100
)

print(round(coverage,2))