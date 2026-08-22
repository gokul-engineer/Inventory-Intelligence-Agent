import pandas as pd

from utils import PROCESSED_DATA
from features.pipeline import create_features

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

df = pd.read_parquet(
    PROCESSED_DATA / "training_dataset.parquet"
)

# -------------------------
# DEVELOPMENT SAMPLE
# -------------------------

df = (
    df[df["store_id"] == "CA_1"]
    .head(100000)
    .copy()
)

print(f"Rows Loaded : {len(df):,}")

print("=" * 70)
print("CREATING FEATURES")
print("=" * 70)

df = create_features(df)

print(df.head())

print("\nColumns")

print(df.columns.tolist())

print("\nShape")

print(df.shape)