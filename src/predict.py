import joblib
import pandas as pd
from pathlib import Path

from utils import PROCESSED_DATA
from features.pipeline import create_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_DIR / "xgboost_model.pkl"
)

df = pd.read_parquet(
    PROCESSED_DATA / "training_dataset.parquet"
)

df = (
    df[df["store_id"] == "CA_1"]
    .head(100)
    .copy()
)

df = create_features(df)

df = df.dropna()

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "year",
    "month",
    "day",
    "weekday",
    "weekend",
    "sell_price",
    "price_change"
]

predictions = model.predict(df[feature_columns])

df["prediction"] = predictions

from inventory import calculate_inventory_metrics

df = calculate_inventory_metrics(
    df,
    current_stock=20,
    lead_time=7,
    safety_stock=15
)

print(
    df[
        [
            "date",
            "item_id",
            "sales",
            "prediction",
            "reorder_point",
            "recommended_order_qty"
        ]
    ].head(20)
)