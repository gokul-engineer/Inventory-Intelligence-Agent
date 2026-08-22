import pandas as pd
import joblib
from pathlib import Path

from utils import PROCESSED_DATA
from features.pipeline import create_features

from sklearn.linear_model import LinearRegression

from model_utils import evaluate_model

print("=" * 70)
print("TRAINING MODEL")
print("=" * 70)

df = pd.read_parquet(
    PROCESSED_DATA / "training_dataset.parquet"
)

# Development sample
df = (
    df[df["store_id"] == "CA_1"]
    .head(100000)
    .copy()
)

df = create_features(df)

# Remove rows created by lag/rolling features
df = df.dropna()

print(f"Rows after Feature Engineering : {len(df):,}")

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

target = "sales"

# -------------------------
# Time Series Split
# -------------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]

test = df.iloc[split_index:]

X_train = train[feature_columns]
y_train = train[target]

X_test = test[feature_columns]
y_test = test[target]

print(f"Training Rows : {len(train):,}")
print(f"Testing Rows  : {len(test):,}")

# -------------------------
# Baseline Model
# -------------------------

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

evaluate_model(y_test, predictions)

print("=" * 60)
print("SAVING MODEL")
print("=" * 60)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "baseline_model.pkl"
)

print("Model Saved Successfully!")