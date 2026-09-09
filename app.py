import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Inventory Intelligence Dashboard",
    page_icon="📦",
    layout="wide"
)

from src.features.pipeline import create_features
from src.inventory import calculate_inventory_metrics
from src.utils import PROCESSED_DATA

MODEL_DIR = Path(__file__).resolve().parent / "models"

@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_DIR / "xgboost_model.pkl"
    )

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_parquet(
        PROCESSED_DATA / "training_dataset.parquet"
    )

    df = (
    df[df["store_id"] == "CA_1"]
    .head(100)
    .copy()
    )

    return df

df = load_data()


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
    "price_change",
]

@st.cache_data
def predict_inventory():
    df = load_data()

    df = create_features(df)

    df = df.dropna()

    predictions = model.predict(df[feature_columns])

    df["prediction"] = predictions

    df = calculate_inventory_metrics(
        df,
        current_stock=current_stock,
        lead_time=lead_time,
        safety_stock=safety_stock
    )

    return df


current_stock = st.sidebar.number_input(
    "Current Stock",
    min_value=0,
    value=20
)

lead_time = st.sidebar.slider(
    "Lead Time (Days)",
    1,
    30,
    7
)

safety_stock = st.sidebar.slider(
    "Safety Stock",
    0,
    100,
    15
)

df = calculate_inventory_metrics(
    df,
    current_stock=current_stock,
    lead_time=lead_time,
    safety_stock=safety_stock
)


st.title("📦 Inventory Intelligence Agent")
st.caption("Demand Forecasting & Inventory Recommendation System")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Forecast Demand",
        value=f"{df['prediction'].iloc[0]:.2f}"
    )

with col2:
    st.metric(
        label="Current Stock",
        value=current_stock
    )

with col3:
    st.metric(
        label="Reorder Point",
        value=f"{df['reorder_point'].iloc[0]:.2f}"
    )

with col4:
    st.metric(
        label="Recommended Order",
        value=f"{df['recommended_order_qty'].iloc[0]:.2f}"
    )

st.divider()

st.subheader("Forecast Results")

st.dataframe(
    df[
        [
            "date",
            "prediction",
            "reorder_point",
            "recommended_order_qty"
        ]
    ].head(20),
    use_container_width=True
)

st.divider()

st.subheader("Inventory Recommendation")

st.dataframe(
    df[
        [
            "date",
            "item_id",
            "sales",
            "prediction"
        ]
    ].head(20),
    use_container_width=True
)