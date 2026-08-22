import pandas as pd


def calculate_inventory_metrics(
    df: pd.DataFrame,
    current_stock: int,
    lead_time: int = 7,
    safety_stock: int = 20,
):

    df = df.copy()

    df["forecast_daily_demand"] = df["prediction"]

    df["lead_time_demand"] = (
        df["forecast_daily_demand"] * lead_time
    )

    df["reorder_point"] = (
        df["lead_time_demand"] + safety_stock
    )

    df["current_stock"] = current_stock

    df["recommended_order_qty"] = (
        df["reorder_point"] - current_stock
    ).clip(lower=0)

    return df
