import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for demand forecasting.
    """

    print("Creating Lag Features...")

    group = ["item_id", "store_id"]

    df["lag_1"] = (
        df.groupby(group)["sales"]
        .shift(1)
    )

    df["lag_7"] = (
        df.groupby(group)["sales"]
        .shift(7)
    )

    df["lag_28"] = (
        df.groupby(group)["sales"]
        .shift(28)
    )

    print("Lag Features Created")

    return df