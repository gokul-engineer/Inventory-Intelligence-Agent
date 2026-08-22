import pandas as pd


def add_price_features(df: pd.DataFrame):

    print("Creating Price Features...")

    group = ["item_id", "store_id"]

    df["price_change"] = (
        df.groupby(group)["sell_price"]
        .pct_change()
    )

    print("Price Features Created")

    return df