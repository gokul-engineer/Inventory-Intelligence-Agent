import pandas as pd


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:

    print("Creating Rolling Features...")

    group = ["item_id", "store_id"]

    df["rolling_mean_7"] = (
        df.groupby(group)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_mean_28"] = (
        df.groupby(group)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    print("Rolling Features Created")

    return df