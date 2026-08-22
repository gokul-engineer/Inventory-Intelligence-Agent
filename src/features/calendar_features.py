import pandas as pd


def add_calendar_features(df: pd.DataFrame):

    print("Creating Calendar Features...")

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday

    df["weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    print("Calendar Features Created")

    return df