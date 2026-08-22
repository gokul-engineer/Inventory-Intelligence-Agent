import pandas as pd

from .lag_features import add_lag_features
from .rolling_features import add_rolling_features
from .calendar_features import add_calendar_features
from .price_features import add_price_features


def create_features(df: pd.DataFrame):

    df = add_lag_features(df)

    df = add_rolling_features(df)

    df = add_calendar_features(df)

    df = add_price_features(df)

    return df