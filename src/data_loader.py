from pathlib import Path
import pandas as pd
from utils import RAW_DATA


class DataLoader:
    """Loads all raw M5 datasets."""

    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent
        self.raw_path = RAW_DATA

    def load_calendar(self):
        return pd.read_csv(self.raw_path / "calendar.csv")

    def load_sales(self):
        return pd.read_csv(self.raw_path / "sales_train_validation.csv")

    def load_prices(self):
        return pd.read_csv(self.raw_path / "sell_prices.csv")


if __name__ == "__main__":

    loader = DataLoader()

    calendar = loader.load_calendar()
    sales = loader.load_sales()
    prices = loader.load_prices()

    print(calendar.shape)
    print(sales.shape)
    print(prices.shape)