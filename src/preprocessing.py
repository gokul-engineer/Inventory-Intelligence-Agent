from data_loader import DataLoader
from utils import PROCESSED_DATA
import pandas as pd

loader = DataLoader()

print("=" * 70)
print("Loading datasets...")
print("=" * 70)

sales = loader.load_sales()
calendar = loader.load_calendar()
prices = loader.load_prices()

print("Sales Loaded")
print("Calendar Loaded")
print("Prices Loaded")

print("=" * 70)
print("Converting Wide Format -> Long Format")
print("=" * 70)

sales_long = pd.melt(
    sales,
    id_vars=[
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ],
    var_name="d",
    value_name="sales"
)

print("Done")

print(sales_long.head())

print("\nShape")

print(sales_long.shape)

print("=" * 70)
print("Merging Calendar...")
print("=" * 70)

sales_long = sales_long.merge(
    calendar[["d", "date", "wm_yr_wk"]],
    on="d",
    how="left"
)

print("Calendar merged")

print("=" * 70)
print("Merging Prices...")
print("=" * 70)

sales_long = sales_long.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

print("Prices merged")

print("\nFinal Shape")
print(sales_long.shape)

print("\nPreview")
print(sales_long.head())

print("=" * 70)
print("Saving Processed Dataset...")
print("=" * 70)

PROCESSED_DATA.mkdir(exist_ok=True)

sales_long.to_parquet(
    PROCESSED_DATA / "m5_processed.parquet",
    index=False
)

print("Saved Successfully!")