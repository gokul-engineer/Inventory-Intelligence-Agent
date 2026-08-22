from data_loader import DataLoader

loader = DataLoader()

sales = loader.load_sales()

sales_columns = [col for col in sales.columns if col.startswith("d_")]

sales["Total_Sales"] = sales[sales_columns].sum(axis=1)
sales["Average_Daily_Sales"] = sales[sales_columns].mean(axis=1)
sales["Days_With_Sales"] = (sales[sales_columns] > 0).sum(axis=1)
sales["Zero_Sales_Days"] = (sales[sales_columns] == 0).sum(axis=1)

print("=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

print(f"Average Daily Sales : {sales['Average_Daily_Sales'].mean():.2f}")
print(f"Maximum Total Sales : {sales['Total_Sales'].max():,.0f}")
print(f"Minimum Total Sales : {sales['Total_Sales'].min():,.0f}")
print(f"Average Total Sales : {sales['Total_Sales'].mean():,.0f}")

print("\nTop 10 Selling Products")
print(
    sales[["item_id", "store_id", "Total_Sales"]]
    .sort_values("Total_Sales", ascending=False)
    .head(10)
)

print("\nProducts Never Sold")
print((sales["Total_Sales"] == 0).sum())

print("\nAverage Zero Sale Days")
print(round(sales["Zero_Sales_Days"].mean(), 2))

print("\nTop 10 Categories by Total Sales")
print(
    sales.groupby("cat_id")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Departments by Total Sales")
print(
    sales.groupby("dept_id")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop Stores by Total Sales")
print(
    sales.groupby("store_id")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)