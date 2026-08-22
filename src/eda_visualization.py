from data_loader import DataLoader
import plotly.express as px

loader = DataLoader()

sales = loader.load_sales()

sales_columns = [c for c in sales.columns if c.startswith("d_")]

sales["Total_Sales"] = sales[sales_columns].sum(axis=1)

# -------------------------
# Sales by Category
# -------------------------

category_sales = (
    sales.groupby("cat_id")["Total_Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    category_sales,
    x="cat_id",
    y="Total_Sales",
    title="Total Sales by Category",
    text_auto=True
)

fig.show()

# -------------------------
# Sales by Store
# -------------------------

store_sales = (
    sales.groupby("store_id")["Total_Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    store_sales,
    x="store_id",
    y="Total_Sales",
    title="Total Sales by Store",
    text_auto=True
)

fig.show()