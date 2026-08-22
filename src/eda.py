from data_loader import DataLoader

loader = DataLoader()

calendar = loader.load_calendar()
sales = loader.load_sales()
prices = loader.load_prices()

print("=" * 70)
print("M5 DATASET SUMMARY")
print("=" * 70)

print("\nCalendar")
print(calendar.info())

print("\nSales")
print(sales.info())

print("\nPrices")
print(prices.info())

print("\n")

print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(calendar.isnull().sum())

print("=" * 70)
print("SALES")
print("=" * 70)

print(sales.isnull().sum())

print("=" * 70)
print("PRICES")
print("=" * 70)

print(prices.isnull().sum())