from data_loader import DataLoader

loader = DataLoader()

calendar = loader.load_calendar()
sales = loader.load_sales()
prices = loader.load_prices()

print("="*70)
print("BUSINESS SUMMARY")
print("="*70)

print(f"Products           : {sales['item_id'].nunique():,}")
print(f"Departments        : {sales['dept_id'].nunique():,}")
print(f"Categories         : {sales['cat_id'].nunique():,}")
print(f"Stores             : {sales['store_id'].nunique():,}")
print(f"States             : {sales['state_id'].nunique():,}")

print("\nStore Names")
print(sales['store_id'].unique())

print("\nState Names")
print(sales['state_id'].unique())

print("\nCategory Names")
print(sales['cat_id'].unique())

print("\nDepartment Names")
print(sales['dept_id'].unique())