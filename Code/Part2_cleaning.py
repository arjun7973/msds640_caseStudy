import pandas as pd

# Load dataset
df = pd.read_csv("20191226-items.csv")

# 1. Handle missing values (brand)
df['brand'].fillna('Unknown', inplace=True)

# 2. Clean numeric columns
# Ensure correct data types
df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
df['originalPrice'] = pd.to_numeric(df['originalPrice'], errors='coerce').fillna(0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['totalReviews'] = pd.to_numeric(df['totalReviews'], errors='coerce').fillna(0)

# 3. Check duplicates by 'asin'
df.drop_duplicates(subset=['asin'], inplace=True)

# 4. Standardize brand names (capitalize)
df['brand'] = df['brand'].str.title()

# 5. Drop unnecessary columns
df_cleaned = df.drop(columns=['url', 'image', 'reviewUrl'])

# Save cleaned dataset
df_cleaned.to_csv("cleaned_items.csv", index=False)

print("Data cleaning complete. Cleaned dataset saved as 'cleaned_items.csv'")
