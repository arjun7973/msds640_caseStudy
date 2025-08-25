import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv("cleaned_items.csv")
except FileNotFoundError:
    df = pd.read_csv("20191226-items.csv")

# Bar chart: Top 10 brands by product count
brand_counts = df['brand'].value_counts().head(10)
plt.figure(figsize=(8,5))
brand_counts.plot(kind='bar')
plt.title("Top 10 Brands by Product Count")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_brands.png")
plt.close()

# Histogram: Price distribution
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=40, edgecolor='black')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("viz_price_distribution.png")
plt.close()

# Boxplot: Price variation across top 5 brands
top5_brands = df['brand'].value_counts().head(5).index
plt.figure(figsize=(8,6))
df[df['brand'].isin(top5_brands)].boxplot(column='price', by='brand')
plt.title("Price Variation Across Top 5 Brands")
plt.suptitle("")
plt.xlabel("Brand")
plt.ylabel("Price")
plt.savefig("boxplot_top5_brands.png")
plt.close()

print("Visualization plots saved: top10_brands.png, viz_price_distribution.png, boxplot_top5_brands.png")