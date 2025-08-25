import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (cleaned if available)
try:
    df = pd.read_csv("cleaned_items.csv")
except FileNotFoundError:
    df = pd.read_csv("20191226-items.csv")

print("Dataset shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe())

# Top 10 brands by product count
top_brands = df['brand'].value_counts().head(10)
print("\nTop 10 Brands by Count:")
print(top_brands)

# Distribution of prices
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=50, edgecolor='black')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("price_distribution.png")
plt.close()

# Distribution of ratings
plt.figure(figsize=(6,4))
plt.hist(df['rating'], bins=20, edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig("rating_distribution.png")
plt.close()

# Scatter plot: Price vs Rating
plt.figure(figsize=(6,4))
plt.scatter(df['price'], df['rating'], alpha=0.3)
plt.title("Price vs Rating")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.savefig("price_vs_rating.png")
plt.close()

print("EDA complete. Plots saved as PNG files.")