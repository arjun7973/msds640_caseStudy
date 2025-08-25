#!/usr/bin/env python3
"""
full_project.py

Combined 8-part project for the dataset `20191226-items.csv`:
1) Overview
2) Cleaning & Preprocessing
3) EDA
4) Visualization
5) Predictive Modeling
6) Ethical Considerations
7) Social Justice Impact
8) Conclusions & Recommendations

Dependencies:
- pandas, numpy, matplotlib
- scikit-learn (optional; only used in Part 5).

Usage:
    python full_project.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: sklearn only used in Part 5
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

DATA_RAW = "20191226-items.csv"
CLEANED_CSV = "cleaned_items.csv"


# ---------------- Part 1: Overview ----------------
def part1_overview(df: pd.DataFrame):
    print("\n========== Part 1: Introduction & Dataset Overview ==========")
    print("Loaded dataset:", DATA_RAW)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample rows:")
    with pd.option_context("display.max_colwidth", 80):
        print(df.head(5))
    print("\nMissing values per column:")
    print(df.isnull().sum())


# ---------------- Part 2: Cleaning & Preprocessing ----------------
def part2_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("\n========== Part 2: Data Cleaning & Preprocessing ==========")
    df = df.copy()

    # 1) Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # 2) Basic missing handling
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").str.title()
    if "category" in df.columns:
        df["category"] = df["category"].fillna("Uncategorized").astype(str).str.strip()

    # 3) Coerce numeric types
    for col in ["price", "originalPrice", "rating", "totalReviews"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Remove duplicate ASINs if present
    if "asin" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["asin"])
        after = len(df)
        print(f"Dropped {before - after} duplicate rows by ASIN.")

    # 5) Derive discount flags
    if {"price", "originalPrice"}.issubset(df.columns):
        df["has_discount"] = np.where(
            (df["originalPrice"].notna()) & (df["price"].notna()) &
            (df["originalPrice"] > df["price"]) & (df["price"] > 0),
            1, 0
        )
        df["discount_pct"] = np.where(
            df["has_discount"] == 1,
            1 - (df["price"] / df["originalPrice"]),
            np.nan
        )

    # 6) Export a lean cleaned CSV (drop heavy URL-like columns if present)
    drop_cols = [c for c in ["url", "image", "reviewUrl"] if c in df.columns]
    df_export = df.drop(columns=drop_cols) if drop_cols else df
    df_export.to_csv(CLEANED_CSV, index=False)
    print(f"Saved cleaned dataset -> {CLEANED_CSV} ({df_export.shape[0]} rows, {df_export.shape[1]} cols)")

    return df_export


# ---------------- Part 3: EDA ----------------
def part3_eda(df: pd.DataFrame):
    print("\n========== Part 3: Exploratory Data Analysis ==========")
    numeric_cols = [c for c in ["price", "originalPrice", "rating", "totalReviews"] if c in df.columns]
    if numeric_cols:
        print("Numeric summary:\n", df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T)

    if "brand" in df.columns:
        print("\nTop 10 brands by count:")
        print(df["brand"].value_counts().head(10))

    if "category" in df.columns:
        print("\nTop 10 categories by count:")
        print(df["category"].value_counts().head(10))

    # Simple correlations (if at least 2 numeric columns present)
    if len(numeric_cols) >= 2:
        print("\nPearson correlations among numeric fields:")
        print(df[numeric_cols].corr(method="pearson"))


# ---------------- Part 4: Visualization ----------------
def part4_visualization(df: pd.DataFrame):
    print("\n========== Part 4: Data Visualization ==========")
    os.makedirs("plots", exist_ok=True)

    # 1) Price distribution
    if "price" in df.columns:
        plt.figure()
        plt.hist(df["price"].dropna(), bins=40)
        plt.title("Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.tight_layout()
        out = os.path.join("plots", "price_distribution.png")
        plt.savefig(out)
        plt.close()
        print("Saved plot:", out)

    # 2) Rating distribution
    if "rating" in df.columns:
        plt.figure()
        plt.hist(df["rating"].dropna(), bins=20)
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        out = os.path.join("plots", "rating_distribution.png")
        plt.savefig(out)
        plt.close()
        print("Saved plot:", out)

    # 3) Top 10 brands bar
    if "brand" in df.columns:
        top = df["brand"].value_counts().head(10)
        plt.figure()
        top.plot(kind="bar")
        plt.title("Top 10 Brands by Product Count")
        plt.xlabel("Brand")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out = os.path.join("plots", "top10_brands.png")
        plt.savefig(out)
        plt.close()
        print("Saved plot:", out)

    # 4) Price vs Rating scatter
    if {"price", "rating"}.issubset(df.columns):
        sub = df[["price", "rating"]].dropna()
        if not sub.empty:
            plt.figure()
            plt.scatter(sub["price"], sub["rating"], alpha=0.4, s=10)
            plt.title("Price vs Rating")
            plt.xlabel("Price")
            plt.ylabel("Rating")
            plt.tight_layout()
            out = os.path.join("plots", "price_vs_rating.png")
            plt.savefig(out)
            plt.close()
            print("Saved plot:", out)


# ---------------- Part 5: Predictive Modeling ----------------
def part5_modeling(df: pd.DataFrame):
    print("\n========== Part 5: Predictive Modeling ==========")
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available; skipping modeling step.")
        return

    needed = {"price", "brand", "category"}
    if not needed.issubset(df.columns):
        print(f"Missing columns for modeling: {sorted(list(needed - set(df.columns)))}")
        return

    # Use rows with price > 0 and non-null
    data = df.copy()
    data = data[data["price"].notna() & (data["price"] > 0)]
    data["brand"] = data["brand"].astype(str)
    data["category"] = data["category"].astype(str)

    # Encode categoricals
    le_brand = LabelEncoder()
    le_cat = LabelEncoder()
    data["brand_enc"] = le_brand.fit_transform(data["brand"])
    data["category_enc"] = le_cat.fit_transform(data["category"])

    X = data[["brand_enc", "category_enc"]]
    y = data["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression -> MSE: {mse:.4f} | R^2: {r2:.4f}")

    # Save diagnostics
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("model_diagnostics.csv", index=False)
    print("Saved: model_diagnostics.csv")


# ---------------- Part 6: Ethical Considerations ----------------
def part6_ethics(df: pd.DataFrame):
    print("\n========== Part 6: Ethical Considerations ==========")

    # Sensitive columns check (dataset should avoid these; if present, treat carefully)
    for col in ["gender", "race", "age"]:
        print(f"Sensitive column present? {col}: {col in df.columns}")

    # Representation: brand frequency
    if "brand" in df.columns:
        counts = df["brand"].value_counts()
        print("\nTop 10 brands by count:")
        print(counts.head(10))

    # Price inequality (avg price per brand)
    if {"brand", "price"}.issubset(df.columns):
        avg_price = df.groupby("brand")["price"].mean().sort_values(ascending=False)
        print("\nAverage price per brand (top 10):")
        print(avg_price.head(10))


# ---------------- Part 7: Social Justice Impact ----------------
def part7_social_justice(df: pd.DataFrame):
    print("\n========== Part 7: Social Justice Impact ==========")

    # Price accessibility: share of affordable items at thresholds
    if "price" in df.columns:
        p = df["price"].dropna()
        if not p.empty:
            for thr in [50, 100, 200, 300]:
                share = (p <= thr).mean()
                print(f"Share of items priced <= {thr}: {share:.2%}")

    # Global vs Local brand proxy (by frequency)
    if "brand" in df.columns:
        counts = df["brand"].value_counts()
        local = (counts < 5).sum()
        global_ = (counts >= 5).sum()
        print(f"Global brands (>=5 items): {global_} | Local brands (<5 items): {local}")


# ---------------- Part 8: Conclusions & Recommendations ----------------
def part8_conclusions(df: pd.DataFrame):
    print("\n========== Part 8: Conclusions & Recommendations ==========")
    print("Dataset shape:", df.shape)
    print("Missing values summary:\n", df.isnull().sum())

    print("\nKey Findings:")
    print("- Significant variance in prices and brand/category representation.")
    print("- Basic linear regression provides a baseline but may miss complex relations.")
    print("- Potential bias if models overfit dominant brands/categories.")
    print("- No explicit sensitive attributes detected in most schemas, which is privacy-positive.")

    print("\nRecommendations:")
    print("1) Publish transparent documentation for cleaning and modeling steps.")
    print("2) Run periodic fairness audits across brands/categories and regions.")
    print("3) If user-level data is added, use privacy-preserving techniques (aggregation, DP).")
    print("4) Balance representation by sampling/weighting to avoid dominance of a few brands.")
    print("5) Communicate uncertainty and avoid overstating causal claims from observational data.")


def main():
    if not os.path.exists(DATA_RAW):
        print(f"ERROR: '{DATA_RAW}' not found. Place the CSV in this folder and rerun.")
        sys.exit(1)

    # Load
    df_raw = pd.read_csv(DATA_RAW)

    # Run pipeline
    part1_overview(df_raw)
    df_clean = part2_cleaning(df_raw)
    part3_eda(df_clean)
    part4_visualization(df_clean)
    part5_modeling(df_clean)
    part6_ethics(df_clean)
    part7_social_justice(df_clean)
    part8_conclusions(df_clean)

    print("\nAll parts completed. Outputs saved in ./plots and model_diagnostics.csv (if sklearn available).")


if __name__ == "__main__":
    main()
