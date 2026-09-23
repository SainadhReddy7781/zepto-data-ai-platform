import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "products_raw.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load product data from the raw JSON file."""
    with open(RAW_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["products"]


def clean_data(products):
    """Clean and validate product records."""

    df = pd.DataFrame(products)

    # Keep only the fields required for analysis
    columns = [
        "id",
        "title",
        "category",
        "price",
        "discountPercentage",
        "rating",
        "stock",
        "brand",
        "sku",
    ]

    df = df[columns]

    # Remove duplicate products
    df = df.drop_duplicates(subset=["id"])

    # Handle missing values
    df["brand"] = df["brand"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")

    # Convert numeric columns
    numeric_columns = [
        "price",
        "discountPercentage",
        "rating",
        "stock",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove records without essential values
    df = df.dropna(subset=["id", "title", "price", "category"])

    # Make sure prices and stock values are valid
    df = df[df["price"] >= 0]
    df = df[df["stock"] >= 0]

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    return df


def save_clean_data(df):
    """Save the cleaned dataset as CSV."""

    output_file = PROCESSED_DIR / "products_clean.csv"

    df.to_csv(output_file, index=False)

    print(f"Cleaned data saved to: {output_file}")
    print(f"Clean records: {len(df)}")


def main():
    print("Starting data cleaning...")

    products = load_raw_data()

    print(f"Raw records: {len(products)}")

    cleaned_df = clean_data(products)

    save_clean_data(cleaned_df)

    print("Data cleaning completed successfully.")


if __name__ == "__main__":
    main()