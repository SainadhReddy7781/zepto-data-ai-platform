import json
from pathlib import Path

import requests


API_URL = "https://dummyjson.com/products?limit=100"

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_products():
    """Fetch product data from the public API."""
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data


def save_raw_data(data):
    """Save the raw API response as JSON."""
    output_file = RAW_DATA_DIR / "products_raw.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Raw data saved to: {output_file}")


def main():
    print("Starting product data extraction...")

    data = fetch_products()

    print(f"Products fetched: {len(data.get('products', []))}")

    save_raw_data(data)

    print("Data extraction completed successfully.")


if __name__ == "__main__":
    main()