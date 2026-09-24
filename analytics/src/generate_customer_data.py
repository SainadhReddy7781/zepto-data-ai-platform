from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PRODUCT_FILE = (
    BASE_DIR
    / "data_pipeline"
    / "data"
    / "processed"
    / "products_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "analytics" / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "customer_orders.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42
NUMBER_OF_CUSTOMERS = 300
NUMBER_OF_ORDERS = 1500


# ---------------------------------------------------------
# Generate synthetic customer orders
# ---------------------------------------------------------

def generate_customer_orders(products):
    """Generate reproducible synthetic customer-order data."""

    rng = np.random.default_rng(RANDOM_SEED)

    customer_ids = [
        f"CUST_{number:04d}"
        for number in range(1, NUMBER_OF_CUSTOMERS + 1)
    ]

    orders = []

    for order_number in range(1, NUMBER_OF_ORDERS + 1):

        customer_id = rng.choice(customer_ids)

        product = products.iloc[
            rng.integers(0, len(products))
        ]

        quantity = int(rng.integers(1, 6))

        order_date = pd.Timestamp("2026-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 180))
        )

        unit_price = float(product["price"])

        discount = float(product["discountPercentage"])

        discounted_price = unit_price * (1 - discount / 100)

        order_amount = round(
            discounted_price * quantity,
            2
        )

        orders.append(
            {
                "order_id": f"ORD_{order_number:05d}",
                "customer_id": customer_id,
                "order_date": order_date.date(),
                "product_id": int(product["id"]),
                "product_name": product["title"],
                "category": product["category"],
                "brand": product["brand"],
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_percentage": discount,
                "order_amount": order_amount,
                "product_rating": float(product["rating"]),
            }
        )

    return pd.DataFrame(orders)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading product catalog...")

    products = pd.read_csv(PRODUCT_FILE)

    print(f"Products available: {len(products)}")

    print("Generating synthetic customer orders...")

    orders_df = generate_customer_orders(products)

    orders_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Orders generated: {len(orders_df)}")

    print(f"Dataset saved to: {OUTPUT_FILE}")

    print("\nCustomer order dataset created successfully.")


if __name__ == "__main__":
    main()