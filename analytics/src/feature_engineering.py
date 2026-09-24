from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "analytics" / "data" / "customer_orders.csv"
OUTPUT_DIR = BASE_DIR / "analytics" / "data" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def create_customer_features(df):
    reference_date = df["order_date"].max()

    customer_features = (
        df.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            total_spend=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
            total_quantity=("quantity", "sum"),
            average_discount=("discount_percentage", "mean"),
            average_rating=("product_rating", "mean"),
            unique_categories=("category", "nunique"),
            unique_products=("product_id", "nunique"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    customer_features["days_since_last_order"] = (
        reference_date - customer_features["last_order_date"]
    ).dt.days

    customer_features["customer_lifetime_days"] = (
        df.groupby("customer_id")["order_date"].max()
        - df.groupby("customer_id")["order_date"].min()
    ).dt.days.values

    customer_features.drop(columns=["last_order_date"], inplace=True)

    return customer_features


def save_features(customer_features):
    output_file = OUTPUT_DIR / "customer_features.csv"
    customer_features.to_csv(output_file, index=False)

    print(f"Customer features saved to: {output_file}")
    print(f"Customers: {len(customer_features)}")
    print(f"Features: {customer_features.shape[1]}")
    print("\nFeature columns:")
    print(list(customer_features.columns))


def main():
    print("=" * 60)
    print("CUSTOMER FEATURE ENGINEERING")
    print("=" * 60)

    df = load_data()

    print(f"Orders loaded: {len(df)}")

    customer_features = create_customer_features(df)

    save_features(customer_features)

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()