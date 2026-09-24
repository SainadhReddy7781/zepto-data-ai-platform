from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "analytics" / "data" / "features" / "customer_features.csv"
OUTPUT_DIR = BASE_DIR / "analytics" / "data" / "ml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_ml_data():
    df = pd.read_csv(INPUT_FILE)

    # Define inactivity based on recency.
    # Customers with more than 60 days since their last order
    # are classified as inactive.
    df["inactive_customer"] = (
        df["days_since_last_order"] > 60
    ).astype(int)

    output_file = OUTPUT_DIR / "customer_ml_data.csv"
    df.to_csv(output_file, index=False)

    print("=" * 60)
    print("ML DATA PREPARATION")
    print("=" * 60)

    print(f"Customers: {len(df)}")
    print(f"Features: {df.shape[1]}")

    print("\nTarget distribution:")
    print(df["inactive_customer"].value_counts())

    print("\nTarget percentages:")
    print(
        (df["inactive_customer"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print(f"\nML dataset saved to: {output_file}")


def main():
    prepare_ml_data()
    print("\nML data preparation completed successfully.")


if __name__ == "__main__":
    main()