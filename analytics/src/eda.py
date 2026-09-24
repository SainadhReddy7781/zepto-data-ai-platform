from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "analytics"
    / "data"
    / "customer_orders.csv"
)

OUTPUT_DIR = BASE_DIR / "analytics" / "data" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    """Load the customer order dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["order_date"] = pd.to_datetime(df["order_date"])

    return df


# ---------------------------------------------------------
# Basic profiling
# ---------------------------------------------------------

def profile_data(df):
    """Generate basic dataset profiling information."""

    profile_file = OUTPUT_DIR / "data_profile.txt"

    with open(profile_file, "w", encoding="utf-8") as file:

        file.write("CUSTOMER ORDER DATA PROFILE\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Rows: {df.shape[0]}\n")
        file.write(f"Columns: {df.shape[1]}\n\n")

        file.write("Column Names:\n")
        file.write(str(list(df.columns)))
        file.write("\n\n")

        file.write("Data Types:\n")
        file.write(str(df.dtypes))
        file.write("\n\n")

        file.write("Missing Values:\n")
        file.write(str(df.isnull().sum()))
        file.write("\n\n")

        file.write("Duplicate Rows:\n")
        file.write(str(df.duplicated().sum()))
        file.write("\n\n")

        file.write("Numerical Summary:\n")
        file.write(str(df.describe()))
        file.write("\n\n")

        file.write("Category Distribution:\n")
        file.write(str(df["category"].value_counts().head(15)))

    print(f"Profile saved to: {profile_file}")


# ---------------------------------------------------------
# Category analysis
# ---------------------------------------------------------

def category_analysis(df):
    """Analyze orders and revenue by category."""

    category_summary = (
        df.groupby("category")
        .agg(
            orders=("order_id", "count"),
            revenue=("order_amount", "sum"),
            quantity=("quantity", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )

    output_file = OUTPUT_DIR / "category_summary.csv"

    category_summary.to_csv(output_file)

    print(f"Category summary saved to: {output_file}")

    # Top categories by revenue
    plt.figure(figsize=(10, 6))

    top_categories = category_summary.head(10)

    sns.barplot(
        x=top_categories["revenue"],
        y=top_categories.index,
    )

    plt.title("Top Categories by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Category")

    plt.tight_layout()

    chart_file = OUTPUT_DIR / "top_categories_revenue.png"

    plt.savefig(chart_file, dpi=150)

    plt.close()

    print(f"Chart saved to: {chart_file}")


# ---------------------------------------------------------
# Order amount distribution
# ---------------------------------------------------------

def order_amount_distribution(df):
    """Analyze the distribution of order amounts."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["order_amount"],
        bins=30,
        kde=True,
    )

    plt.title("Distribution of Order Amounts")
    plt.xlabel("Order Amount")
    plt.ylabel("Number of Orders")

    plt.tight_layout()

    chart_file = OUTPUT_DIR / "order_amount_distribution.png"

    plt.savefig(chart_file, dpi=150)

    plt.close()

    print(f"Chart saved to: {chart_file}")


# ---------------------------------------------------------
# Monthly revenue
# ---------------------------------------------------------

def monthly_revenue(df):
    """Calculate monthly revenue."""

    monthly = (
        df.set_index("order_date")
        .resample("ME")["order_amount"]
        .sum()
    )

    output_file = OUTPUT_DIR / "monthly_revenue.csv"

    monthly.to_csv(output_file)

    plt.figure(figsize=(10, 6))

    monthly.plot()

    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")

    plt.tight_layout()

    chart_file = OUTPUT_DIR / "monthly_revenue.png"

    plt.savefig(chart_file, dpi=150)

    plt.close()

    print(f"Monthly revenue saved to: {output_file}")
    print(f"Chart saved to: {chart_file}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 50)
    print("CUSTOMER ORDER EDA")
    print("=" * 50)

    df = load_data()

    print(f"Dataset loaded: {len(df)} rows")

    profile_data(df)

    category_analysis(df)

    order_amount_distribution(df)

    monthly_revenue(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()