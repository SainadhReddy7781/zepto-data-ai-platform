from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = BASE_DIR / "analytics" / "models" / "random_forest.pkl"
OUTPUT_DIR = BASE_DIR / "analytics" / "data" / "ml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = joblib.load(MODEL_FILE)

    feature_columns = [
        "total_orders",
        "total_spend",
        "average_order_value",
        "total_quantity",
        "average_discount",
        "average_rating",
        "unique_categories",
        "unique_products",
        "customer_lifetime_days",
    ]

    importances = model.feature_importances_

    print(f"Model features: {model.n_features_in_}")
    print(f"Feature names: {len(feature_columns)}")
    print(f"Importance values: {len(importances)}")

    if len(feature_columns) != len(importances):
        raise ValueError(
            "Feature name count does not match model importance count."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    output_file = OUTPUT_DIR / "feature_importance.csv"
    importance_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    chart_file = OUTPUT_DIR / "feature_importance.png"

    plt.savefig(chart_file, dpi=150)
    plt.close()

    print(f"\nFeature importance saved to: {output_file}")
    print(f"Chart saved to: {chart_file}")
    print("\nFeature importance analysis completed successfully.")


if __name__ == "__main__":
    main()