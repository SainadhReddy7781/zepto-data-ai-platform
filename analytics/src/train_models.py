from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "analytics" / "data" / "ml" / "customer_ml_data.csv"
MODEL_DIR = BASE_DIR / "analytics" / "models"
OUTPUT_DIR = BASE_DIR / "analytics" / "data" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)

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

    X = df[feature_columns]
    y = df["inactive_customer"]

    return X, y, feature_columns


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def save_confusion_matrix(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Active", "Inactive"])
    plt.yticks([0, 1], ["Active", "Inactive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    output_file = OUTPUT_DIR / filename

    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {output_file}")


def main():
    print("=" * 60)
    print("CUSTOMER INACTIVITY MODEL TRAINING")
    print("=" * 60)

    X, y, feature_columns = load_data()

    print(f"Dataset rows: {len(X)}")
    print(f"Features used: {len(feature_columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    logistic_model.fit(X_train_scaled, y_train)

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test_scaled,
        y_test,
    )

    save_confusion_matrix(
        "Logistic Regression",
        logistic_model,
        X_test_scaled,
        y_test,
    )

    joblib.dump(
        logistic_model,
        MODEL_DIR / "logistic_regression.pkl",
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "standard_scaler.pkl",
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    random_forest.fit(X_train, y_train)

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest,
        X_test,
        y_test,
    )

    save_confusion_matrix(
        "Random Forest",
        random_forest,
        X_test,
        y_test,
    )

    joblib.dump(
        random_forest,
        MODEL_DIR / "random_forest.pkl",
    )

    # -----------------------------
    # Compare Models
    # -----------------------------
    results = pd.DataFrame(
        [
            logistic_results,
            random_forest_results,
        ]
    )

    results_file = OUTPUT_DIR / "model_comparison.csv"
    results.to_csv(results_file, index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results.to_string(index=False))

    print(f"\nModel comparison saved to: {results_file}")
    print(f"Models saved to: {MODEL_DIR}")

    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()