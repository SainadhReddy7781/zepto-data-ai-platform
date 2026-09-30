import json
import re
from pathlib import Path

import pandas as pd
import joblib


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAQ_FILE = BASE_DIR / "data" / "faq.json"

PROJECT_ROOT = BASE_DIR.parent

CUSTOMER_DATA_FILE = (
    PROJECT_ROOT
    / "analytics"
    / "data"
    / "features"
    / "customer_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "analytics"
    / "models"
    / "random_forest.pkl"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "analytics"
    / "models"
    / "standard_scaler.pkl"
)

FEATURE_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "analytics"
    / "data"
    / "ml"
    / "feature_importance.csv"
)


# =========================================================
# LOAD FAQ
# =========================================================

def load_faq():
    """Load FAQ information from JSON."""

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# LOAD CUSTOMER DATA
# =========================================================

def load_customer_data():
    """Load customer analytics data."""

    return pd.read_csv(CUSTOMER_DATA_FILE)


# =========================================================
# LOAD ML MODEL
# =========================================================

def load_model():
    """Load Random Forest model and scaler."""

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    return model, scaler


# =========================================================
# LOAD FEATURE IMPORTANCE
# =========================================================

def load_feature_importance():
    """Load Random Forest feature importance."""

    return pd.read_csv(FEATURE_IMPORTANCE_FILE)


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text):
    """Convert text into lowercase words."""

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# =========================================================
# CUSTOMER INACTIVITY PREDICTION
# =========================================================

def predict_customer_inactivity(
    question,
    df,
    model,
    scaler,
    feature_importance
):
    """Predict customer inactivity using Random Forest."""

    question_lower = question.lower()

    prediction_keywords = [
        "predict inactivity",
        "predict inactive",
        "likely to be inactive",
        "will be inactive",
        "inactivity prediction",
        "inactive prediction"
    ]

    # -----------------------------------------------------
    # Check whether the question is about inactivity
    # -----------------------------------------------------

    if not any(
        keyword in question_lower
        for keyword in prediction_keywords
    ):
        return None

    # -----------------------------------------------------
    # Extract customer ID
    # -----------------------------------------------------

    customer_match = re.search(
        r"\bCUST_\d{4}\b",
        question.upper()
    )

    if not customer_match:
        return (
            "Please provide a customer ID such as "
            "CUST_0003 for inactivity prediction."
        )

    customer_id = customer_match.group()

    # -----------------------------------------------------
    # Find customer
    # -----------------------------------------------------

    customer = df[
        df["customer_id"].str.upper() == customer_id
    ]

    if customer.empty:
        return (
            f"I could not find {customer_id} "
            "in the customer dataset."
        )

    # -----------------------------------------------------
    # Features used during model training
    # -----------------------------------------------------

    features = [
        "total_orders",
        "total_spend",
        "average_order_value",
        "total_quantity",
        "average_discount",
        "average_rating",
        "unique_categories",
        "unique_products",
        "customer_lifetime_days"
    ]

    # -----------------------------------------------------
    # Select customer features
    # -----------------------------------------------------

    customer_features = customer[features]

    # -----------------------------------------------------
    # Apply scaler
    # -----------------------------------------------------

    scaled_features = scaler.transform(
        customer_features
    )

    # Convert back to DataFrame
    # This removes the sklearn feature-name warning.

    scaled_features = pd.DataFrame(
        scaled_features,
        columns=features,
        index=customer_features.index
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        scaled_features
    )[0]

    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    probability = model.predict_proba(
        scaled_features
    )[0][1]

    # -----------------------------------------------------
    # Prediction status
    # -----------------------------------------------------

    if prediction == 1:
        status = "likely to be inactive"
    else:
        status = "not likely to be inactive"

    # -----------------------------------------------------
    # Top feature importance
    # -----------------------------------------------------

    top_features = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False
        )
        .head(4)
    )

    # -----------------------------------------------------
    # Build explanation
    # -----------------------------------------------------

    explanation = (
        "\nKey factors considered by the model:\n"
    )

    for _, row in top_features.iterrows():

        feature_name = row["feature"]
        importance = row["importance"]

        explanation += (
            f"- {feature_name}: "
            f"{importance:.4f}\n"
        )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return (
        f"Customer {customer_id} is {status}.\n"
        f"Inactivity probability: {probability:.2%}"
        f"{explanation}"
    )


# =========================================================
# DATA-BASED QUESTIONS
# =========================================================

def answer_data_question(question, df):
    """Answer questions using customer analytics data."""

    question_lower = question.lower()

    # -----------------------------------------------------
    # CUSTOMER-SPECIFIC LOOKUP
    # -----------------------------------------------------

    customer_match = re.search(
        r"\bCUST_\d{4}\b",
        question.upper()
    )

    if customer_match:

        customer_id = customer_match.group()

        customer = df[
            df["customer_id"].str.upper() == customer_id
        ]

        if customer.empty:
            return (
                f"I could not find {customer_id} "
                "in the customer dataset."
            )

        customer = customer.iloc[0]

        return (
            f"Customer: {customer['customer_id']}\n"
            f"Total orders: "
            f"{int(customer['total_orders'])}\n"
            f"Total spend: "
            f"{customer['total_spend']:.2f}\n"
            f"Average order value: "
            f"{customer['average_order_value']:.2f}\n"
            f"Total quantity: "
            f"{int(customer['total_quantity'])}\n"
            f"Average discount: "
            f"{customer['average_discount']:.2f}%\n"
            f"Average rating: "
            f"{customer['average_rating']:.2f}\n"
            f"Unique categories: "
            f"{int(customer['unique_categories'])}\n"
            f"Unique products: "
            f"{int(customer['unique_products'])}\n"
            f"Days since last order: "
            f"{int(customer['days_since_last_order'])}\n"
            f"Customer lifetime: "
            f"{int(customer['customer_lifetime_days'])} days"
        )

    # -----------------------------------------------------
    # TOP 5 CUSTOMERS
    # -----------------------------------------------------

    if (
        "top 5 customers" in question_lower
        or "top five customers" in question_lower
        or "highest spending customers" in question_lower
    ):

        top_customers = (
            df.sort_values(
                "total_spend",
                ascending=False
            )
            .head(5)
        )

        result = "Top 5 customers by spending:\n"

        for index, (_, customer) in enumerate(
            top_customers.iterrows(),
            start=1
        ):

            result += (
                f"{index}. "
                f"{customer['customer_id']} - "
                f"{customer['total_spend']:.2f}\n"
            )

        return result.rstrip()

    # -----------------------------------------------------
    # CUSTOMER WITH MOST ORDERS
    # -----------------------------------------------------

    if (
        "customer has the most orders"
        in question_lower
        or "most orders" in question_lower
        or "highest number of orders"
        in question_lower
    ):

        top_customer = df.loc[
            df["total_orders"].idxmax()
        ]

        return (
            f"{top_customer['customer_id']} "
            f"has the most orders with "
            f"{int(top_customer['total_orders'])} orders."
        )

    # -----------------------------------------------------
    # NUMBER OF CUSTOMERS
    # -----------------------------------------------------

    if (
        "how many customers" in question_lower
        or "number of customers" in question_lower
        or "how many unique customers" in question_lower
    ):

        value = df["customer_id"].nunique()

        return (
            f"There are {value} customers "
            "in the customer analytics dataset."
        )

    # -----------------------------------------------------
    # AVERAGE ORDER VALUE
    # -----------------------------------------------------

    if (
        "average order value" in question_lower
        or "avg order value" in question_lower
    ):

        value = df["average_order_value"].mean()

        return (
            f"The average customer order value is "
            f"{value:.2f}."
        )

    # -----------------------------------------------------
    # AVERAGE CUSTOMER SPEND
    # -----------------------------------------------------

    if (
        "average customer spend" in question_lower
        or "average spend" in question_lower
        or "avg spend" in question_lower
    ):

        value = df["total_spend"].mean()

        return (
            f"The average customer spend is "
            f"{value:.2f}."
        )

    # -----------------------------------------------------
    # TOTAL REVENUE
    # -----------------------------------------------------

    if (
        "total revenue" in question_lower
        or "total spend" in question_lower
    ):

        value = df["total_spend"].sum()

        return (
            f"The total customer spend in the dataset is "
            f"{value:.2f}."
        )

    # -----------------------------------------------------
    # AVERAGE ORDERS
    # -----------------------------------------------------

    if (
        "average orders" in question_lower
        or "average number of orders" in question_lower
    ):

        value = df["total_orders"].mean()

        return (
            f"The average number of orders per customer is "
            f"{value:.2f}."
        )

    # -----------------------------------------------------
    # AVERAGE QUANTITY
    # -----------------------------------------------------

    if (
        "average quantity" in question_lower
        or "average products purchased" in question_lower
    ):

        value = df["total_quantity"].mean()

        return (
            f"The average quantity purchased per customer is "
            f"{value:.2f}."
        )

    # -----------------------------------------------------
    # AVERAGE DISCOUNT
    # -----------------------------------------------------

    if (
        "average discount" in question_lower
        or "avg discount" in question_lower
    ):

        value = df["average_discount"].mean()

        return (
            f"The average discount across customers is "
            f"{value:.2f}%."
        )

    # -----------------------------------------------------
    # AVERAGE RATING
    # -----------------------------------------------------

    if (
        "average rating" in question_lower
        or "avg rating" in question_lower
    ):

        value = df["average_rating"].mean()

        return (
            f"The average customer-associated "
            f"product rating is {value:.2f}."
        )

    # -----------------------------------------------------
    # UNIQUE PRODUCTS
    # -----------------------------------------------------

    if (
        "how many products" in question_lower
        or "number of products" in question_lower
        or "unique products" in question_lower
    ):

        value = df["unique_products"].sum()

        return (
            f"The customer dataset contains "
            f"{int(value)} total customer-level "
            f"unique-product counts."
        )

    return None


# =========================================================
# FAQ MATCHING
# =========================================================

def find_faq_answer(question, faq_data):
    """Find the most relevant FAQ using keyword similarity."""

    question_words = tokenize(question)

    best_match = None
    best_score = 0

    for item in faq_data:

        faq_text = (
            item["question"]
            + " "
            + item["answer"]
        )

        faq_words = tokenize(faq_text)

        common_words = (
            question_words.intersection(faq_words)
        )

        score = len(common_words)

        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score > 0:
        return best_match["answer"]

    return None


# =========================================================
# MAIN ASSISTANT
# =========================================================

def main():

    # -----------------------------------------------------
    # Load all resources
    # -----------------------------------------------------

    faq_data = load_faq()

    customer_df = load_customer_data()

    model, scaler = load_model()

    feature_importance = load_feature_importance()

    # -----------------------------------------------------
    # Display assistant header
    # -----------------------------------------------------

    print("=" * 60)

    print(
        "       ZEPTO DATA AI PLATFORM SUPPORT ASSISTANT"
    )

    print("=" * 60)

    print(
        "Ask questions about the project "
        "or customer analytics."
    )

    print(
        "Type 'exit' to stop the assistant."
    )

    print()

    # -----------------------------------------------------
    # Chat loop
    # -----------------------------------------------------

    while True:

        question = input("You: ").strip()

        # Exit
        if question.lower() == "exit":

            print("Assistant: Goodbye!")

            break

        # Empty input
        if not question:

            print(
                "Assistant: Please enter a question."
            )

            continue

        # -------------------------------------------------
        # 1. Machine learning prediction
        # -------------------------------------------------

        answer = predict_customer_inactivity(
            question,
            customer_df,
            model,
            scaler,
            feature_importance
        )

        # -------------------------------------------------
        # 2. Customer/data analytics
        # -------------------------------------------------

        if answer is None:

            answer = answer_data_question(
                question,
                customer_df
            )

        # -------------------------------------------------
        # 3. FAQ
        # -------------------------------------------------

        if answer is None:

            answer = find_faq_answer(
                question,
                faq_data
            )

        # -------------------------------------------------
        # 4. Fallback
        # -------------------------------------------------

        if answer is None:

            answer = (
                "Sorry, I don't have information "
                "about that yet. Please ask something "
                "related to the Zepto Data AI Platform."
            )

        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        print(f"Assistant: {answer}")

        print()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":
    main()