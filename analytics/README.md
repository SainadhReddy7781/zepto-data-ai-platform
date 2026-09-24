# Analytics Module

## 1. Overview

The Analytics module performs exploratory data analysis, customer-level feature engineering, and machine learning on synthetic customer-order data generated from the product data collected by the Data Pipeline.

The workflow includes:

1. Exploratory Data Analysis (EDA)
2. Customer-level feature engineering
3. ML dataset preparation
4. Logistic Regression modeling
5. Random Forest modeling
6. Model evaluation
7. Random Forest feature-importance analysis

> **Important:** The customer and order data used in this module are synthetic. They were generated for this capstone project using product data collected by the Data Pipeline. The results should not be interpreted as real Zepto customer behavior or business performance.

---

## 2. Input Data

The product information originates from the Data Pipeline:

`data_pipeline/data/processed/products_clean.csv`

This product data is used to generate a synthetic customer-order dataset.

Generated dataset:

`analytics/data/customer_orders.csv`

Dataset information:

- Product records available: 100
- Generated customer IDs: 300
- Generated orders: 1,500
- Customers appearing in the generated order dataset: 298

The customer-order dataset contains:

- `order_id`
- `customer_id`
- `order_date`
- `product_id`
- `product_name`
- `category`
- `brand`
- `quantity`
- `unit_price`
- `discount_percentage`
- `order_amount`
- `product_rating`

---

## 3. Exploratory Data Analysis

The EDA process is implemented in:

`analytics/src/eda.py`

The generated EDA outputs are stored in:

`analytics/data/eda/`

Generated files include:

- `data_profile.txt`
- `category_summary.csv`
- `monthly_revenue.csv`
- `top_categories_revenue.png`
- `order_amount_distribution.png`
- `monthly_revenue.png`

### Category-Level Findings

The following observations were identified within the synthetic dataset:

- `kitchen-accessories` recorded **423 orders** and generated **22,662.04** in revenue.
- `groceries` recorded **407 orders** and **1,213 units**, generating **5,858.16** in revenue.
- `mens-shoes` generated **21,381.27** in revenue from **71 orders**.
- `fragrances` generated **15,509.59** in revenue from **66 orders**.
- `mobile-accessories` generated **11,599.77** in revenue from **41 orders**.

These results demonstrate that order volume and revenue can differ substantially between product categories.

Because the dataset is synthetic, these findings describe the generated dataset only and should not be interpreted as real Zepto business performance.

---

## 4. Customer Feature Engineering

Customer-level features are generated using:

`analytics/src/feature_engineering.py`

The resulting dataset is:

`analytics/data/features/customer_features.csv`

The feature set contains:

- `customer_id`
- `total_orders`
- `total_spend`
- `average_order_value`
- `total_quantity`
- `average_discount`
- `average_rating`
- `unique_categories`
- `unique_products`
- `days_since_last_order`
- `customer_lifetime_days`

A total of **298 customers** appear in the generated order dataset.

---

## 5. ML Target Definition

The ML preparation process is implemented in:

`analytics/src/prepare_ml_data.py`

The target variable is:

`inactive_customer`

The project defines a customer as inactive when:

`days_since_last_order > 60`

Therefore:

- `inactive_customer = 1` represents an inactive customer.
- `inactive_customer = 0` represents an active customer.

### Target Distribution

| Class | Percentage |
|---|---:|
| Active customers | 80.2% |
| Inactive customers | 19.8% |

The 60-day threshold is a project-defined modeling assumption and is not a real Zepto business rule.

---

## 6. Machine Learning Methodology

Two classification models were trained.

### Logistic Regression

Logistic Regression was used as a baseline classification model.

### Random Forest

Random Forest was used as a nonlinear ensemble classification model.

The dataset was divided into:

- **80% training data**
- **20% testing data**

The split used:

- `random_state = 42`
- Stratified sampling

The following evaluation metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The model-training implementation is:

`analytics/src/train_models.py`

---

## 7. Target Leakage Prevention

During development, `days_since_last_order` was initially included as a model feature.

However, the target variable itself was created directly from:

`days_since_last_order > 60`

Including this same feature in the model would introduce target leakage and could produce artificially high model performance.

Therefore, `days_since_last_order` was removed from the final model feature set.

The final models use these nine features:

- `total_orders`
- `total_spend`
- `average_order_value`
- `total_quantity`
- `average_discount`
- `average_rating`
- `unique_categories`
- `unique_products`
- `customer_lifetime_days`

This provides a more meaningful evaluation of the remaining customer attributes.

---

## 8. Model Evaluation

The final leakage-free test-set results are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.67% | 83.33% | 41.67% | 55.56% | 94.62% |
| Random Forest | 90.00% | 75.00% | 75.00% | 75.00% | 95.49% |

### Logistic Regression

Logistic Regression achieved:

- Accuracy: **86.67%**
- Precision: **83.33%**
- Recall: **41.67%**
- F1-score: **55.56%**
- ROC-AUC: **94.62%**

### Random Forest

Random Forest achieved:

- Accuracy: **90.00%**
- Precision: **75.00%**
- Recall: **75.00%**
- F1-score: **75.00%**
- ROC-AUC: **95.49%**

The Random Forest produced higher recall and F1-score than Logistic Regression on this synthetic test dataset.

These results are based on synthetic data and should not be interpreted as production-level model performance.

---

## 9. Model Outputs

The model comparison results are stored in:

`analytics/data/ml/model_comparison.csv`

Confusion matrices are stored as:

- `analytics/data/ml/logistic_regression_confusion_matrix.png`
- `analytics/data/ml/random_forest_confusion_matrix.png`

Trained models are stored in:

`analytics/models/`

Files include:

- `logistic_regression.pkl`
- `random_forest.pkl`
- `standard_scaler.pkl`

---

## 10. Feature Importance

Random Forest feature importance is generated using:

`analytics/src/feature_importance.py`

The outputs are:

- `analytics/data/ml/feature_importance.csv`
- `analytics/data/ml/feature_importance.png`

### Final Feature Importance

| Feature | Importance |
|---|---:|
| `customer_lifetime_days` | 0.401655 |
| `total_spend` | 0.109338 |
| `total_quantity` | 0.101894 |
| `total_orders` | 0.077972 |
| `average_order_value` | 0.075442 |
| `average_rating` | 0.073600 |
| `average_discount` | 0.069562 |
| `unique_products` | 0.067641 |
| `unique_categories` | 0.022897 |

`customer_lifetime_days` has the highest feature-importance value in the trained Random Forest model.

Feature importance indicates how much the Random Forest relied on a feature when making predictions. It does not establish that a feature causes customer inactivity.

---

## 11. Reproducibility

Run the following commands from the project root.

### Generate synthetic customer-order data

```bash
python analytics/src/generate_customer_data.py