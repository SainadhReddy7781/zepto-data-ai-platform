# Zepto Data AI Platform

An end-to-end data engineering and machine learning platform for e-commerce product and customer analytics.

## 🚀 Project Overview

The Zepto Data AI Platform collects e-commerce product data, cleans and stores the data in PostgreSQL, performs customer-level analytics and feature engineering, and uses machine learning to predict customer inactivity.

The project also includes a Support Assistant that allows users to ask questions about the project and customer analytics directly from the terminal.

## 🏗️ Architecture

```text
E-commerce Product Data
        │
        ▼
   Data Pipeline
        │
        ├── Data Collection
        ├── Data Cleaning
        └── PostgreSQL Storage
                │
                ▼
        Customer Analytics
                │
                ├── EDA
                ├── Feature Engineering
                └── ML Dataset
                        │
                        ▼
              Machine Learning
                        │
                ┌───────┴────────┐
                ▼                ▼
       Logistic Regression   Random Forest
                │                │
                └───────┬────────┘
                        ▼
                Model Evaluation
                        │
                        ▼
              Customer Inactivity
                   Prediction
                        │
                        ▼
              Support Assistant