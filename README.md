# Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using classification models.

## Overview

Customer churn is when a customer stops using a company's service. This project builds a model that predicts which customers are likely to churn, allowing the company to take action before losing them.

## Dataset

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7043 customers, 21 features
- **Target:** `Churn` — whether the customer left (Yes/No)

## Project Steps

1. Exploratory Data Analysis (EDA)
2. Data cleaning and preprocessing
3. Feature encoding (Label Encoding + One-Hot Encoding)
4. Model training — Logistic Regression and Random Forest
5. Model evaluation — Accuracy, Confusion Matrix, AUC Score
6. Feature importance analysis

## Results

| Model | Accuracy | AUC Score |
|---|---|---|
| Logistic Regression | 79% | 0.8320 |
| Random Forest | 79% | 0.8194 |

**Best model:** Logistic Regression with AUC of 0.83

## Key Findings

- `TotalCharges`, `MonthlyCharges` and `tenure` are the strongest predictors of churn
- Customers on **month-to-month contracts** churn the most
- Customers on **two-year contracts** are the most loyal
- **Fiber optic** internet users and **electronic check** payment users are high-risk

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Kaggle Notebook

[View full notebook on Kaggle](https://www.kaggle.com/code/midnight1708/customer-churn-prediction)