import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer will churn based on their details.")

@st.cache_data
def load_and_train():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

model, feature_cols = load_and_train()

st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges = monthly_charges * tenure

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

if st.button("Predict Churn"):
    input_dict = {col: 0 for col in feature_cols}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior == "Yes" else 0
    input_dict['Partner'] = 1 if partner == "Yes" else 0
    input_dict['Dependents'] = 1 if dependents == "Yes" else 0
    input_dict['PaperlessBilling'] = 1 if paperless == "Yes" else 0
    input_dict['PhoneService'] = 1

    if contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1

    if internet == "Fiber optic":
        input_dict['InternetService_Fiber optic'] = 1
    elif internet == "No":
        input_dict['InternetService_No'] = 1

    if payment == "Credit card (automatic)":
        input_dict['PaymentMethod_Credit card (automatic)'] = 1
    elif payment == "Electronic check":
        input_dict['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check":
        input_dict['PaymentMethod_Mailed check'] = 1

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN — {probability*100:.1f}% probability")
    else:
        st.success(f"✅ This customer is likely to STAY — {(1-probability)*100:.1f}% probability of staying")

    st.markdown(f"**Churn Probability:** `{probability*100:.1f}%`")