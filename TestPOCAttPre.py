import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Load Machine Learning Model and Scaler
###############################################################################
def predict_attrition(employee_data):
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    # Debugging Logs
    print("Model type:", type(model))
    print("Expected Features:", feature_columns)
    print("Input Data Columns:", df_input.columns)

    # Check if predict_proba exists
    if hasattr(model, "predict_proba"):
        ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    else:
        ml_probability = model.predict(X_scaled)[0] * 100  # Fallback

    rule_probability = compute_weighted_attrition(employee_data)

    # Override ML Probability for Extreme Cases
    if rule_probability >= 80:
        ml_probability = max(ml_probability, rule_probability)

    # Adjust ML weight to 30% and Rule-Based to 70%
    combined_score = (0.3 * ml_probability) + (0.7 * rule_probability)

    return min(100, combined_score)

###############################################################################
# Rule-Based Scoring with Enhanced Compounding Effect
###############################################################################
def compute_weighted_attrition(employee):
    score = 0
    extreme_factors = 0  # Counter for extreme attrition cases

    # Age Weighting (6%) - Normalizing the impact
    age_diff = abs(employee["Employee Age"] - employee["Average Employee Age"])
    score += (age_diff / 25) * 6 if age_diff >= 5 else 0

    # Extreme Case: Female in Male-Dominated Workplace (<10% Female Ratio)
    if employee.get("Gender", "Male") == "Female" and employee["Female Employee Ratio"] < 10:
        score += 18  # Increased from 15 → 18 for stronger impact
        extreme_factors += 1

    # Tenure Weighting (6%) - Adjusted to reduce low-tenure bias
    score += 6 if employee["Tenure (Months)"] >= 36 else (3 if employee["Tenure (Months)"] >= 12 else 0)

    # Extreme Case: Promotion Delay (Now 35%) - Stronger weight
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 35
        extreme_factors += 1

    # Extreme Case: Last Performance Rating = 1 (40%) - Stronger impact
    performance_map = {1: 40, 2: 25, 3: 15, 4: 5, 5: 0}
    score += performance_map.get(employee["Last Performance Rating"], 5)
    if employee["Last Performance Rating"] == 1:
        extreme_factors += 1

    # Extreme Case: Compa Ratio <70% (Now 40%) - Stronger impact
    if employee["Compa Ratio"] < 70:
        score += 40
        extreme_factors += 1

    # Extreme Cases for Low Retention Rates
    if employee["College Tier Retention"] < 15:
        score += 15
        extreme_factors += 1
    if employee["Industry Retention"] < 15:
        score += 15
        extreme_factors += 1
    if employee["Company Type Retention"] < 15:
        score += 15
        extreme_factors += 1

    # ✅ Apply Stronger Non-Linear Compounding Effect for 3+ Extreme Factors
    if extreme_factors >= 3:
        multiplier = 1.6 if extreme_factors == 3 else (1.8 if extreme_factors == 4 else 2.3)
        score = min(100, score * multiplier)

    return min(100, score)

###############################################################################
# Streamlit UI for User Inputs
###############################################################################
st.title("Employee Attrition Prediction Tool")

with st.form("attrition_form"):
    employee_data = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Avg Employee Age", 18, 65, 35),
        "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
        "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "No. of Promotion": st.number_input("Number of Promotions", 0, 10, 1),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }

    submit_button = st.form_submit_button("Predict")

if submit_button:
    prediction = predict_attrition(employee_data)
    st.write(f"**Estimated Attrition Probability**: {prediction:.2f}%")
