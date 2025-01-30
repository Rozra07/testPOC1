import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

def compute_weighted_attrition(employee):
    score = 0

    # Age Weighting (6%)
    age_diff = abs(employee["Employee Age"] - employee["Average Employee Age"])
    if age_diff >= 25:
        score += 6
    else:
        score += (age_diff / 25) * 6

    # Gender Weighting (9%)
    if employee["Gender"] == "Female":
        if employee["Female Employee Ratio"] < 25:
            score += 9
        elif employee["Female Employee Ratio"] >= 50:
            score += 0
        else:
            score += (1 - (employee["Female Employee Ratio"] - 25) / 25) * 9

    # Tenure Weighting (6%)
    if employee["Tenure (Months)"] < 10:
        score += 0
    elif employee["Tenure (Months)"] < 24:
        score += 3
    else:
        score += 6

    # Last Promotion Weighting (10%)
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 10
    elif employee["Hasn't been promoted"] <= employee["Minimum Promotion Cycle"]:
        score += 0
    else:
        score += ((employee["Hasn't been promoted"] - employee["Minimum Promotion Cycle"]) / 
                  employee["Minimum Promotion Cycle"]) * 10

    # Pulse Weighting (10%)
    pulse_weights = {"High": 10, "Medium": 5, "Low": 0}
    score += pulse_weights[employee["Pulse"]]

    # College Tier Retention (5%)
    if employee["College Tier Retention"] >= 70:
        score += 0
    elif employee["College Tier Retention"] <= 30:
        score += 5
    else:
        score += ((1 - (employee["College Tier Retention"] - 30) / 40) * 5)

    # Industry Retention (5%)
    if employee["Industry Retention"] >= 70:
        score += 0
    elif employee["Industry Retention"] <= 30:
        score += 5
    else:
        score += ((1 - (employee["Industry Retention"] - 30) / 40) * 5)

    # Company Type Retention (5%)
    if employee["Company Type Retention"] >= 70:
        score += 0
    elif employee["Company Type Retention"] <= 30:
        score += 5
    else:
        score += ((1 - (employee["Company Type Retention"] - 30) / 40) * 5)

    # Last Performance Rating (20%)
    performance_weights = {1: 20, 2: 15, 3: 10, 4: 5, 5: 0}
    score += performance_weights[employee["Last Performance Rating"]]

    # Number of Promotions (10%)
    if employee["No. of Promotion"] >= 2:
        score += 0
    elif employee["No. of Promotion"] == 1:
        score += 5
    else:
        score += 10

    # Joining CTC and Compa Ratio (15%)
    if employee["Compa Ratio"] >= 110:
        score += 0
    elif employee["Compa Ratio"] <= 80:
        score += 15
    else:
        score += ((1 - (employee["Compa Ratio"] - 80) / 30) * 15)

    # Salary Increase from Last Company (10%)
    if employee["Increase from last company"] >= 25:
        score += 0
    elif employee["Increase from last company"] <= 5:
        score += 10
    else:
        score += ((1 - (employee["Increase from last company"] - 5) / 20) * 10)

    return min(100, max(0, score))  # Ensure score remains between 0-100

def predict_attrition(employee_data):
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)

    df_input = pd.DataFrame([employee_data])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    ml_prediction = model.predict_proba(scaler.transform(df_input))[:, 1][0] * 100
    rule_based_prediction = compute_weighted_attrition(employee_data)
    return (0.6 * ml_prediction) + (0.4 * rule_based_prediction)

# Streamlit UI for user input
st.title("Employee Attrition Prediction Tool")

# Collect user inputs
with st.form("attrition_form"):
    employee_age = st.slider("Employee Age", min_value=18, max_value=65, value=30)
    avg_employee_age = st.slider("Average Employee Age in Company", min_value=18, max_value=65, value=35)
    gender = st.radio("Employee Gender", ["Male", "Female"], horizontal=True)
    female_ratio = st.slider("Percentage of Female Employees in Company", min_value=0, max_value=100, value=40)
    
    tenure = st.slider("How much time in company (Months)", min_value=0, max_value=240, value=36)
    last_promotion = st.selectbox("Hasn't been promoted for", ["1 year", "1.5 years", "2 years", "2.5 years", "3 years", "3.5 years", "4+ years"])
    min_promotion_cycle = st.slider("Minimum Recruitment Tenure for Promotion (Years)", min_value=1, max_value=10, value=3)
    
    pulse = st.slider("Chances of leaving according to manager (%)", min_value=0, max_value=100, value=50)
    pulse_category = "Low" if pulse < 30 else "Medium" if pulse < 70 else "High"
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    employee_data["Pulse"] = pulse_category
    prediction = predict_attrition(employee_data)
    st.subheader("Prediction Result")
    st.write(f"Estimated Attrition Probability: {prediction:.2f}%")
    
    if prediction > 70:
        st.error("High Risk of Attrition")
    elif prediction > 40:
        st.warning("Medium Risk of Attrition")
    else:
        st.success("Low Risk of Attrition")
