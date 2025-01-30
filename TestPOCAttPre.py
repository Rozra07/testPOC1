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
        score += (
            (employee["Hasn't been promoted"] - employee["Minimum Promotion Cycle"])
            / employee["Minimum Promotion Cycle"]
        ) * 10

    # Pulse Weighting (10%)
    pulse_weights = {"High": 10, "Medium": 5, "Low": 0}
    score += pulse_weights[employee["Pulse"]]

    # College Tier Retention (5%)
    if employee["College Tier Retention"] >= 70:
        score += 0
    elif employee["College Tier Retention"] <= 30:
        score += 5
    else:
        score += (1 - (employee["College Tier Retention"] - 30) / 40) * 5

    # Industry Retention (5%)
    if employee["Industry Retention"] >= 70:
        score += 0
    elif employee["Industry Retention"] <= 30:
        score += 5
    else:
        score += (1 - (employee["Industry Retention"] - 30) / 40) * 5

    # Company Type Retention (5%)
    if employee["Company Type Retention"] >= 70:
        score += 0
    elif employee["Company Type Retention"] <= 30:
        score += 5
    else:
        score += (1 - (employee["Company Type Retention"] - 30) / 40) * 5

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
        score += (1 - (employee["Compa Ratio"] - 80) / 30) * 15

    # Salary Increase from Last Company (10%)
    if employee["Increase from last company"] >= 25:
        score += 0
    elif employee["Increase from last company"] <= 5:
        score += 10
    else:
        score += (1 - (employee["Increase from last company"] - 5) / 20) * 10

    # Keep final score between 0 and 100
    return min(100, max(0, score))

def predict_attrition(employee_data):
    # Load the trained ML model, scaler, and feature columns
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)

    # Prepare dataframe according to training feature set
    df_input = pd.DataFrame([employee_data])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # Predict probabilities using the ML model
    ml_prediction = model.predict_proba(scaler.transform(df_input))[:, 1][0] * 100

    # Calculate rule-based score
    rule_based_prediction = compute_weighted_attrition(employee_data)

    # Combine (60% ML + 40% Rule-Based)
    return (0.6 * ml_prediction) + (0.4 * rule_based_prediction)

# Streamlit UI
st.title("Employee Attrition Prediction Tool")

# Collect user inputs
with st.form("attrition_form"):
    employee_age = st.slider("Employee Age", 18, 65, 30)
    avg_employee_age = st.slider("Average Employee Age in Company", 18, 65, 35)
    gender = st.radio("Employee Gender", ["Male", "Female"], horizontal=True)
    female_ratio = st.slider("Percentage of Female Employees in Company", 0, 100, 40)
    tenure = st.slider("How much time in company (Months)", 0, 240, 36)
    last_promotion_years = st.selectbox(
        "Hasn't been promoted for",
        ["1 year", "1.5 years", "2 years", "2.5 years", "3 years", "3.5 years", "4+ years"],
    )
    # Convert year strings to months for better numeric handling
    years_only = float(last_promotion_years.replace(" years", "").replace("+", ""))
    last_promotion_months = int(years_only * 12)

    min_promotion_cycle = st.slider("Minimum Recruitment Tenure for Promotion (Years)", 1, 10, 3)
    pulse_percent = st.slider("Chances of leaving according to manager (%)", 0, 100, 50)
    
    # Map pulse_percent to categories for rule-based weighting
    if pulse_percent >= 66:
        pulse = "High"
    elif pulse_percent >= 33:
        pulse = "Medium"
    else:
        pulse = "Low"

    # For demonstration, map college tier to some numeric retention value
    # (In practice, you'd have your own logic for these.)
    college_tier = st.radio("College Tier", ["Tier 1", "Tier 2", "Tier 3"], horizontal=True)
    # Simulated retentions for example
    tier_retention_map = {"Tier 1": 80, "Tier 2": 50, "Tier 3": 30}
    college_tier_retention = tier_retention_map[college_tier]

    industry_experience = st.selectbox("Industry Experience", ["IT", "Finance", "Healthcare", "Manufacturing"])
    # Simulated industry retention
    industry_retention_map = {"IT": 70, "Finance": 60, "Healthcare": 50, "Manufacturing": 40}
    industry_retention = industry_retention_map[industry_experience]

    company_type = st.radio("Company Type", ["MNC", "Startup", "Mid-Size", "Small"], horizontal=True)
    # Simulated company type retention
    company_type_retention_map = {"MNC": 80, "Startup": 35, "Mid-Size": 60, "Small": 50}
    company_type_retention = company_type_retention_map[company_type]

    last_performance_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", 0, 10, 1)
    joining_ctc = st.number_input("Joining CTC (INR)", 300000, 2500000, 1000000)
    compa_ratio = st.slider("Compa Ratio for the Role (%)", 50, 150, 100)
    salary_increase = st.slider("Increase from Last Company (%)", 0, 50, 10)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Prepare the employee data as a dictionary
    employee_data = {
        "Employee Age": employee_age,
        "Average Employee Age": avg_employee_age,
        "Gender": gender,
        "Female Employee Ratio": female_ratio,
        "Tenure (Months)": tenure,
        "Hasn't been promoted": last_promotion_months,
        "Minimum Promotion Cycle": min_promotion_cycle * 12,  # convert years to months
        "Pulse": pulse,
        "College Tier Retention": college_tier_retention,
        "Industry Retention": industry_retention,
        "Company Type Retention": company_type_retention,
        "Last Performance Rating": last_performance_rating,
        "No. of Promotion": num_promotions,
        "Compa Ratio": compa_ratio,
        "Increase from last company": salary_increase,
        # Include any additional numeric fields your model needs:
        # For example, "Joining CTC" if it's part of feature_columns
        "Joining CTC": joining_ctc,
    }

    # Get final prediction
    final_prediction = predict_attrition(employee_data)

    # Write the output
    st.write(f"**Predicted Attrition Probability (60% ML + 40% Factor Method):** {final_prediction:.2f}%")
