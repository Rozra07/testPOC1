import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

###############################################################################
# Helper Functions
###############################################################################
def parse_last_promotion_months(promotion_str):
    """
    Convert a string like '1 year', '1.5 years', '2 years' into an integer month value.
    Example: '1.5 years' -> 18 months
    """
    # Safely split on space; the first token should be the numeric portion:
    years = float(promotion_str.split()[0])
    return int(years * 12)

def map_pulse_to_category(pulse_percent):
    """
    Convert a pulse probability (0-100) into High/Medium/Low for the compute_weighted_attrition function.
    """
    if pulse_percent > 66:
        return "High"
    elif pulse_percent > 33:
        return "Medium"
    else:
        return "Low"

def map_college_tier_to_retention(tier):
    """
    Example mapping: You should adjust these to reflect
    your own definition of 'College Tier Retention' percentages.
    """
    mapping = {
        "Tier 1": 80,
        "Tier 2": 50,
        "Tier 3": 30
    }
    return mapping.get(tier, 50)  # Default fallback if not found

def map_industry_to_retention(ind):
    """
    Example mapping for 'Industry Retention'.
    Adjust as appropriate for your dataset/business logic.
    """
    mapping = {
        "IT": 65,
        "Finance": 55,
        "Healthcare": 45,
        "Manufacturing": 35
    }
    return mapping.get(ind, 50)

def map_company_type_to_retention(ct):
    """
    Example mapping for 'Company Type Retention'.
    Adjust as appropriate for your dataset/business logic.
    """
    mapping = {
        "MNC": 75,
        "Startup": 30,
        "Mid-Size": 50,
        "Small": 40
    }
    return mapping.get(ct, 50)

###############################################################################
# Rule-based score calculation
###############################################################################
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
    # (Comparing months for 'Hasn't been promoted' vs 'Minimum Promotion Cycle')
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 10
    elif employee["Hasn't been promoted"] <= employee["Minimum Promotion Cycle"]:
        score += 0
    else:
        score += ((employee["Hasn't been promoted"] - employee["Minimum Promotion Cycle"]) /
                  employee["Minimum Promotion Cycle"]) * 10

    # Pulse Weighting (10%)
    pulse_weights = {"High": 10, "Medium": 5, "Low": 0}
    score += pulse_weights.get(employee["Pulse"], 0)

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
    score += performance_weights.get(employee["Last Performance Rating"], 10)

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

    return min(100, max(0, score))  # Ensure score is between 0 and 100

###############################################################################
# ML Prediction Function
###############################################################################
def predict_attrition(employee_data):
    # Load your trained model, scaler, and feature columns
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)

    # Build a DataFrame with the correct columns (fill missing ones with 0)
    df_input = pd.DataFrame([employee_data])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # Compute the model prediction probability (class=1) in percentage
    ml_prediction = model.predict_proba(scaler.transform(df_input))[:, 1][0] * 100

    # Compute the rule-based (weighted) score
    rule_based_prediction = compute_weighted_attrition(employee_data)

    # Combine the predictions (60% ML, 40% rule-based)
    return (0.6 * ml_prediction) + (0.4 * rule_based_prediction)

###############################################################################
# Streamlit UI
###############################################################################
st.title("Employee Attrition Prediction Tool")

with st.form("attrition_form"):
    # Collect user inputs
    employee_age = st.slider("Employee Age", 18, 65, 30)
    avg_employee_age = st.slider("Average Employee Age in Company", 18, 65, 35)
    gender = st.radio("Employee Gender", ["Male", "Female"], horizontal=True)
    female_ratio = st.slider("Percentage of Female Employees in Company", 0, 100, 40)
    tenure = st.slider("How much time in company (Months)", 0, 240, 36)
    last_promotion = st.selectbox(
        "Hasn't been promoted for",
        ["1 year", "1.5 years", "2 years", "2.5 years", "3 years", "3.5 years", "4+ years"]
    )
    min_promotion_cycle = st.slider("Minimum Recruitment Tenure for Promotion (Years)", 1, 10, 3)
    pulse_percent = st.slider("Chances of leaving according to manager (%)", 0, 100, 50)
    college_tier = st.radio("College Tier", ["Tier 1", "Tier 2", "Tier 3"], horizontal=True)
    industry_experience = st.selectbox("Industry Experience", ["IT", "Finance", "Healthcare", "Manufacturing"])
    company_type = st.radio("Company Type", ["MNC", "Startup", "Mid-Size", "Small"], horizontal=True)
    last_performance_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", 0, 10, 1)
    joining_ctc = st.number_input("Joining CTC (INR)", 300000, 2500000, 1000000)
    compa_ratio = st.slider("Compa Ratio for the Role (%)", 50, 150, 100)
    salary_increase = st.slider("Increase from Last Company (%)", 0, 50, 10)
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    ############################################################################
    # Construct the employee_data dictionary with the exact keys expected
    ############################################################################
    employee_data = {
        # As required by compute_weighted_attrition:
        "Employee Age": employee_age,
        "Average Employee Age": avg_employee_age,
        "Gender": gender,
        "Female Employee Ratio": female_ratio,
        "Tenure (Months)": tenure,
        "Hasn't been promoted": parse_last_promotion_months(last_promotion),
        "Minimum Promotion Cycle": min_promotion_cycle * 12,  # in months
        "Pulse": map_pulse_to_category(pulse_percent),
        "College Tier Retention": map_college_tier_to_retention(college_tier),
        "Industry Retention": map_industry_to_retention(industry_experience),
        "Company Type Retention": map_company_type_to_retention(company_type),
        "Last Performance Rating": last_performance_rating,
        "No. of Promotion": num_promotions,
        "Compa Ratio": compa_ratio,
        "Increase from last company": salary_increase,

        # If your model expects these (for example):
        "Joining CTC (INR)": joining_ctc
    }

    # Run the prediction
    prediction = predict_attrition(employee_data)

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Estimated Attrition Probability: {prediction:.2f}%")
