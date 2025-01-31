import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

###############################################################################
# Step 1: Train a Dummy Logistic Regression Model (or Use Existing .pkl)
###############################################################################
RUN_TRAINING = True  # Set to False if .pkl files already exist

def train_and_save_model():
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame({
        "Employee Age": np.random.randint(20, 60, size=n_samples),
        "Average Employee Age": np.random.randint(25, 50, size=n_samples),
        "Female Employee Ratio": np.random.randint(0, 100, size=n_samples),
        "Tenure (Months)": np.random.randint(0, 240, size=n_samples),
        "Hasn't been promoted": np.random.randint(0, 60, size=n_samples),
        "Minimum Promotion Cycle": np.random.randint(12, 60, size=n_samples),
        "College Tier Retention": np.random.randint(10, 80, size=n_samples),
        "Industry Retention": np.random.randint(10, 80, size=n_samples),
        "Company Type Retention": np.random.randint(10, 80, size=n_samples),
        "Last Performance Rating": np.random.randint(1, 6, size=n_samples),
        "No. of Promotion": np.random.randint(0, 3, size=n_samples),
        "Compa Ratio": np.random.randint(50, 120, size=n_samples),
        "Increase from last company": np.random.randint(0, 30, size=n_samples),
        "Joining CTC (INR)": np.random.randint(300000, 2500000, size=n_samples),
        "Gender": np.random.choice(["Male", "Female"], size=n_samples),
        "Pulse": np.random.choice(["High", "Medium", "Low"], size=n_samples)
    })

    y = np.random.randint(0, 2, size=n_samples)

    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])
    feature_columns = df_encoded.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)

if RUN_TRAINING:
    train_and_save_model()

###############################################################################
# Step 2: Rule-Based Scoring with Extreme Case Adjustments
###############################################################################
def compute_weighted_attrition(employee):
    score = 0

    # Age Weighting (6%)
    age_diff = abs(employee["Employee Age"] - employee["Average Employee Age"])
    score += 6 if age_diff >= 25 else (age_diff / 25) * 6

    # Extreme Case: Female in Male-Dominated Workplace (<10% Female Ratio)
    if employee.get("Gender", "Male") == "Female" and employee["Female Employee Ratio"] < 10:
        score += 12

    # Tenure Weighting (6%)
    score += 6 if employee["Tenure (Months)"] >= 24 else (3 if employee["Tenure (Months)"] >= 10 else 0)

    # Extreme Case: Promotion Delay (Now 25%)
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 25  
    elif employee["Hasn't been promoted"] <= employee["Minimum Promotion Cycle"]:
        score += 0
    else:
        diff = employee["Hasn't been promoted"] - employee["Minimum Promotion Cycle"]
        score += (diff / employee["Minimum Promotion Cycle"]) * 10

    # Extreme Case: Last Performance Rating = 1 (30%)
    performance_map = {1: 30, 2: 20, 3: 10, 4: 5, 5: 0}
    score += performance_map.get(employee["Last Performance Rating"], 10)

    # Extreme Case: Compa Ratio <70% (Now 30%)
    if employee["Compa Ratio"] < 70:
        score += 30  
    elif employee["Compa Ratio"] >= 110:
        score += 0
    else:
        score += (1 - (employee["Compa Ratio"] - 80) / 30) * 15

    # Extreme Cases for Low Retention Rates
    if employee["College Tier Retention"] < 20:
        score += 10
    if employee["Industry Retention"] < 20:
        score += 10
    if employee["Company Type Retention"] < 20:
        score += 10

    return min(100, max(0, score))

###############################################################################
# Step 3: Machine Learning Prediction (Logistic Regression)
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

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability = compute_weighted_attrition(employee_data)

    combined_score = 0.6 * ml_probability + 0.4 * rule_probability
    return combined_score

###############################################################################
# Step 4: Streamlit UI (All Inputs Included)
###############################################################################
st.title("Employee Attrition Prediction Tool")

with st.form("attrition_form"):
    employee_age = st.slider("Employee Age", 18, 65, 30)
    avg_employee_age = st.slider("Avg Employee Age", 18, 65, 35)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    female_ratio = st.slider("Female Employee Ratio (%)", 0, 100, 40)
    tenure = st.slider("Tenure (Months)", 0, 240, 36)
    hasnt_promoted = st.slider("Months Since Last Promotion", 0, 60, 12)
    min_promo_cycle = st.slider("Min Promotion Cycle (Months)", 12, 60, 24)
    pulse = st.radio("Pulse", ["High", "Medium", "Low"], horizontal=True)
    college_retention = st.slider("College Tier Retention (%)", 10, 100, 60)
    industry_retention = st.slider("Industry Retention (%)", 10, 100, 60)
    company_retention = st.slider("Company Type Retention (%)", 10, 100, 60)
    last_perf_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", 0, 10, 1)
    compa_ratio = st.slider("Compa Ratio (%)", 50, 150, 100)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    employee_data = {
        "Employee Age": employee_age,
        "Average Employee Age": avg_employee_age,
        "Gender": gender,
        "Female Employee Ratio": female_ratio,
        "Tenure (Months)": tenure,
        "Hasn't been promoted": hasnt_promoted,
        "Minimum Promotion Cycle": min_promo_cycle,
        "Pulse": pulse,
        "College Tier Retention": college_retention,
        "Industry Retention": industry_retention,
        "Company Type Retention": company_retention,
        "Last Performance Rating": last_perf_rating,
        "No. of Promotion": num_promotions,
        "Compa Ratio": compa_ratio
    }

    prediction = predict_attrition(employee_data)
    st.write(f"**Estimated Attrition Probability**: {prediction:.2f}%")
