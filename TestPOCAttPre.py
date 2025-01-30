import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

###############################################################################
# Step 1: (Optional) Train a dummy Logistic Regression and save as .pkl
#         In a real project, you already have a trained model. Just ensure
#         that file indeed contains a valid scikit‐learn estimator.
###############################################################################

RUN_TRAINING = True  # Set to False if you already have your .pkl files

def train_and_save_model():
    # For demonstration, generate dummy data with 500 samples and a few columns
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame({
        "Employee Age": np.random.randint(20, 60, size=n_samples),
        "Average Employee Age": np.random.randint(25, 50, size=n_samples),
        "Female Employee Ratio": np.random.randint(0, 100, size=n_samples),
        "Tenure (Months)": np.random.randint(0, 240, size=n_samples),
        "Hasn't been promoted": np.random.randint(0, 60, size=n_samples),  # months
        "Minimum Promotion Cycle": np.random.randint(12, 60, size=n_samples),  # months
        "College Tier Retention": np.random.randint(20, 80, size=n_samples),
        "Industry Retention": np.random.randint(20, 80, size=n_samples),
        "Company Type Retention": np.random.randint(20, 80, size=n_samples),
        "Last Performance Rating": np.random.randint(1, 6, size=n_samples),
        "No. of Promotion": np.random.randint(0, 3, size=n_samples),
        "Compa Ratio": np.random.randint(70, 120, size=n_samples),
        "Increase from last company": np.random.randint(0, 30, size=n_samples),
        "Joining CTC (INR)": np.random.randint(300000, 2500000, size=n_samples),
        # We'll treat Gender + Pulse as categorical
        # We'll one‐hot them or just encode them as numeric for the dummy data
        "Gender": np.random.choice(["Male", "Female"], size=n_samples),
        "Pulse": np.random.choice(["High", "Medium", "Low"], size=n_samples)
    })

    # For the target, let's say 1 means "attrition" and 0 means "no attrition"
    y = np.random.randint(0, 2, size=n_samples)

    # We must transform or one-hot encode categorical columns
    # For simplicity, let's do a minimal encoding
    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])

    # Keep track of the column order
    feature_columns = df_encoded.columns

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # Train a basic logistic regression
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    # Save model, scaler, and feature columns to .pkl
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)

if RUN_TRAINING:
    train_and_save_model()

###############################################################################
# Step 2: Rule-based scoring function
###############################################################################
def compute_weighted_attrition(employee):
    score = 0

    # Age Weighting (6%)
    age_diff = abs(employee["Employee Age"] - employee["Average Employee Age"])
    if age_diff >= 25:
        score += 6
    else:
        score += (age_diff / 25) * 6

    # Gender Weighting (9%) – only if gender is female
    # If "Gender" is a string, handle it. If we end up one‐hotting it, we adapt.
    if employee.get("Gender", "Male") == "Female":
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
        diff = employee["Hasn't been promoted"] - employee["Minimum Promotion Cycle"]
        score += (diff / employee["Minimum Promotion Cycle"]) * 10

    # Pulse Weighting (10%)
    # If "Pulse" is e.g. "High"/"Medium"/"Low", handle it:
    pulse_map = {"High": 10, "Medium": 5, "Low": 0}
    pulse_score = pulse_map.get(employee.get("Pulse", "Medium"), 5)
    score += pulse_score

    # College Tier Retention (5%)
    if employee["College Tier Retention"] >= 70:
        score += 0
    elif employee["College Tier Retention"] <= 30:
        score += 5
    else:
        numerator = (employee["College Tier Retention"] - 30)
        score += (1 - numerator / 40) * 5

    # Industry Retention (5%)
    if employee["Industry Retention"] >= 70:
        score += 0
    elif employee["Industry Retention"] <= 30:
        score += 5
    else:
        numerator = (employee["Industry Retention"] - 30)
        score += (1 - numerator / 40) * 5

    # Company Type Retention (5%)
    if employee["Company Type Retention"] >= 70:
        score += 0
    elif employee["Company Type Retention"] <= 30:
        score += 5
    else:
        numerator = (employee["Company Type Retention"] - 30)
        score += (1 - numerator / 40) * 5

    # Last Performance Rating (20%)
    performance_map = {1: 20, 2: 15, 3: 10, 4: 5, 5: 0}
    perf_score = performance_map.get(employee["Last Performance Rating"], 10)
    score += perf_score

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
        numerator = (employee["Compa Ratio"] - 80)
        score += (1 - numerator / 30) * 15

    # Salary Increase from Last Company (10%)
    if employee["Increase from last company"] >= 25:
        score += 0
    elif employee["Increase from last company"] <= 5:
        score += 10
    else:
        numerator = (employee["Increase from last company"] - 5)
        score += (1 - numerator / 20) * 10

    return min(100, max(0, score))

###############################################################################
# Step 3: Combined prediction (load model & scaler, run .predict_proba)
###############################################################################
def predict_attrition(employee_data):
    # Load model, scaler, and feature columns
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Convert the single dict into a 1-row DataFrame
    df_input = pd.DataFrame([employee_data])

    # One-hot encode columns that might appear in the model
    df_input = pd.get_dummies(df_input)

    # Make sure we have the same columns as the model expects
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # Scale with the same scaler
    X_scaled = scaler.transform(df_input)

    # Model’s predicted probability for class=1
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100

    # Rule-based score
    rule_probability = compute_weighted_attrition(employee_data)

    # Combine (60% ML + 40% rule)
    combined_score = 0.6 * ml_probability + 0.4 * rule_probability
    return combined_score

###############################################################################
# Step 4: Streamlit UI
###############################################################################
st.title("Employee Attrition Prediction Tool (Demo)")

with st.form("attrition_form"):
    # Collect user inputs
    employee_age = st.slider("Employee Age", 18, 65, 30)
    avg_employee_age = st.slider("Average Employee Age in Company", 18, 65, 35)
    gender = st.radio("Employee Gender", ["Male", "Female"], horizontal=True)
    female_ratio = st.slider("Female Employee Ratio (%)", 0, 100, 40)
    tenure = st.slider("Tenure (Months)", 0, 240, 36)
    hasnt_promoted = st.slider("Months Since Last Promotion", 0, 60, 12)
    min_promo_cycle = st.slider("Min Promotion Cycle (Months)", 12, 60, 24)
    pulse = st.radio("Pulse (Manager's View)", ["High", "Medium", "Low"], horizontal=True)
    college_retention = st.slider("College Tier Retention (%)", 20, 100, 60)
    industry_retention = st.slider("Industry Retention (%)", 20, 100, 60)
    company_retention = st.slider("Company Type Retention (%)", 20, 100, 60)
    last_perf_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", 0, 10, 1)
    compa_ratio = st.slider("Compa Ratio (%)", 50, 150, 100)
    salary_increase = st.slider("Increase from Last Company (%)", 0, 50, 10)
    joining_ctc = st.number_input("Joining CTC (INR)", 300000, 2500000, 1000000)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Construct the dictionary
    employee_data = {
        "Employee Age": employee_age,
        "Average Employee Age": avg_employee_age,
        "Gender": gender,  # e.g. "Male"/"Female"
        "Female Employee Ratio": female_ratio,
        "Tenure (Months)": tenure,
        "Hasn't been promoted": hasnt_promoted,
        "Minimum Promotion Cycle": min_promo_cycle,
        "Pulse": pulse,  # e.g. "High"/"Medium"/"Low"
        "College Tier Retention": college_retention,
        "Industry Retention": industry_retention,
        "Company Type Retention": company_retention,
        "Last Performance Rating": last_perf_rating,
        "No. of Promotion": num_promotions,
        "Compa Ratio": compa_ratio,
        "Increase from last company": salary_increase,
        "Joining CTC (INR)": joining_ctc
    }

    # Predict
    try:
        prediction = predict_attrition(employee_data)
        st.write(f"**Estimated Attrition Probability**: {prediction:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
