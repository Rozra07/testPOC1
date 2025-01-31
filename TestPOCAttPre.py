import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Load Model and Preprocessing Objects
###############################################################################
def load_model():
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

###############################################################################
# Rule-Based Scoring with Empathetic Explanations
###############################################################################
def compute_weighted_attrition(employee):
    score = 0
    extreme_factors = 0
    reasons = []
    advice = []
    
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1  
        reasons.append("Low gender diversity may create an unsupportive environment for female employees.")
        advice.append("Consider implementing diversity and inclusion programs to support female employees.")
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        reasons.append("The employee has waited too long for a promotion, which may cause dissatisfaction.")
        advice.append("Ensure clear career progression and timely promotions for employee satisfaction.")
    if employee["Last Performance Rating"] == 1:
        score += 25
        extreme_factors += 1
        reasons.append("Poor performance ratings could indicate a lack of motivation or support.")
        advice.append("Implement coaching programs and regular feedback sessions to improve performance.")
    if employee["Last Performance Rating"] == 2:
        score += 15
        extreme_factors += 0.5
        reasons.append("An average performance rating may indicate the need for additional training or mentorship.")
        advice.append("Enhance skill development initiatives and mentorship programs.")
    if employee["Last Performance Rating"] == 5:
        score -= 15
        extreme_factors -= 0.5
        reasons.append("A high performance rating suggests strong engagement and motivation.")
        advice.append("Recognize and reward high performers to sustain motivation.")
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        reasons.append("Compensation may not be competitive, leading to higher attrition risk.")
        advice.append("Benchmark salaries with industry standards and offer competitive compensation packages.")
    if employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        reasons.append("A high compensation ratio suggests strong financial incentives are in place.")
        advice.append("Maintain competitive pay structures while ensuring internal equity.")
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        reasons.append("High employee dissatisfaction detected in surveys (Pulse).")
        advice.append("Address employee concerns through open discussions and engagement initiatives.")
    if employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        reasons.append("Employees report high satisfaction in surveys (Pulse).")
        advice.append("Continue fostering a positive work environment and employee engagement efforts.")
    
    if extreme_factors == 2:
        score = min(100, score * 1.3)
    if extreme_factors == 3:
        score = min(100, score * 1.6)
    if extreme_factors >= 4:
        score = min(100, score * 2)
    
    return min(100, max(0, score)), reasons, advice

###############################################################################
# Machine Learning Prediction
###############################################################################
def predict_attrition(employee_data):
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, reasons, advice = compute_weighted_attrition(employee_data)

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, reasons, advice

###############################################################################
# Streamlit UI
###############################################################################
st.title("📊 Employee Attrition Prediction Tool")
st.markdown("### Identify potential attrition risks and gain insights into employee retention.")

with st.form("attrition_form"):
    employee_data = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Avg Employee Age", 18, 65, 35),
        "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
        "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Pulse": st.radio("Employee dissatisfaction according to Pulse", ["High", "Medium", "Low"], horizontal=True),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }
    
    submit_button = st.form_submit_button("🚀 Predict")

if submit_button:
    prediction, reasons, advice = predict_attrition(employee_data)
    
    st.markdown("---")
    
    if prediction > 70:
        st.markdown(f"## ⚠️ High Attrition Risk: {prediction:.2f}%")
        st.write("This employee has a high likelihood of leaving the organization. Below are some potential reasons:")
        for reason in reasons:
            st.markdown(f"- {reason}")
        
        st.markdown("### 🛠️ Recommended Actions to Improve Organization")
        for tip in advice:
            st.markdown(f"- **{tip}**")
    else:
        st.markdown(f"## ✅ Low Attrition Risk: {prediction:.2f}%")
        st.write("This employee has a low likelihood of leaving the organization. Keep up the good work!")
