import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: AI-Enhanced Attrition Prediction with Risk Factor Identification
###############################################################################
def predict_attrition_with_risks(employee_data):
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

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability

    # Identify key risk factors
    risk_factors = []
    if employee_data["Female Employee Ratio"] <= 15:
        risk_factors.append("Low Gender Diversity")
    if employee_data["Hasn't been promoted"] >= 2 * employee_data["Minimum Promotion Cycle"]:
        risk_factors.append("Long Promotion Gap")
    if employee_data["Compa Ratio"] < 70:
        risk_factors.append("Low Compensation")
    if employee_data["College Tier Retention"] < 15:
        risk_factors.append("Low College Tier Retention")
    if employee_data["Industry Retention"] < 15:
        risk_factors.append("Low Industry Retention")
    if employee_data["Pulse"] == "Low":
        risk_factors.append("Low Employee Engagement")

    return combined_score, risk_factors

###############################################################################
# Step 2: AI-Driven Insights & Recommendations
###############################################################################
def generate_insights(risk_factors, industry):
    insights = []
    
    if "Low Gender Diversity" in risk_factors:
        if industry == "Manufacturing":
            insights.append("Manufacturing traditionally struggles with gender diversity due to workplace conditions. Consider implementing safety policies, flexible work hours, and leadership programs to retain female employees.")
        else:
            insights.append("Your company has low gender diversity. Research shows that improving diversity enhances innovation and employee retention. Consider diversity hiring programs and mentorship initiatives.")
    
    if "Long Promotion Gap" in risk_factors:
        insights.append("Employees who stay too long without promotion are more likely to leave. Consider introducing a structured promotion cycle or skill-based career progression.")
    
    if "Low Compensation" in risk_factors:
        insights.append("Low compensation compared to industry standards is a key driver of attrition. Conduct salary benchmarking and adjust pay scales to stay competitive.")
    
    if "Low Employee Engagement" in risk_factors:
        insights.append("Employees with low engagement are at higher risk of leaving. Investing in team-building, recognition programs, and career development can improve retention.")
    
    return insights

###############################################################################
# Step 3: Streamlit UI
###############################################################################
st.markdown("<h2 style='text-align: center;'>🌟 AI-Enhanced Attrition Prediction 🚀</h2>", unsafe_allow_html=True)

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
    industry = st.selectbox("Industry Type", ["Tech", "Finance", "Manufacturing", "Retail", "Healthcare", "Other"])
    submit_button = st.form_submit_button("🚀 Predict")

if submit_button:
    prediction, risk_factors = predict_attrition_with_risks(employee_data)
    
    if prediction >= 75:
        st.markdown(f'<div style="background-color:#ff4d4d; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">⚠️ HIGH Attrition Risk! <br> {prediction:.2f}% 🚨</div>', unsafe_allow_html=True)
    elif 60 <= prediction < 75:
        st.markdown(f'<div style="background-color:#ff9933; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">⚠️ Moderate to High Risk <br> {prediction:.2f}% ⚡</div>', unsafe_allow_html=True)
    elif 35 <= prediction < 60:
        st.markdown(f'<div style="background-color:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">⚖️ Moderate Attrition Risk <br> {prediction:.2f}% 📉</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color:#28a745; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">✅ SAFE! Low Attrition Risk <br> {prediction:.2f}% 🌱</div>', unsafe_allow_html=True)
    
    if risk_factors:
        st.markdown("### 🔍 Identified Risk Factors")
        for risk in risk_factors:
            st.markdown(f"- {risk}")
        
        st.markdown("### 💡 AI-Generated Insights & Solutions")
        insights = generate_insights(risk_factors, industry)
        for insight in insights:
            st.markdown(f"✅ {insight}")
