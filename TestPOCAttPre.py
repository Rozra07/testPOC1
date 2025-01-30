import pickle
import pandas as pd
import numpy as np
import streamlit as st

def preprocess_data(df, feature_columns):
    df["Days Since Last Promotion"] = df["Hasn't been promoted"].map({
        "1 year": 365, "1.5 years": 547, "2 years": 730, "2.5 years": 912,
        "3 years": 1095, "3.5 years": 1277, "4+ years": 1500
    })
    df["Age"] = df["Employee Age"]
    df["Tenure (Months)"] = df["Tenure (Months)"]
    
    categorical_cols = ["Pulse", "College Tier", "Industry Experience", "Company Type", "Gender"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure feature consistency
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

def predict_attrition(employee_data):
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)
    
    df_input = pd.DataFrame([employee_data])
    df_input = preprocess_data(df_input, feature_columns)
    probability = model.predict_proba(scaler.transform(df_input))[:, 1][0]
    return probability * 100

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
    
    st.write("Select College Tier (Does this tier have high retention in your company?):")
    college_tier = st.radio("", ["Tier 1", "Tier 2", "Tier 3"], horizontal=True)
    college_preference = st.slider("Likelihood of retention for this tier (%)", min_value=0, max_value=100, value=50)
    
    industry_experience = st.selectbox("Industry Experience (Does this industry background have high retention in your company?)", ["IT", "Finance", "Healthcare", "Manufacturing"])
    industry_preference = st.slider("Likelihood of retention for this industry (%)", min_value=0, max_value=100, value=50)
    
    st.write("Select Company Type (Does this company type have high retention in your company?):")
    company_type = st.radio("", ["MNC", "Startup", "Mid-Size", "Small"], horizontal=True)
    company_preference = st.slider("Likelihood of retention for this company type (%)", min_value=0, max_value=100, value=50)
    
    last_performance_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
    joining_ctc = st.number_input("Joining CTC (INR)", min_value=300000, max_value=2500000, value=1000000)
    compa_ratio = st.slider("Compa Ratio for the Role (%)", min_value=50, max_value=150, value=100)
    salary_increase = st.slider("Increase from Last Company (%)", min_value=0, max_value=50, value=10)
    submit_button = st.form_submit_button("Predict")

if submit_button:
    employee_data = {
        "Employee Age": employee_age,
        "Average Employee Age": avg_employee_age,
        "Gender": gender,
        "Female Employee Ratio": female_ratio,
        "Tenure (Months)": tenure,
        "Hasn't been promoted": last_promotion,
        "Minimum Promotion Cycle": min_promotion_cycle,
        "Pulse": pulse_category,
        "College Tier": college_tier,
        "College Tier Retention": college_preference,
        "Industry Experience": industry_experience,
        "Industry Retention": industry_preference,
        "Company Type": company_type,
        "Company Type Retention": company_preference,
        "Last Performance Rating": last_performance_rating,
        "No. of Promotion": num_promotions,
        "Joining CTC (INR)": joining_ctc,
        "Compa Ratio": compa_ratio,
        "Increase from last company": salary_increase
    }
    
    prediction = predict_attrition(employee_data)
    st.subheader("Prediction Result")
    st.write(f"Estimated Attrition Probability: {prediction:.2f}%")
    
    if prediction > 70:
        st.error("High Risk of Attrition")
    elif prediction > 40:
        st.warning("Medium Risk of Attrition")
    else:
        st.success("Low Risk of Attrition")
