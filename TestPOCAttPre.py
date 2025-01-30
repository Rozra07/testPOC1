import pickle
import pandas as pd
import numpy as np
import streamlit as st

def preprocess_data(df, feature_columns):
    df["Age"] = pd.to_datetime("today").year - pd.to_datetime(df["DOB"], errors="coerce").dt.year
    df["Tenure (Years)"] = pd.to_datetime("today").year - pd.to_datetime(df["Joining Date"], errors="coerce").dt.year
    df["Days Since Last Promotion"] = (pd.to_datetime("today") - pd.to_datetime(df["Last Promotion Date"], errors="coerce")).dt.days
    
    categorical_cols = ["Pulse", "College Tier", "Industry Experience", "Company Type"]
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
    dob = st.date_input("Date of Birth")
    joining_date = st.date_input("Joining Date")
    last_promotion_date = st.date_input("Last Promotion Date")
    pulse = st.selectbox("Pulse", ["High", "Medium", "Low"])
    college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
    industry_experience = st.selectbox("Industry Experience", ["IT", "Finance", "Healthcare", "Manufacturing"])
    company_type = st.selectbox("Company Type", ["MNC", "Startup", "Mid-Size", "Small"])
    last_performance_rating = st.slider("Last Performance Rating", 1, 5, 3)
    num_promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
    joining_ctc = st.number_input("Joining CTC (INR)", min_value=300000, max_value=2500000, value=1000000)
    salary_increase = st.number_input("Increase from Last Company (%)", min_value=0, max_value=50, value=10)
    submit_button = st.form_submit_button("Predict")

if submit_button:
    employee_data = {
        "DOB": dob,
        "Joining Date": joining_date,
        "Last Promotion Date": last_promotion_date,
        "Pulse": pulse,
        "College Tier": college_tier,
        "Industry Experience": industry_experience,
        "Company Type": company_type,
        "Last Performance Rating": last_performance_rating,
        "No. of Promotion": num_promotions,
        "Joining CTC (INR)": joining_ctc,
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
