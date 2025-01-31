import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: Train and Save a Dummy Logistic Regression Model
###############################################################################
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
        "Compa Ratio": np.random.randint(50, 120, size=n_samples),
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

train_and_save_model()

###############################################################################
# Step 2: Rule-Based Scoring with Extreme Case Adjustments
###############################################################################
def compute_weighted_attrition(employee):
    score = 0
    extreme_factors = 0

    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1  
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
    if employee["Last Performance Rating"] == 1:
        score += 25
        extreme_factors += 1
    if employee["Last Performance Rating"] == 2:
        score += 15
        extreme_factors += 0.5
    if employee["Last Performance Rating"] == 5:
        score -= 15
        extreme_factors -= 0.5
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
    if employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
    if employee["College Tier Retention"] < 15:
        score += 15
        extreme_factors += 0.5
    if employee["Industry Retention"] < 15:
        score += 15
        extreme_factors += 0.5
    if employee["Company Type Retention"] < 15:
        score += 15
        extreme_factors += 0.5
    if employee["Pulse"] == "High" :
        score += 20
        extreme_factors += 0.5
    if employee["Pulse"] == "Low" :
        score -= 20
        extreme_factors -= 0.5

    if extreme_factors == 2:
        score = min(100, score * 1.3)
    if extreme_factors == 3:
        score = min(100, score * 1.6)
    if extreme_factors >= 4:
        score = min(100, score * 2)

    return min(100, max(0, score))

###############################################################################
# Step 3: Machine Learning Prediction
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

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score

###############################################################################
# Step 4: Streamlit UI
###############################################################################
# Streamlit Title with Reduced Size
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>",
    unsafe_allow_html=True
)
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
    prediction = predict_attrition(employee_data)
    
    if prediction >=75:
        st.markdown(
            f'<div style="background-color:#ff4d4d; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ HIGH Attrition Risk! <br> {prediction:.2f}% 🚨</div>',
            unsafe_allow_html=True
        )
    elif 60 <= prediction < 75:
        st.markdown(
            f'<div style="background-color:#ff9933; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ Moderate to High Risk <br> {prediction:.2f}% ⚡</div>',
            unsafe_allow_html=True
        )
    elif 35 <= prediction < 60:
        st.markdown(
            f'<div style="background-color:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚖️ Moderate Attrition Risk <br> {prediction:.2f}% 📉</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color:#28a745; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'✅ SAFE! Low Attrition Risk <br> {prediction:.2f}% 🌱</div>',
            unsafe_allow_html=True
        )

