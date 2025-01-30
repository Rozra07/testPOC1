print("✅ Model, Scaler, and Feature Columns saved successfully! Now use 'predict.py' to make predictions.")

# predict.py
import pickle
import pandas as pd
import numpy as np

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

print("✅ Prediction script ready! Load the trained model and call predict_attrition(employee_data)")
