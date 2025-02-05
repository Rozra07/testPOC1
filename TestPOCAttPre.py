import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# NEW MODULE: FEATURE ANALYSIS, MODEL TRAINING, AND WEIGHTED SCORING
# =============================================================================

def analyze_features(training_df, attrition_col):
    """
    Analyzes each non-target column in training_df to produce a relevance score (in %)
    that indicates how much variation in attrition is explained by that column.
    For numeric columns, we use the absolute Pearson correlation.
    For categorical columns, we use the maximum difference in attrition rates.
    Returns a dict mapping column names to relevance percentages.
    """
    relevances = {}
    target = training_df[attrition_col]
    for col in training_df.columns:
        if col == attrition_col:
            continue
        if pd.api.types.is_numeric_dtype(training_df[col]):
            # Compute absolute correlation (multiplied by 100, capped at 100%)
            corr = training_df[col].corr(target)
            rel = min(100, abs(corr)*100) if pd.notnull(corr) else 0
        else:
            # For categorical columns, compute difference in attrition rates across groups.
            try:
                grp = training_df.groupby(col)[attrition_col].mean()
                if len(grp) > 1:
                    rel = min(100, (grp.max() - grp.min()) * 100)
                else:
                    rel = 0
            except Exception as e:
                rel = 0
        relevances[col] = round(rel, 2)
    return relevances

def train_model_new(training_df, attrition_col, selected_features, industry):
    """
    Trains a new model using only the selected features.
    Also saves the trained model, scaler, feature list, and the calculated feature relevances.
    """
    # Use only the selected features (make sure to drop any rows with missing values)
    training_df = training_df.dropna(subset=selected_features+[attrition_col])
    X = training_df[selected_features].copy()
    y = training_df[attrition_col]
    
    # One-hot encode categorical features.
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    
    # Scale features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Train the logistic regression model.
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model artifacts.
    model_filename = f"{industry}_model.pkl"
    scaler_filename = f"{industry}_scaler.pkl"
    features_filename = f"{industry}_feature_columns.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_filename, "wb") as f:
        pickle.dump(feature_columns, f)
    
    st.success("Model trained and saved successfully!")
    training_accuracy = model.score(X_scaled, y) * 100
    st.info(f"Training Accuracy (Confidence): {training_accuracy:.2f}%")
    
    # Save the selected features and their relevances in the user settings.
    user = st.session_state.user
    settings = user.get("settings", {})
    settings["selected_features"] = selected_features
    # Assume the relevance scores were already computed and stored in session_state.
    settings["feature_relevances"] = st.session_state.get("feature_relevances", {})
    user["settings"] = settings
    # (Also update persistent storage as in your original code.)
    users = load_users()
    users[user["email"]] = user
    save_users(users)
    save_user_event(user["email"], "training", {"action": "Model retrained", "industry": industry})
    
def load_model_new(industry):
    """
    Loads the trained model and associated artifacts.
    """
    model_filename = f"{industry}_model.pkl"
    scaler_filename = f"{industry}_scaler.pkl"
    features_filename = f"{industry}_feature_columns.pkl"
    if os.path.exists(model_filename) and os.path.exists(scaler_filename) and os.path.exists(features_filename):
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        with open(scaler_filename, "rb") as f:
            scaler = pickle.load(f)
        with open(features_filename, "rb") as f:
            feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    else:
        st.error("No trained model found for the selected industry. Please train your model in Train Mode first.")
        return None, None, None

def compute_weighted_attrition_new(employee, selected_features, feature_relevances):
    """
    Computes a weighted attrition risk score based on the employee's data and the selected features.
    For each selected feature that has known logic (taken from your original code ideas),
    compare the employee's value to a global benchmark (stored in st.session_state) and add to the score.
    The contribution from each feature is weighted by its relevance.
    
    Returns a tuple (score, extreme_factors, details) where details is a dict showing the contribution.
    """
    score = 0
    extreme_factors = 0
    details = {}  # For debugging, store contribution from each feature
    
    # Example logic for known features (you can customize or add more):
    if "Gender" in selected_features and "Female Employee Ratio" in employee:
        if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
            contrib = 30 * (feature_relevances.get("Gender", 50)/50)
            score += contrib
            extreme_factors += 1
            details["Gender"] = contrib
            
    if "Hasn't been promoted" in selected_features and "Minimum Promotion Cycle" in employee:
        if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
            contrib = 30 * (feature_relevances.get("Hasn't been promoted", 50)/50)
            score += contrib
            extreme_factors += 1
            details["Hasn't been promoted"] = contrib
            
    if "Last Performance Rating" in selected_features:
        if employee["Last Performance Rating"] == 1:
            contrib = 25 * (feature_relevances.get("Last Performance Rating", 50)/50)
            score += contrib
            extreme_factors += 1
            details["Last Performance Rating"] = contrib
        elif employee["Last Performance Rating"] == 2:
            contrib = 15 * (feature_relevances.get("Last Performance Rating", 50)/50)
            score += contrib
            extreme_factors += 0.5
            details["Last Performance Rating"] = contrib
        elif employee["Last Performance Rating"] == 5:
            contrib = -15 * (feature_relevances.get("Last Performance Rating", 50)/50)
            score += contrib
            extreme_factors -= 0.5
            details["Last Performance Rating"] = contrib

    if "Compa Ratio" in selected_features:
        if employee["Compa Ratio"] < 80:
            contrib = 20 * (feature_relevances.get("Compa Ratio", 50)/50)
            score += contrib
            extreme_factors += 0.8
            details["Compa Ratio"] = contrib
        elif employee["Compa Ratio"] < 70:
            contrib = 25 * (feature_relevances.get("Compa Ratio", 50)/50)
            score += contrib
            extreme_factors += 1
            details["Compa Ratio"] = contrib
        elif employee["Compa Ratio"] > 110:
            contrib = -15 * (feature_relevances.get("Compa Ratio", 50)/50)
            score += contrib
            extreme_factors -= 0.5
            details["Compa Ratio"] = contrib

    if "Employee Age" in selected_features and "global_avg_age" in st.session_state:
        # If employee age is significantly higher than the company average, add risk.
        avg_age = st.session_state.global_avg_age
        if employee["Employee Age"] > avg_age + 5:
            contrib = 10 * (feature_relevances.get("Employee Age", 50)/50)
            score += contrib
            extreme_factors += 0.5
            details["Employee Age"] = contrib

    # (You can add more logic for other features that you know about.)
    
    # Adjust the score based on the number of extreme factors
    if extreme_factors == 2:
        score = min(100, score * 1.3)
    elif extreme_factors == 3:
        score = min(100, score * 1.6)
    elif extreme_factors >= 4:
        score = min(100, score * 2)
        
    final_score = min(100, max(0, score))
    return final_score, extreme_factors, details

def predict_attrition_new(employee_data, industry, selected_features, feature_relevances):
    """
    Predicts attrition by combining the ML model prediction with the rule‐based weighted score.
    """
    model, scaler, feature_columns = load_model_new(industry)
    if model is None:
        return None, None, None, None
    # Prepare data for the ML model
    df_input = pd.DataFrame([employee_data])
    df_input = df_input[selected_features].copy()
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    
    # Compute the weighted score using our new function.
    rule_probability, extreme_factors, rule_details = compute_weighted_attrition_new(employee_data, selected_features, feature_relevances)
    
    # Combine the two scores (you can adjust the weighting as desired)
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    
    return combined_score, rule_details, ml_probability, extreme_factors

# =============================================================================
# UI SNIPPET: UPON UPLOADING TRAINING DATA (TRAIN MODE)
# =============================================================================
#
# In your Train Mode section you would do something like the following.
# (This snippet assumes that the user has already selected the industry.)
#
if st.session_state.get("main_mode", "Train Mode") == "Train Mode":
    st.header("Train Mode")
    selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
    with col2:
        st.markdown("### Training File Guide")
        st.markdown("""
        **Your training file must include:**
        - A **target column** (e.g., `Attrition` – use binary values 0/1).
        - Other feature columns (e.g., `Employee Age`, `Gender`, `Tenure (Months)`, etc.).
        """)
        st.download_button(
            label="Download Dummy Training File",
            data=generate_dummy_training_file(),
            file_name="dummy_training_file.csv",
            mime="text/csv"
        )
    target_column = st.text_input("Enter the name of the target column", value="Attrition")
    
    if uploaded_train is not None:
        try:
            if uploaded_train.name.endswith(".csv"):
                train_df = pd.read_csv(uploaded_train)
            else:
                train_df = pd.read_excel(uploaded_train)
            st.write("### Preview of Uploaded Training Data")
            st.dataframe(train_df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
        
        # ANALYZE features (excluding the attrition column)
        relevances = analyze_features(train_df, target_column)
        st.session_state.feature_relevances = relevances  # store for later use
        
        st.markdown("### Feature Relevance Summary")
        # Show a checkbox for each candidate feature along with its relevance %
        candidate_features = [col for col in train_df.columns if col != target_column]
        selected_features = []
        confidence_list = []
        for feat in candidate_features:
            # Display checkbox with relevance percentage in the label.
            if st.checkbox(f"{feat} (Relevance: {relevances.get(feat,0)}%)", key=f"feat_{feat}"):
                selected_features.append(feat)
                confidence_list.append(relevances.get(feat,0))
        
        if selected_features:
            model_confidence = np.mean(confidence_list)
            st.info(f"**Current Model Confidence:** {model_confidence:.2f}% (based on selected features)")
        else:
            st.info("Select at least one feature (in addition to the target) to train the model.")
        
        if st.button("Update Aggregated Data and Retrain Model") and selected_features:
            aggregated_df = update_aggregated_training_data(selected_train_industry, train_df)
            st.write("### Aggregated Training Data Preview")
            st.dataframe(aggregated_df.head())
            train_model_new(aggregated_df, target_column, selected_features, selected_train_industry)
            
# =============================================================================
# UI SNIPPET: PREDICTION IN TEST MODE (SINGLE EMPLOYEE)
# =============================================================================
#
# In Test Mode, retrieve the saved selected_features and feature_relevances from the user settings.
#
if st.session_state.get("main_mode", "Train Mode") == "Test Mode":
    st.header("Test Mode")
    # Use the same industry as used in training, or allow a selection.
    default_test_industry = st.session_state.get("train_industry", industry_options[0])
    selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=industry_options.index(default_test_industry) if default_test_industry in industry_options else 0, key="test_industry")
    
    # Retrieve saved feature settings from the user record (or session_state)
    user_settings = st.session_state.user.get("settings", {})
    selected_features = user_settings.get("selected_features", [])
    feature_relevances = user_settings.get("feature_relevances", {})
    
    st.markdown("### Enter Employee / Company Details")
    with st.form("attrition_form"):
        # (Provide input widgets for at least the features that were selected in training.
        #  For known features you may also require additional context, e.g., global settings.)
        input_data = {}
        for feat in selected_features:
            if feat in ["Employee Age", "Tenure (Months)", "Hasn't been promoted", "Minimum Promotion Cycle", "Last Performance Rating", "Compa Ratio"]:
                # Numeric input (customize ranges as needed)
                input_data[feat] = st.number_input(f"{feat}", value=30)
            else:
                # For other features assume text input
                input_data[feat] = st.text_input(f"{feat}")
        # Also include any global context needed by the rule‐based logic.
        if "Employee Age" in selected_features:
            st.session_state.global_avg_age = st.slider("Average Employee Age in Company", 18, 100, st.session_state.user.get("settings", {}).get("global_avg_age", 35))
        if "Gender" in selected_features:
            input_data["Gender"] = st.radio("Gender", ["Male", "Female"], horizontal=True)
            input_data["Female Employee Ratio"] = st.slider("Female Employee Ratio (%)", 0, 100, st.session_state.user.get("settings", {}).get("global_female_ratio", 40))
        submit_single = st.form_submit_button("🚀 Predict")
        
        if submit_single:
            combined_score, rule_details, ml_confidence, extreme_factors = predict_attrition_new(
                input_data, selected_test_industry, selected_features, feature_relevances
            )
            st.session_state.prediction_made = True
            st.session_state.score = combined_score
            st.session_state.rule_details = rule_details
            st.session_state.ml_confidence = ml_confidence
            st.session_state.extreme_factors = extreme_factors
            save_user_event(st.session_state.user["email"], "test_single", {"input_data": input_data, "result": combined_score})
            
    if st.session_state.get("prediction_made"):
        score = st.session_state.score
        st.markdown("---")
        with st.container():
            if score >= 75:
                bg_color = "#ff4d4d"
                msg_html = f"⚠️ HIGH Attrition Risk <br> {score:.2f}% 🚨"
            elif score >= 60:
                bg_color = "#ff9933"
                msg_html = f"⚠️ Moderate to High Risk <br> {score:.2f}% ⚡"
            elif score >= 35:
                bg_color = "#ffd700"
                msg_html = f"⚖️ Moderate Attrition Risk <br> {score:.2f}% 📉"
            else:
                bg_color = "#28a745"
                msg_html = f"✅ SAFE! Low Attrition Risk <br> {score:.2f}% 🌱"
            st.markdown(
                f"""
                <div style="background-color:{bg_color}; color:white; padding:15px; border-radius:10px; 
                            text-align:center; font-size:24px; font-weight:bold;">
                    {msg_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"**Model Confidence:** {st.session_state.ml_confidence:.2f}%")
            st.markdown(f"**Extreme Factors:** {st.session_state.extreme_factors}")
            st.markdown("**Rule-Based Contributions:**")
            st.json(st.session_state.rule_details)
