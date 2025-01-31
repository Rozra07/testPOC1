import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

###############################################################################
# Step 1: Train and Save a Dummy Logistic Regression Model (unchanged logic)
###############################################################################
def train_and_save_model():
    """
    Trains a dummy Logistic Regression model on synthetic data, then saves
    the model, scaler, and feature columns. This is unchanged core logic.
    """

    # Optional: In real usage, you might replace this synthetic data
    # with actual HR data from a CSV or database
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

    # Synthetic binary target
    y = np.random.randint(0, 2, size=n_samples)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])
    feature_columns = df_encoded.columns

    # For demonstration, let's do a train/test split to check basic metrics
    X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Basic validation metrics
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Validation Metrics:")
    print(f"  Accuracy:  {acc:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall:    {rec:.2f}")
    print(f"  ROC AUC:   {roc_auc:.2f}")

    # Save model artifacts
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)


###############################################################################
# (Run training once; comment out if you want to avoid retraining each time)
###############################################################################
train_and_save_model()

###############################################################################
# Step 2: Rule-Based Scoring Logic (unchanged from earlier)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    """
    Computes a rule-based attrition score (0-100) for a single employee dict.
    Also returns triggers if return_triggers=True.
    """
    score = 0
    extreme_factors = 0
    triggers = []

    # Condition 1: Low Gender Diversity
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # Condition 2: Stagnant promotions
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")

    # Condition 3: Performance rating
    if employee["Last Performance Rating"] == 1:
        score += 25
        extreme_factors += 1
        triggers.append("Very low performance rating")
    elif employee["Last Performance Rating"] == 2:
        score += 15
        extreme_factors += 0.5
        triggers.append("Low performance rating")
    elif employee["Last Performance Rating"] == 5:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("Excellent performance rating")

    # Condition 4: Compensation
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")

    # Condition 5: Retention metrics
    if employee["College Tier Retention"] < 15:
        score += 15
        extreme_factors += 0.5
        triggers.append("Low college tier retention")

    if employee["Industry Retention"] < 15:
        score += 15
        extreme_factors += 0.5
        triggers.append("Low industry retention")

    if employee["Company Type Retention"] < 15:
        score += 15
        extreme_factors += 0.5
        triggers.append("Low company type retention")

    # Condition 6: Pulse
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")

    # Scale based on extreme_factors
    if extreme_factors == 2:
        score = min(100, score * 1.3)
    elif extreme_factors == 3:
        score = min(100, score * 1.6)
    elif extreme_factors >= 4:
        score = min(100, score * 2)

    final_score = min(100, max(0, score))

    if return_triggers:
        return final_score, triggers
    else:
        return final_score


###############################################################################
# Step 3: ML Prediction - Combines ML Probability + Rule-Based Score
###############################################################################
def predict_attrition(employee_data):
    """
    Loads the saved model, scaler, and feature columns, then:
      1. Encodes + scales the employee_data
      2. Predicts ML probability
      3. Computes rule-based score + triggers
      4. Combines them to get final risk
    Returns: (combined_score, triggers)
    """
    # Load model artifacts
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Prepare input for ML
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    # ML Probability
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100

    # Rule-based
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    # Combine
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers


###############################################################################
# Step 4A: Single-Employee Prediction UI
###############################################################################
def single_employee_ui():
    """
    Provides a Streamlit form for single-employee input,
    then displays the predicted attrition risk & triggers.
    """
    st.markdown("## Single Employee Attrition Prediction")
    with st.form("single_form"):
        employee_data = {
            "Employee Age": st.slider("Employee Age", 18, 65, 30),
            "Average Employee Age": st.slider("Average Employee Age", 18, 65, 35),
            "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
            "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
            "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
            "Pulse": st.radio("Employee dissatisfaction (Pulse)", ["High", "Medium", "Low"], horizontal=True),
            "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
            "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
            "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
            "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
            "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
            "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
            "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
        }

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        # Predict
        combined_score, triggers = predict_attrition(employee_data)

        # Show risk color-coded
        if combined_score >= 75:
            st.markdown(
                f'<div style="background-color:#ff4d4d; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
                f'⚠️ HIGH Attrition Risk! <br> {combined_score:.2f}% 🚨</div>',
                unsafe_allow_html=True
            )
        elif 60 <= combined_score < 75:
            st.markdown(
                f'<div style="background-color:#ff9933; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
                f'⚠️ Moderate to High Risk <br> {combined_score:.2f}% ⚡</div>',
                unsafe_allow_html=True
            )
        elif 35 <= combined_score < 60:
            st.markdown(
                f'<div style="background-color:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
                f'⚖️ Moderate Attrition Risk <br> {combined_score:.2f}% 📉</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#28a745; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
                f'✅ SAFE! Low Attrition Risk <br> {combined_score:.2f}% 🌱</div>',
                unsafe_allow_html=True
            )

        # Show triggers
        if triggers:
            st.write("### Key Contributing Factors:")
            for t in triggers:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No major negative triggers identified.*")


###############################################################################
# Step 4B: Batch/Multiple Employee Prediction UI
###############################################################################
def batch_prediction_ui():
    """
    Provides an interface for uploading a CSV/Excel file of employees,
    runs batch predictions, and displays aggregated results.
    """
    st.markdown("## Batch Prediction for Employee Attrition")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv","xlsx"])

    if uploaded_file:
        # Identify file type
        if uploaded_file.name.endswith(".csv"):
            df_batch = pd.read_csv(uploaded_file)
        else:
            df_batch = pd.read_excel(uploaded_file)

        st.write("**File Preview**:")
        st.dataframe(df_batch.head())

        # Check required columns
        required_cols = [
            "Employee Age", "Average Employee Age", "Gender", "Female Employee Ratio",
            "Tenure (Months)", "Pulse", "Hasn't been promoted", "Minimum Promotion Cycle",
            "College Tier Retention", "Industry Retention", "Company Type Retention",
            "Last Performance Rating", "Compa Ratio"
        ]
        missing_cols = [col for col in required_cols if col not in df_batch.columns]

        if missing_cols:
            st.error(f"Missing columns in the uploaded file: {missing_cols}")
            return

        if st.button("Run Batch Prediction"):
            scores = []
            triggers_list = []

            for idx, row in df_batch.iterrows():
                # Convert row to dict matching the keys used in 'predict_attrition'
                employee_dict = row.to_dict()

                # Run prediction
                combined_score, triggers = predict_attrition(employee_dict)

                scores.append(combined_score)
                # Convert triggers list to a comma-separated string
                triggers_str = ", ".join(triggers)
                triggers_list.append(triggers_str)

            # Add results to dataframe
            df_batch["Attrition Score"] = scores
            df_batch["Triggers"] = triggers_list

            st.write("## Batch Prediction Results")
            st.dataframe(df_batch)

            # Aggregate stats
            high_risk_count = (df_batch["Attrition Score"] >= 75).sum()
            moderate_risk_count = ((df_batch["Attrition Score"] >= 35) & (df_batch["Attrition Score"] < 75)).sum()
            low_risk_count = (df_batch["Attrition Score"] < 35).sum()

            st.write(f"**High Risk (>= 75)**: {high_risk_count}")
            st.write(f"**Moderate Risk (35–74)**: {moderate_risk_count}")
            st.write(f"**Low Risk (< 35)**: {low_risk_count}")

            # Trigger Frequency
            all_triggers_split = df_batch["Triggers"].str.split(", ")
            all_triggers_flat = [
                t for sublist in all_triggers_split
                if isinstance(sublist, list) for t in sublist
            ]
            if all_triggers_flat:
                trigger_counts = pd.Series(all_triggers_flat).value_counts()
                st.write("### Top Triggers in This Batch")
                trigger_counts_df = pd.DataFrame(trigger_counts, columns=["Count"])
                st.bar_chart(trigger_counts_df)
            else:
                st.info("No negative triggers found in the batch.")


###############################################################################
# Step 5: Streamlit Main Entry - Choose between Single or Batch
###############################################################################
def main():
    st.title("Employee Attrition Prediction System")

    mode = st.sidebar.selectbox("Select Mode", ["Single Employee", "Batch Prediction"])

    if mode == "Single Employee":
        single_employee_ui()
    else:
        batch_prediction_ui()


if __name__ == "__main__":
    main()
