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
import altair as alt
import matplotlib.pyplot as plt

# -----------------------------
# Define and enable a dark Altair theme
# -----------------------------
def dark_theme():
    return {
        "config": {
            "background": "black",
            "view": {"fill": "black"},
            "title": {"color": "white"},
            "axis": {
                "domainColor": "white",
                "gridColor": "#444444",
                "labelColor": "white",
                "titleColor": "white"
            },
            "legend": {"labelColor": "white", "titleColor": "white"}
        }
    }
alt.themes.register("dark_theme", dark_theme)
alt.themes.enable("dark_theme")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(layout="wide")

# ---------------------------------------
# Helper function for safe rerun
# ---------------------------------------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Refresh functionality is not available. Please update Streamlit (>=0.65.0).")

# ----------------------------------------------------
# Initialize st.session_state keys if not already set
# ----------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "nav" not in st.session_state:
    st.session_state.nav = "Tabs"  # "Tabs" indicates main UI (not My Account)
if "user" not in st.session_state:
    st.session_state.user = {}
if "bulk_prediction_complete" not in st.session_state:
    st.session_state.bulk_prediction_complete = False
if "bulk_result" not in st.session_state:
    st.session_state.bulk_result = None
if "enable_what_if" not in st.session_state:
    st.session_state.enable_what_if = False
if "custom_charts" not in st.session_state:
    st.session_state.custom_charts = []  # list to store custom charts

# ----------------------------------------------------
# Helper functions for user storage
# ----------------------------------------------------
USERS_FILE = "users.json"
USER_DATA_DIR = "user_data"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def save_user_event(email, event_type, event_data):
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR)
    history_file = os.path.join(USER_DATA_DIR, f"{email}_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "event_data": event_data
    }
    history.append(event)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

def load_user_history(email):
    history_file = os.path.join(USER_DATA_DIR, f"{email}_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    else:
        return []

# ----------------------------------------------------
# Global: Expanded Industry Options
# ----------------------------------------------------
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]

# ----------------------------------------------------
# Functions for model training/prediction
# ----------------------------------------------------
def train_model(training_df, target_column, industry):
    st.write("Training on data shape:", training_df.shape)
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)
    st.write("Model coefficients:", model.coef_)
    
    # -------------------------------
    # Model Evaluation Metrics
    # -------------------------------
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    preds = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, model.predict(X_scaled))
    report = classification_report(y, model.predict(X_scaled), output_dict=True)
    
    st.subheader("Model Evaluation Metrics")
    st.write(f"**ROC AUC:** {roc_auc:.2f}")
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="best")
    st.pyplot(fig)
    
    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    
    st.write("**Classification Report:**")
    st.json(report)
    
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
    
    update_industry_record(industry, model_filename, scaler_filename, features_filename)
    
    # Save global settings to user record
    user = st.session_state.user
    user_settings = user.get("settings") or {}
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3
    user_settings["bulk_industry_retention"] = {
        ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind=="Tech" else 50)
        for ind in industry_options
    }
    user_settings["bulk_company_retention"] = {
        "Startup": st.session_state.bulk_startup,
        "Small Size": st.session_state.bulk_small,
        "Mid Size": st.session_state.bulk_mid,
        "MNC/Giant Company": st.session_state.bulk_mnc
    }
    user["settings"] = user_settings
    users = load_users()
    users[user["email"]] = user
    save_users(users)
    save_user_event(user["email"], "training", {"action": "Model retrained", "industry": industry})

def update_industry_record(industry, model_file, scaler_file, feature_file):
    record = {
        "Industry": industry,
        "Model_File": model_file,
        "Scaler_File": scaler_file,
        "Feature_File": feature_file,
        "Training_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    csv_filename = "industry_models.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if industry in df["Industry"].values:
            df.loc[df["Industry"] == industry, ["Model_File", "Scaler_File", "Feature_File", "Training_Date"]] = \
                [model_file, scaler_file, feature_file, record["Training_Date"]]
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(csv_filename, index=False)

def load_model(industry):
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

# ----------------------------------------------------
# Trigger Details (for recommended solutions)
# ----------------------------------------------------
TRIGGER_DETAILS = {
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "Not enough female applicants are applying.",
            "lack_female_mentors": "There are few female mentors or leaders available.",
            "rigid_policies": "The policies are too rigid (e.g., no maternity or remote options)."
        },
        "solutions": {
            "lack_female_applicants": "Partner with women’s universities or female‑oriented professional groups and emphasize diversity in recruitment.",
            "lack_female_mentors": "Implement formal mentorship programs and sponsor leadership development for female employees.",
            "rigid_policies": "Introduce flexible working hours, remote/hybrid work options, and enhance family‑friendly benefits."
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are not transparent.",
            "no_mentorship": "There is a lack of mentorship or upskilling tracks.",
            "bureaucratic_structure": "The organizational structure is overly bureaucratic."
        },
        "solutions": {
            "unclear_criteria": "Publish clear promotion guidelines with KPIs and provide regular feedback.",
            "no_mentorship": "Launch mentoring programs and provide upskilling opportunities.",
            "bureaucratic_structure": "Streamline decision‑making processes or reduce hierarchical layers to foster agility."
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job roles or expectations are unclear or mismatched.",
            "no_feedback": "There is a lack of continuous feedback or one‑on‑one sessions.",
            "skill_gaps": "Training needs are not being addressed."
        },
        "solutions": {
            "misaligned_role": "Clarify job responsibilities, set SMART goals, and align roles with employees’ strengths.",
            "no_feedback": "Implement frequent one‑on‑one check‑ins and real‑time performance dashboards.",
            "skill_gaps": "Offer targeted training, certification reimbursements, and peer‑to‑peer learning opportunities."
        }
    },
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Job roles or expectations are unclear or mismatched.",
            "no_feedback": "Continuous feedback is lacking.",
            "skill_gaps": "Training needs are not addressed."
        },
        "solutions": {
            "misaligned_role": "Clarify job responsibilities and ensure roles align with employees’ strengths.",
            "no_feedback": "Implement regular one‑on‑one check‑ins and provide ongoing coaching.",
            "skill_gaps": "Offer targeted training sessions and promote cross‑functional learning."
        }
    },
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary is below market rates.",
            "minimal_bonus": "Bonuses or variable pay are minimal or nonexistent.",
            "poor_benefits": "The benefits package is insufficient."
        },
        "solutions": {
            "below_market": "Conduct market benchmarking to adjust salaries to at least median levels.",
            "minimal_bonus": "Introduce performance‑based incentives or profit‑sharing schemes.",
            "poor_benefits": "Offer competitive benefits including health insurance and retirement plans."
        }
    }
}

# ----------------------------------------------------
# Rule-based attrition computation
# ----------------------------------------------------
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []
    
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30; extreme_factors += 1; triggers.append("Low gender diversity")
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30; extreme_factors += 1; triggers.append("Stagnant promotions")
    if employee["Last Performance Rating"] == 1:
        score += 25; extreme_factors += 1; triggers.append("Very low performance rating")
    elif employee["Last Performance Rating"] == 2:
        score += 15; extreme_factors += 0.5; triggers.append("Low performance rating")
    elif employee["Last Performance Rating"] == 5:
        score -= 15; extreme_factors -= 0.5; triggers.append("Excellent performance rating")
    if employee["Compa Ratio"] < 80:
        score += 20; extreme_factors += 0.8; triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] < 70:
        score += 25; extreme_factors += 1; triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15; extreme_factors -= 0.5; triggers.append("High compensation ratio")
    if employee["College Tier Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low college tier retention")
    if employee["Industry Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low industry retention")
    if employee["Company Type Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low company type retention")
    if employee["Pulse"] == "High":
        score += 20; extreme_factors += 0.5; triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20; extreme_factors -= 0.5; triggers.append("Low dissatisfaction (Pulse)")
    
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

# ----------------------------------------------------
# Predict attrition using both ML and rule-based approaches
# ----------------------------------------------------
def predict_attrition(employee_data, industry):
    model, scaler, feature_columns = load_model(industry)
    if model is None:
        return None, None, None
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)
    combined_score = 0.5 * rule_probability + 0.5 * ml_probability
    return combined_score, triggers, ml_probability

def generate_sample_csv():
    sample_csv = pd.DataFrame({
        "Employee Age": [30, 45],
        "Gender": ["Male", "Female"],
        "Tenure (Months)": [36, 48],
        "Pulse": ["Medium", "High"],
        "Hasn't been promoted": [12, 36],
        "Minimum Promotion Cycle": [24, 24],
        "College Tier": ["Tier 1", "Tier 2"],
        "Industry": ["Tech", "Finance"],
        "Company Type": ["Startup", "Enterprise"],
        "Last Performance Rating": [3, 1],
        "Compa Ratio": [90, 65]
    })
    csv_buffer = io.StringIO()
    sample_csv.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_dummy_training_file():
    dummy_df = pd.DataFrame({
        "Name": ["Example 1", "Example 2", "Example 3"],
        "Employee Age": [30, 40, 35],
        "Gender": ["Male", "Female", "Male"],
        "Tenure (Months)": [36, 48, 24],
        "Pulse": ["Medium", "High", "Low"],
        "Hasn't been promoted": [12, 30, 15],
        "Minimum Promotion Cycle": [24, 24, 24],
        "College Tier": ["Tier 1", "Tier 2", "Tier 3"],
        "Industry": ["Tech", "Finance", "Healthcare"],
        "Company Type": ["Startup", "Enterprise", "SME"],
        "Last Performance Rating": [3, 1, 4],
        "Compa Ratio": [90, 65, 100],
        "Attrition": [0, 1, 0]
    })
    csv_buffer = io.StringIO()
    dummy_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# ----------------------------------------------------
# A centralized helper function to get the filtered dataframe.
# This function is called in the analysis section so that every chart uses the same filters.
# -----------------------------
def get_filtered_df():
    if not st.session_state.bulk_prediction_complete:
        return None
    df = st.session_state.bulk_result.copy()
    # All filter widgets are placed in the sidebar for reactivity.
    filter_score_min = st.sidebar.slider("Filter: Attrition Score (Min)", 0, 100, 0, key="score_min")
    filter_score_max = st.sidebar.slider("Filter: Attrition Score (Max)", 0, 100, 100, key="score_max")
    possible_cols = [col for col in df.columns if col not in 
                     ["Attrition Score", "What-If Attrition Score", "What-If Negative Triggers", "Prediction Time"]]
    custom_col = st.sidebar.selectbox("Filter: Select column", possible_cols, key="custom_col")
    series = df[custom_col]
    if pd.api.types.is_numeric_dtype(series):
        custom_range = st.sidebar.slider(f"Filter: {custom_col} Range", float(series.min()), float(series.max()), (float(series.min()), float(series.max())), key="custom_range")
        condition = (df[custom_col] >= custom_range[0]) & (df[custom_col] <= custom_range[1])
    else:
        unique_vals = series.unique().tolist()
        selected_vals = st.sidebar.multiselect(f"Filter: {custom_col} values", unique_vals, default=unique_vals, key="custom_vals")
        condition = df[custom_col].isin(selected_vals)
    filtered = df[(df["Attrition Score"] >= filter_score_min) &
                  (df["Attrition Score"] <= filter_score_max) &
                  condition]
    return filtered

# ---------------------------------------
# Helper function: Graph header with tooltip
# ---------------------------------------
def graph_header(title, explanation):
    return f'<h4 style="color: white;">{title} <span title="{explanation}" style="cursor: help; color: #ccc;">&#9432;</span></h4>'

# ---------------------------------------
# Login/Sign Up System
# ---------------------------------------
if not st.session_state.logged_in:
    st.title("Employee Attrition Prediction Tool - Login / Sign Up")
    auth_mode = st.radio("Select Mode", ["Login", "Sign Up"], index=0)
    if auth_mode == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                users = load_users()
                if email in users and users[email]["password"] == password:
                    st.success(f"Welcome back, {users[email]['name']}!")
                    st.session_state.user = users[email]
                    st.session_state.logged_in = True
                    safe_rerun()
                else:
                    st.error("Invalid email or password.")
    else:
        with st.form("signup_form"):
            name = st.text_input("Name")
            designation = st.text_input("Designation")
            company = st.text_input("Company Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    users = load_users()
                    if email in users:
                        st.error("Email already exists. Please log in.")
                    else:
                        user = {
                            "name": name,
                            "designation": designation,
                            "company": company,
                            "email": email,
                            "password": password,
                            "settings": {}
                        }
                        users[email] = user
                        save_users(users)
                        st.success("Account created and logged in!")
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        safe_rerun()
    if not st.session_state.logged_in:
        st.stop()

# ---------------------------------------
# Top Header with Title, My Account Icon, and Logout
# ---------------------------------------
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    safe_rerun()

header_container = st.container()
with header_container:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>Employee Attrition Prediction Tool</h1>", unsafe_allow_html=True)
    with col2:
        if st.button("👤 My Account", key="account_button"):
            st.session_state.nav = "My Account"
        if st.button("Logout", key="logout_button"):
            logout()

# ---------------------------------------
# Sidebar: Global Settings and Mode Selection
# ---------------------------------------
if st.session_state.nav != "My Account":
    with st.sidebar:
        mode = st.radio("Select Mode", ["Train Mode", "Test Mode"], index=0, key="main_mode")
        disabled_flag = (mode == "Test Mode")
        st.markdown("### Global Settings for Bulk Analysis\n*These settings MUST be filled for bulk analysis*")
        global_avg_age = st.slider(
            "Average Employee Age in Company", 18, 100,
            st.session_state.user.get("settings", {}).get("global_avg_age", 35),
            key="global_avg_age", disabled=disabled_flag
        )
        global_female_ratio = st.slider(
            "Women % in Organization", 0, 100,
            st.session_state.user.get("settings", {}).get("global_female_ratio", 40),
            key="global_female_ratio", disabled=disabled_flag
        )
        with st.expander("College Tier Retention Settings", expanded=False):
            bulk_tier1 = st.slider(
                "Tier 1 Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_tier1", 60),
                key="bulk_tier1", disabled=disabled_flag
            )
            bulk_tier2 = st.slider(
                "Tier 2 Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_tier2", 50),
                key="bulk_tier2", disabled=disabled_flag
            )
            bulk_tier3 = st.slider(
                "Tier 3 Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_tier3", 40),
                key="bulk_tier3", disabled=disabled_flag
            )
        with st.expander("Industry Retention Settings", expanded=False):
            bulk_industry_retention = {}
            for ind in industry_options:
                default_val = st.session_state.user.get("settings", {}).get("bulk_industry_retention", {}).get(ind, 60 if ind=="Tech" else 50)
                bulk_industry_retention[ind] = st.slider(
                    f"{ind} Retention (%)", 10, 100, default_val,
                    key=f"bulk_ind_{ind}", disabled=disabled_flag
                )
        with st.expander("Company Type Retention Settings", expanded=False):
            bulk_startup = st.slider(
                "Startup Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60),
                key="bulk_startup", disabled=disabled_flag
            )
            bulk_small = st.slider(
                "Small Size Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Small Size", 55),
                key="bulk_small", disabled=disabled_flag
            )
            bulk_mid = st.slider(
                "Mid Size Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Mid Size", 50),
                key="bulk_mid", disabled=disabled_flag
            )
            bulk_mnc = st.slider(
                "MNC/Giant Company Retention (%)", 10, 100,
                st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("MNC/Giant Company", 45),
                key="bulk_mnc", disabled=disabled_flag
            )

# ---------------------------------------
# Main Navigation
# ---------------------------------------
if st.session_state.nav == "My Account":
    st.markdown("<div style='text-align: center;'><h2>My Account</h2></div>", unsafe_allow_html=True)
    user = st.session_state.user
    st.write("### Account Information")
    st.write(f"**Name:** {user.get('name', '')}")
    st.write(f"**Designation:** {user.get('designation', '')}")
    st.write(f"**Company:** {user.get('company', '')}")
    st.write(f"**Email:** {user.get('email', '')}")
    st.write("### Saved Global Settings")
    user_settings = user.get("settings") or {}
    if user_settings:
        st.json(user_settings)
    else:
        st.info("No global settings saved. Please train your model to save settings.")
    st.write("### Analysis History")
    history = load_user_history(user["email"])
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.info("No history available yet.")
    if st.button("Back to Main"):
        st.session_state.nav = "Tabs"
else:
    if st.session_state.main_mode == "Test Mode":
        selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=0, key="test_industry")
        st.markdown("""
        <div class="tooltip">Read Instructions
          <span class="tooltiptext">
            Ensure you have trained a model in Train Mode.
            <br><br>
            Upload a CSV/Excel file with columns: Name, Employee Age, Gender, Tenure (Months),
            Pulse, Hasn't been promoted, Minimum Promotion Cycle, College Tier, Industry, 
            Company Type, Last Performance Rating, Compa Ratio.
            <br><br>
            (No Attrition column needed for testing.)
          </span>
        </div>
        <style>
        .tooltip { position: relative; display: inline-block; cursor: pointer; font-weight: bold; color: #0073e6; }
        .tooltip .tooltiptext { visibility: hidden; width: 300px; background-color: #333; color: #ddd; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; top: 125%; left: 50%; margin-left: -150px; box-shadow: 0px 0px 6px 0px rgba(0,0,0,0.2); }
        .tooltip:hover .tooltiptext { visibility: visible; }
        </style>
        """, unsafe_allow_html=True)
    else:
        selected_test_industry = None

    if st.session_state.main_mode == "Train Mode":
        st.header("Train Mode")
        selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
        with col2:
            st.markdown("### Training File Guide")
            st.markdown("""
            Your training file must include:
            - A **target column** (e.g., Attrition; use 0 for active, 1 for non‑active).
            - **Feature columns:** Employee Age, Gender, Tenure (Months), Pulse, 
              Hasn't been promoted, Minimum Promotion Cycle, College Tier, Industry, 
              Company Type, Last Performance Rating, Compa Ratio.
            """)
            st.download_button(
                label="Download Dummy Training File",
                data=generate_dummy_training_file(),
                file_name="dummy_training_file.csv",
                mime="text/csv"
            )
        target_column = st.text_input("Enter the target column name", value="Attrition")
        if uploaded_train is not None:
            try:
                train_df = pd.read_csv(uploaded_train) if uploaded_train.name.endswith(".csv") else pd.read_excel(uploaded_train)
                st.write("### Training Data Preview:")
                st.dataframe(train_df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
            if st.button("Train Model"):
                train_model(train_df, target_column, selected_train_industry)
    else:
        st.header("Bulk Employee Attrition Prediction")
        uploaded_file = st.file_uploader("Upload Bulk Data (CSV or Excel)", type=["csv", "xlsx"], key="bulk_file")
        if uploaded_file is not None:
            try:
                df_bulk = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.stop()
            st.write("### Bulk Data Preview:")
            st.dataframe(df_bulk.head())
            required_cols = [
                "Name", "Employee Age", "Gender", "Tenure (Months)", "Pulse",
                "Hasn't been promoted", "Minimum Promotion Cycle", "College Tier",
                "Industry", "Company Type", "Last Performance Rating", "Compa Ratio"
            ]
            missing = [c for c in required_cols if c not in df_bulk.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("🚀 Run Bulk Prediction"):
                        scores, triggers_list, names = [], [], []
                        for idx, row in df_bulk.iterrows():
                            row_dict = row.to_dict()
                            names.append(row_dict.get("Name"))
                            row_dict["Average Employee Age"] = global_avg_age
                            row_dict["Female Employee Ratio"] = global_female_ratio
                            college_tier = row_dict.get("College Tier")
                            if college_tier == "Tier 1":
                                row_dict["College Tier Retention"] = bulk_tier1
                            elif college_tier == "Tier 2":
                                row_dict["College Tier Retention"] = bulk_tier2
                            elif college_tier == "Tier 3":
                                row_dict["College Tier Retention"] = bulk_tier3
                            else:
                                st.warning(f"Row {idx}: Unknown College Tier '{college_tier}'. Using default 40%.")
                                row_dict["College Tier Retention"] = 40
                            ind_val = row_dict.get("Industry")
                            row_dict["Industry Retention"] = bulk_industry_retention.get(ind_val, 50)
                            ctype_val = row_dict.get("Company Type", "Startup")
                            if ctype_val.lower() == "startup":
                                row_dict["Company Type Retention"] = bulk_startup
                            elif "small" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_small
                            elif "mid" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_mid
                            elif "mnc" in ctype_val.lower() or "giant" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_mnc
                            else:
                                row_dict["Company Type Retention"] = 50
                            try:
                                bulk_score, bulk_trigs, _ = predict_attrition(row_dict, selected_test_industry)
                            except Exception as e:
                                st.error(f"Row {idx}: Prediction failed due to {e}. Skipping this row.")
                                scores.append(None)
                                triggers_list.append("Prediction Failed")
                                continue
                            scores.append(bulk_score)
                            neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
                            triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
                            triggers_list.append(triggers_str)
                        df_bulk["Attrition Score"] = scores
                        df_bulk["Negative Triggers"] = triggers_list
                        df_bulk["Name"] = names
                        st.session_state.bulk_result = df_bulk.copy()
                        st.session_state.bulk_prediction_complete = True
                        st.session_state.bulk_result["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_user_event(st.session_state.user["email"], "bulk_prediction", {"rows": len(df_bulk)})

                with btn_cols[1]:
                    if st.session_state.bulk_prediction_complete:
                        st.session_state.enable_what_if = st.checkbox("Enable What-If Analysis", key="whatif_toggle")
                
                # -------------------------------
                # Analysis Section (filters and charts)
                # -------------------------------
                if st.session_state.bulk_prediction_complete:
                    st.markdown("### Analysis (All Charts Respond to Filters)")
                    
                    # Get the filtered dataframe (filters appear in the sidebar)
                    filtered_df = get_filtered_df()
                    st.write("Filtered Bulk Predictions")
                    st.dataframe(filtered_df)
                    
                    # If What-If Analysis is enabled, use the filtered_df for recalculations.
                    if st.session_state.enable_what_if:
                        with st.container():
                            st.markdown("<h3 style='color: white;'>What-If Analysis</h3>", unsafe_allow_html=True)
                            st.info("Adjust parameters below to simulate changes in attrition predictions.")
                            
                            filtered_whatif_df = filtered_df.copy()
                            trig_series = compute_trigger_counts(filtered_whatif_df, "Negative Triggers")
                            
                            # Define widgets for each known trigger.
                            trigger_widget_config = {
                                "Low gender diversity": {
                                    "widget": "slider",
                                    "label": "Women % in Organization",
                                    "min": 0,
                                    "max": 100,
                                    "default": global_female_ratio,
                                    "param": "female_ratio"
                                },
                                "Stagnant promotions": {
                                    "widget": "slider_pair",
                                    "labels": ["Months Since Last Promotion", "Minimum Promotion Cycle"],
                                    "min": [0, 12],
                                    "max": [60, 60],
                                    "default": [
                                        int(filtered_whatif_df["Hasn't been promoted"].mean()) if not filtered_whatif_df.empty else 0,
                                        int(filtered_whatif_df["Minimum Promotion Cycle"].mean()) if not filtered_whatif_df.empty else 24
                                    ],
                                    "params": ["not_promoted", "min_cycle"]
                                },
                                "Very low performance rating": {
                                    "widget": "selectbox",
                                    "label": "Last Performance Rating",
                                    "options": [1,2,3,4,5],
                                    "default": 3,
                                    "param": "rating"
                                },
                                "Low performance rating": {
                                    "widget": "selectbox",
                                    "label": "Last Performance Rating",
                                    "options": [1,2,3,4,5],
                                    "default": 3,
                                    "param": "rating"
                                },
                                "Low compensation competitiveness": {
                                    "widget": "slider",
                                    "label": "Compa Ratio (%)",
                                    "min": 50,
                                    "max": 150,
                                    "default": 90,
                                    "param": "compa_ratio"
                                },
                                "High compensation ratio": {
                                    "widget": "slider",
                                    "label": "Compa Ratio (%)",
                                    "min": 50,
                                    "max": 150,
                                    "default": 90,
                                    "param": "compa_ratio"
                                },
                                "Low college tier retention": {
                                    "widget": "slider_group",
                                    "labels": ["Tier 1 Retention (%)", "Tier 2 Retention (%)", "Tier 3 Retention (%)"],
                                    "min": [10, 10, 10],
                                    "max": [100, 100, 100],
                                    "default": [bulk_tier1, bulk_tier2, bulk_tier3],
                                    "params": ["tier1", "tier2", "tier3"]
                                },
                                "Low industry retention": {
                                    "widget": "slider",
                                    "label": "Industry Retention (%)",
                                    "min": 10,
                                    "max": 100,
                                    "default": 50,
                                    "param": "industry_retention"
                                },
                                "Low company type retention": {
                                    "widget": "slider",
                                    "label": "Company Type Retention (%)",
                                    "min": 10,
                                    "max": 100,
                                    "default": 60,
                                    "param": "company_retention"
                                },
                                "High dissatisfaction (Pulse)": {
                                    "widget": "selectbox",
                                    "label": "Pulse",
                                    "options": ["High", "Medium", "Low"],
                                    "default": "High",
                                    "param": "pulse"
                                }
                            }
                            
                            whatif_params = {}
                            displayed_params = set()
                            for trigger, config in trigger_widget_config.items():
                                if trigger in trig_series.index:
                                    if config["widget"] == "slider":
                                        if config["param"] not in displayed_params:
                                            param_name = config["param"]
                                            whatif_params[param_name] = st.slider(
                                                config["label"],
                                                config["min"],
                                                config["max"],
                                                config["default"],
                                                key=f"whatif_{param_name}"
                                            )
                                            displayed_params.add(param_name)
                                    elif config["widget"] == "selectbox":
                                        if config["param"] not in displayed_params:
                                            param_name = config["param"]
                                            try:
                                                default_index = config["options"].index(config["default"])
                                            except ValueError:
                                                default_index = 0
                                            whatif_params[param_name] = st.selectbox(
                                                config["label"],
                                                config["options"],
                                                index=default_index,
                                                key=f"whatif_{param_name}"
                                            )
                                            displayed_params.add(param_name)
                                    elif config["widget"] == "slider_pair":
                                        param_names = config["params"]
                                        values = []
                                        for i, p in enumerate(param_names):
                                            values.append(
                                                st.slider(
                                                    config["labels"][i],
                                                    config["min"][i],
                                                    config["max"][i],
                                                    config["default"][i],
                                                    key=f"whatif_{p}"
                                                )
                                            )
                                        for i, p in enumerate(param_names):
                                            whatif_params[p] = values[i]
                                        displayed_params.update(param_names)
                                    elif config["widget"] == "slider_group":
                                        param_names = config["params"]
                                        values = []
                                        for i, p in enumerate(param_names):
                                            values.append(
                                                st.slider(
                                                    config["labels"][i],
                                                    config["min"][i],
                                                    config["max"][i],
                                                    config["default"][i],
                                                    key=f"whatif_{p}"
                                                )
                                            )
                                        for i, p in enumerate(param_names):
                                            whatif_params[p] = values[i]
                                        displayed_params.update(param_names)
                            
                            st.markdown("### Recalculated Predictions with What-If Adjustments")
                            new_scores = []
                            new_triggers_list = []
                            df_bulk_whatif = filtered_whatif_df.copy()
                            for idx, row in df_bulk_whatif.iterrows():
                                new_row = dict(row)
                                new_row["Average Employee Age"] = global_avg_age
                                new_row["Female Employee Ratio"] = whatif_params.get("female_ratio", row.get("Female Employee Ratio", global_female_ratio))
                                new_row["Hasn't been promoted"] = whatif_params.get("not_promoted", row.get("Hasn't been promoted"))
                                new_row["Minimum Promotion Cycle"] = whatif_params.get("min_cycle", row.get("Minimum Promotion Cycle"))
                                new_row["Last Performance Rating"] = whatif_params.get("rating", row.get("Last Performance Rating"))
                                new_row["Compa Ratio"] = whatif_params.get("compa_ratio", row.get("Compa Ratio"))
                                
                                college_tier = row.get("College Tier")
                                if college_tier == "Tier 1":
                                    new_row["College Tier Retention"] = whatif_params.get("tier1", row.get("College Tier Retention", bulk_tier1))
                                elif college_tier == "Tier 2":
                                    new_row["College Tier Retention"] = whatif_params.get("tier2", row.get("College Tier Retention", bulk_tier2))
                                elif college_tier == "Tier 3":
                                    new_row["College Tier Retention"] = whatif_params.get("tier3", row.get("College Tier Retention", bulk_tier3))
                                else:
                                    new_row["College Tier Retention"] = row.get("College Tier Retention", 40)
                                
                                industry_val = row.get("Industry")
                                new_row["Industry Retention"] = whatif_params.get("industry_retention", row.get("Industry Retention", bulk_industry_retention.get(industry_val, 50)))
                                
                                ctype_val = row.get("Company Type", "Startup")
                                if ctype_val.lower() == "startup":
                                    default_company_retention = bulk_startup
                                elif "small" in ctype_val.lower():
                                    default_company_retention = bulk_small
                                elif "mid" in ctype_val.lower():
                                    default_company_retention = bulk_mid
                                elif "mnc" in ctype_val.lower() or "giant" in ctype_val.lower():
                                    default_company_retention = bulk_mnc
                                else:
                                    default_company_retention = 50
                                new_row["Company Type Retention"] = whatif_params.get("company_retention", row.get("Company Type Retention", default_company_retention))
                                new_row["Pulse"] = whatif_params.get("pulse", row.get("Pulse"))
                                
                                try:
                                    new_score, new_trigs, _ = predict_attrition(new_row, selected_test_industry)
                                except Exception as e:
                                    new_score = None
                                    new_trigs = ["Prediction Failed"]
                                
                                new_scores.append(new_score)
                                neg_trigs = [t for t in new_trigs if t in TRIGGER_DETAILS]
                                triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
                                new_triggers_list.append(triggers_str)
                            
                            df_bulk_whatif["What-If Attrition Score"] = new_scores
                            df_bulk_whatif["What-If Negative Triggers"] = new_triggers_list
                            
                            st.dataframe(df_bulk_whatif)
                            
                            high_risk_w = (df_bulk_whatif["What-If Attrition Score"] >= 75).sum()
                            mod_high_w = ((df_bulk_whatif["What-If Attrition Score"] >= 60) & (df_bulk_whatif["What-If Attrition Score"] < 75)).sum()
                            moderate_w = ((df_bulk_whatif["What-If Attrition Score"] >= 35) & (df_bulk_whatif["What-If Attrition Score"] < 60)).sum()
                            low_w = (df_bulk_whatif["What-If Attrition Score"] < 35).sum()
                            risk_df_w = pd.DataFrame({
                                "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                                "Count": [high_risk_w, mod_high_w, moderate_w, low_w]
                            })
                            st.markdown("### What-If Risk Distribution")
                            st.bar_chart(risk_df_w.set_index("Risk Category"))
                    
                    # Else use Standard Analysis
                    else:
                        analysis_col1, analysis_col2 = st.columns([0.35, 0.65])
                        with analysis_col1:
                            st.subheader("Filters")
                            st.write("Filtered Bulk Predictions")
                            st.dataframe(filtered_df)
                            
                            total = len(filtered_df)
                            if total > 0:
                                high_risk = (filtered_df["Attrition Score"] >= 75).sum()
                                mod_high = ((filtered_df["Attrition Score"] >= 60) & (filtered_df["Attrition Score"] < 75)).sum()
                                moderate = ((filtered_df["Attrition Score"] >= 35) & (filtered_df["Attrition Score"] < 60)).sum()
                                low = (filtered_df["Attrition Score"] < 35).sum()
                                risk_df = pd.DataFrame({
                                    "Risk Category": ["High (>=75)", "Moderate High (60-74)", "Moderate (35-59)", "Low (<35)"],
                                    "Percentage": [high_risk/total*100, mod_high/total*100, moderate/total*100, low/total*100]
                                })
                            else:
                                risk_df = pd.DataFrame({
                                    "Risk Category": ["High (>=75)", "Moderate High (60-74)", "Moderate (35-59)", "Low (<35)"],
                                    "Percentage": [0, 0, 0, 0]
                                })
                            
                            st.markdown("### Risk Distribution (%)")
                            risk_chart = alt.Chart(risk_df).mark_bar().encode(
                                x=alt.X("Risk Category:N", sort=None),
                                y=alt.Y("Percentage:Q", title="Percentage (%)"),
                                tooltip=["Risk Category", "Percentage"]
                            )
                            st.altair_chart(risk_chart, use_container_width=True)
                        
                        with analysis_col2:
                            st.markdown("<h3 style='color: white;'>Custom Graph Builder</h3>", unsafe_allow_html=True)
                            with st.form("custom_graph_form"):
                                x_axis = st.selectbox("Select X Axis", options=filtered_df.columns, key="custom_x")
                                y_axis = st.selectbox("Select Y Axis", options=filtered_df.columns, key="custom_y")
                                data_label = st.selectbox("Select Data Label (Optional)", options=["None"] + list(filtered_df.columns), key="custom_label")
                                submitted_custom = st.form_submit_button("Generate Custom Chart")
                            if submitted_custom:
                                if x_axis == "Negative Triggers" or y_axis == "Negative Triggers":
                                    ct = compute_trigger_counts(filtered_df, "Negative Triggers").reset_index()
                                    ct.columns = ["Trigger", "Count"]
                                    custom_chart = alt.Chart(ct).mark_bar(color="#e45756").encode(
                                        x=alt.X("Trigger:N", title="Negative Triggers"),
                                        y=alt.Y("Count:Q", title="Count"),
                                        tooltip=["Trigger", "Count"]
                                    )
                                else:
                                    x_is_numeric = pd.api.types.is_numeric_dtype(filtered_df[x_axis])
                                    y_is_numeric = pd.api.types.is_numeric_dtype(filtered_df[y_axis])
                                    if x_is_numeric and y_is_numeric:
                                        custom_chart = alt.Chart(filtered_df).mark_circle(size=60, color="#4c78a8").encode(
                                            x=alt.X(f"{x_axis}:Q", title=x_axis),
                                            y=alt.Y(f"{y_axis}:Q", title=y_axis),
                                            tooltip=["Name", x_axis, y_axis]
                                        )
                                    elif not x_is_numeric and y_is_numeric:
                                        custom_chart = alt.Chart(filtered_df).mark_boxplot(color="#e45756").encode(
                                            x=alt.X(f"{x_axis}:N", title=x_axis),
                                            y=alt.Y(f"{y_axis}:Q", title=y_axis),
                                            tooltip=[x_axis, y_axis]
                                        )
                                    elif x_is_numeric and not y_is_numeric:
                                        custom_chart = alt.Chart(filtered_df).mark_boxplot(color="#e45756").encode(
                                            x=alt.X(f"{y_axis}:N", title=y_axis),
                                            y=alt.Y(f"{x_axis}:Q", title=x_axis),
                                            tooltip=[x_axis, y_axis]
                                        )
                                    else:
                                        custom_chart = alt.Chart(filtered_df).mark_bar(color="#4c78a8").encode(
                                            x=alt.X(f"{x_axis}:N", title=x_axis),
                                            y=alt.Y("count()", title="Count"),
                                            tooltip=[x_axis]
                                        )
                                st.session_state.custom_charts.insert(0, {
                                    "chart": custom_chart,
                                    "title": f"Custom Chart: {x_axis} vs {y_axis}",
                                    "explanation": "This chart was generated based on your selected axes."
                                })
                                st.success("Custom chart generated and added!")
                            
                            if st.session_state.custom_charts:
                                st.markdown("<h3 style='color: white;'>Your Custom Charts</h3>", unsafe_allow_html=True)
                                for custom in st.session_state.custom_charts:
                                    st.markdown(graph_header(custom["title"], custom["explanation"]), unsafe_allow_html=True)
                                    st.altair_chart(custom["chart"], use_container_width=True)
                            
                            st.markdown("## Quick Charts")
                            
                            # Quick charts use the same filtered_df
                            df_for_charts = filtered_df
                            
                            with st.expander("Distribution Analysis: Overall makeup"):
                                if "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    chart1 = alt.Chart(df_for_charts).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=20), title="Attrition Score"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart1, use_container_width=True)
                                else:
                                    st.write("No data for Attrition Score Distribution.")
                                
                                if "Employee Age" in df_for_charts.columns and not df_for_charts.empty:
                                    chart2 = alt.Chart(df_for_charts).mark_bar(color="#e45756").encode(
                                        x=alt.X("Employee Age:Q", bin=alt.Bin(maxbins=20), title="Employee Age"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart2, use_container_width=True)
                                else:
                                    st.write("No data for Employee Age Distribution.")
                                
                                if "Tenure (Months)" in df_for_charts.columns and not df_for_charts.empty:
                                    chart3 = alt.Chart(df_for_charts).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Tenure (Months):Q", bin=alt.Bin(maxbins=20), title="Tenure (Months)"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart3, use_container_width=True)
                                else:
                                    st.write("No data for Tenure Distribution.")
                                
                                if "Compa Ratio" in df_for_charts.columns and not df_for_charts.empty:
                                    chart4 = alt.Chart(df_for_charts).mark_bar(color="#e45756").encode(
                                        x=alt.X("Compa Ratio:Q", bin=alt.Bin(maxbins=20), title="Compa Ratio"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart4, use_container_width=True)
                                else:
                                    st.write("No data for Compa Ratio Distribution.")
                                
                                if "Last Performance Rating" in df_for_charts.columns and not df_for_charts.empty:
                                    chart5 = alt.Chart(df_for_charts).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Last Performance Rating:Q", bin=alt.Bin(maxbins=5), title="Last Performance Rating"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart5, use_container_width=True)
                                else:
                                    st.write("No data for Performance Rating Distribution.")
                            
                            with st.expander("Comparative Analysis: Relationships"):
                                if "Employee Age" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    chart1 = alt.Chart(df_for_charts).mark_circle(size=60, color="#4c78a8").encode(
                                        x=alt.X("Employee Age:Q", title="Employee Age"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Name", "Employee Age", "Attrition Score"]
                                    )
                                    st.altair_chart(chart1, use_container_width=True)
                                else:
                                    st.write("No data for Employee Age vs Attrition Score.")
                                
                                if "Compa Ratio" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    chart2 = alt.Chart(df_for_charts).mark_circle(size=60, color="#e45756").encode(
                                        x=alt.X("Compa Ratio:Q", title="Compa Ratio"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Name", "Compa Ratio", "Attrition Score"]
                                    )
                                    st.altair_chart(chart2, use_container_width=True)
                                else:
                                    st.write("No data for Compa Ratio vs Attrition Score.")
                                
                                if "Gender" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    chart3 = alt.Chart(df_for_charts).mark_boxplot(color="#4c78a8").encode(
                                        x=alt.X("Gender:N", title="Gender"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Gender", "Attrition Score"]
                                    )
                                    st.altair_chart(chart3, use_container_width=True)
                                else:
                                    st.write("No data for Attrition Score by Gender.")
                                
                                if "College Tier" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    chart4 = alt.Chart(df_for_charts).mark_boxplot(color="#4c78a8").encode(
                                        x=alt.X("College Tier:N", title="College Tier"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["College Tier", "Attrition Score"]
                                    )
                                    st.altair_chart(chart4, use_container_width=True)
                                else:
                                    st.write("No data for Attrition Score by College Tier.")
                                
                                if all(x in df_for_charts.columns for x in ["Tenure (Months)", "Attrition Score", "Industry"]) and not df_for_charts.empty:
                                    chart5 = alt.Chart(df_for_charts).mark_circle(size=60, color="#4c78a8").encode(
                                        x=alt.X("Tenure (Months):Q", title="Tenure (Months)"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        color=alt.Color("Industry:N", title="Industry"),
                                        tooltip=["Industry", "Tenure (Months)", "Attrition Score"]
                                    )
                                    st.altair_chart(chart5, use_container_width=True)
                                else:
                                    st.write("No data for Tenure vs Attrition Score by Industry.")
                            
                            with st.expander("Correlation & Relationship Analysis: Numerical interactions"):
                                try:
                                    numeric_df = df_for_charts.select_dtypes(include=[np.number])
                                    if not numeric_df.empty:
                                        corr = numeric_df.corr().reset_index().melt(id_vars="index")
                                        chart1 = alt.Chart(corr).mark_rect().encode(
                                            x=alt.X("index:N", title=""),
                                            y=alt.Y("variable:N", title=""),
                                            color=alt.Color("value:Q", scale=alt.Scale(scheme='redblue')),
                                            tooltip=["index", "variable", "value"]
                                        )
                                        st.altair_chart(chart1, use_container_width=True)
                                    else:
                                        st.write("No numeric data for correlation heatmap.")
                                except Exception as e:
                                    st.write("Correlation Heatmap not available.")
                                
                                try:
                                    import seaborn as sns
                                    numeric_df = df_for_charts.select_dtypes(include=[np.number])
                                    if not numeric_df.empty:
                                        fig = sns.pairplot(numeric_df).fig
                                        st.pyplot(fig)
                                    else:
                                        st.write("No numeric data for pairwise scatter plot matrix.")
                                except Exception as e:
                                    st.write("Pairwise Scatter Plot Matrix not available.")
                            
                            with st.expander("Trigger & Factor Analysis: Attrition risk factors"):
                                if "Negative Triggers" in df_for_charts.columns and not df_for_charts.empty:
                                    ct = compute_trigger_counts(df_for_charts, "Negative Triggers").reset_index()
                                    ct.columns = ["Trigger", "Count"]
                                    if not ct.empty:
                                        chart1 = alt.Chart(ct).mark_bar(color="#e45756").encode(
                                            x=alt.X("Trigger:N", sort='-y', title="Trigger"),
                                            y=alt.Y("Count:Q", title="Count"),
                                            tooltip=["Trigger", "Count"]
                                        )
                                        st.altair_chart(chart1, use_container_width=True)
                                    else:
                                        st.write("No triggers found.")
                                else:
                                    st.write("No data for Negative Triggers Count.")
                                
                                high_risk_df = df_for_charts[df_for_charts["Attrition Score"] >= 75]
                                if not high_risk_df.empty and "Negative Triggers" in high_risk_df.columns:
                                    ct_high = compute_trigger_counts(high_risk_df, "Negative Triggers").reset_index()
                                    ct_high.columns = ["Trigger", "Count"]
                                    if not ct_high.empty:
                                        chart2 = alt.Chart(ct_high).mark_arc().encode(
                                            theta=alt.Theta(field="Count", type="quantitative"),
                                            color=alt.Color(field="Trigger", type="nominal"),
                                            tooltip=["Trigger", "Count"]
                                        )
                                        st.altair_chart(chart2, use_container_width=True)
                                    else:
                                        st.write("No triggers in High-Risk group.")
                                else:
                                    st.write("No data for High-Risk trigger distribution.")
                            
                            with st.expander("Retention & Industry Analysis: Industry insights"):
                                if "Industry" in df_for_charts.columns and not df_for_charts.empty:
                                    industry_counts = df_for_charts["Industry"].value_counts().reset_index()
                                    industry_counts.columns = ['Industry', 'Count']
                                    chart1 = alt.Chart(industry_counts).mark_arc().encode(
                                        theta=alt.Theta(field="Count", type="quantitative"),
                                        color=alt.Color(field="Industry", type="nominal"),
                                        tooltip=["Industry", "Count"]
                                    )
                                    st.altair_chart(chart1, use_container_width=True)
                                else:
                                    st.write("No data for Industry Distribution.")
                                
                                if all(x in df_for_charts.columns for x in ["Industry", "Tenure (Months)"]) and not df_for_charts.empty:
                                    chart2 = alt.Chart(df_for_charts).mark_boxplot(color="#4c78a8").encode(
                                        x=alt.X("Industry:N", title="Industry"),
                                        y=alt.Y("Tenure (Months):Q", title="Tenure (Months)"),
                                        tooltip=["Industry", "Tenure (Months)"]
                                    )
                                    st.altair_chart(chart2, use_container_width=True)
                                else:
                                    st.write("No data for Tenure by Industry.")
                                
                                if "Company Type" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    compa_df = df_for_charts.groupby("Company Type").agg({"Attrition Score": "mean"}).reset_index()
                                    chart3 = alt.Chart(compa_df).mark_bar(color="#e45756").encode(
                                        x=alt.X("Company Type:N", title="Company Type"),
                                        y=alt.Y("Attrition Score:Q", title="Average Attrition Score"),
                                        tooltip=["Company Type", "Attrition Score"]
                                    )
                                    st.altair_chart(chart3, use_container_width=True)
                                else:
                                    st.write("No data for Company Type analysis.")
                                
                                if "College Tier" in df_for_charts.columns and "Attrition Score" in df_for_charts.columns and not df_for_charts.empty:
                                    college_df = df_for_charts.groupby("College Tier").agg({"Attrition Score": "mean"}).reset_index()
                                    chart4 = alt.Chart(college_df).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("College Tier:N", title="College Tier"),
                                        y=alt.Y("Attrition Score:Q", title="Average Attrition Score"),
                                        tooltip=["College Tier", "Attrition Score"]
                                    )
                                    st.altair_chart(chart4, use_container_width=True)
                                else:
                                    st.write("No data for College Tier analysis.")
                            
                            with st.expander("Temporal Analysis: Trends over time"):
                                if "Prediction Time" in df_for_charts.columns and not df_for_charts.empty:
                                    df_time = df_for_charts.copy()
                                    df_time["Prediction Time"] = pd.to_datetime(df_time["Prediction Time"], errors="coerce")
                                    df_time = df_time.dropna(subset=["Prediction Time"])
                                    if not df_time.empty:
                                        trend_df = df_time.groupby(df_time["Prediction Time"].dt.date).agg({"Attrition Score": "mean"}).reset_index()
                                        trend_df.columns = ["Date", "Average Attrition Score"]
                                        chart1 = alt.Chart(trend_df).mark_line(point=True, color="#e45756").encode(
                                            x=alt.X("Date:T", title="Date"),
                                            y=alt.Y("Average Attrition Score:Q", title="Average Attrition Score"),
                                            tooltip=["Date", "Average Attrition Score"]
                                        )
                                        st.altair_chart(chart1, use_container_width=True)
                                    else:
                                        st.write("No valid Prediction Time data for trend analysis.")
                                else:
                                    st.write("No Prediction Time data available.")
                                
                                if "Prediction Time" in df_for_charts.columns and not df_for_charts.empty:
                                    df_time = df_for_charts.copy()
                                    df_time["Prediction Time"] = pd.to_datetime(df_time["Prediction Time"], errors="coerce")
                                    df_time = df_time.dropna(subset=["Prediction Time"])
                                    if not df_time.empty:
                                        trend_df = df_time.sort_values("Prediction Time").copy()
                                        trend_df["Rolling Avg Attrition Score"] = trend_df["Attrition Score"].rolling(window=3, min_periods=1).mean()
                                        chart2 = alt.Chart(trend_df).mark_line(point=True, color="#4c78a8").encode(
                                            x=alt.X("Prediction Time:T", title="Prediction Time"),
                                            y=alt.Y("Rolling Avg Attrition Score:Q", title="Rolling Average Attrition Score"),
                                            tooltip=["Prediction Time", "Rolling Avg Attrition Score"]
                                        )
                                        st.altair_chart(chart2, use_container_width=True)
                                    else:
                                        st.write("No valid Prediction Time data for rolling average.")
                                else:
                                    st.write("No Prediction Time data available.")
                else:
                    st.info("Run Bulk Prediction to see the analysis.")
