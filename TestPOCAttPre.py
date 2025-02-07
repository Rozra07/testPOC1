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
# Helper: Safe rerun
# ---------------------------------------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Please update Streamlit (>=0.65.0).")

# ----------------------------------------------------
# Session state initialization
# ----------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "nav" not in st.session_state:
    st.session_state.nav = "Tabs"
if "user" not in st.session_state:
    st.session_state.user = {}
if "bulk_prediction_complete" not in st.session_state:
    st.session_state.bulk_prediction_complete = False
if "bulk_result" not in st.session_state:
    st.session_state.bulk_result = None
if "enable_what_if" not in st.session_state:
    st.session_state.enable_what_if = False
if "custom_charts" not in st.session_state:
    st.session_state.custom_charts = []  # List for custom charts
if "enlarged_chart" not in st.session_state:
    st.session_state.enlarged_chart = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# ---------------------------------------
# Helper: Display chart with header and "View Larger" button.
# ---------------------------------------
def display_chart_with_enlarge(chart, title, explanation, key):
    st.markdown(graph_header(title, explanation), unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    if st.button("View Larger", key=f"enlarge_{key}"):
        st.session_state.enlarged_chart = {"chart": chart, "title": title, "explanation": explanation}

# ---------------------------------------
# Helper: Show enlarged chart modal if set.
# ---------------------------------------
def show_enlarged_chart():
    if st.session_state.enlarged_chart is not None:
        st.markdown("""
        <style>
        .modal-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0,0,0,0.7); z-index: 1000;
        }
        .modal-content {
            position: fixed; top: 10%; left: 10%; width: 80%;
            background-color: #222; color: white; padding: 20px; z-index: 1001;
            border: 2px solid #555; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
            st.markdown('<div class="modal-content">', unsafe_allow_html=True)
            st.markdown(f"<h2>{st.session_state.enlarged_chart['title']}</h2>", unsafe_allow_html=True)
            st.altair_chart(st.session_state.enlarged_chart["chart"], use_container_width=True)
            if st.button("Close Enlarged View"):
                st.session_state.enlarged_chart = None
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# Helper: Build report HTML for download.
# ---------------------------------------
def get_report_html(default_charts, custom_charts, df):
    html = "<html><head><style>body{background-color:black;color:white;}</style></head><body>"
    html += "<h1>Bulk Analysis Report</h1>"
    for title, chart in default_charts:
        html += f"<h2>{title}</h2>"
        html += chart.to_html()
    if custom_charts:
        html += "<h2>Your Custom Charts</h2>"
        for custom in custom_charts:
            html += f"<h3>{custom['title']}</h3>"
            html += custom["chart"].to_html()
    # Optionally, include a snapshot of the filtered table.
    html += "<h2>Filtered Data Snapshot</h2>"
    html += df.to_html()
    html += "</body></html>"
    return html

# ---------------------------------------
# Helper: Graph header with tooltip.
# ---------------------------------------
def graph_header(title, explanation):
    return f'<h4 style="color: white;">{title} <span title="{explanation}" style="cursor: help; color: #ccc;">&#9432;</span></h4>'

# ---------------------------------------
# User storage functions
# ---------------------------------------
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
    event = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "event_type": event_type, "event_data": event_data}
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

# ---------------------------------------
# Global Industry Options
# ---------------------------------------
industry_options = ["Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
                    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"]

# ---------------------------------------
# Model Training/Predict Functions
# ---------------------------------------
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
    
    user = st.session_state.user
    user_settings = user.get("settings") or {}
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3
    user_settings["bulk_industry_retention"] = { ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind=="Tech" else 50)
                                                  for ind in industry_options }
    user_settings["bulk_company_retention"] = { "Startup": st.session_state.bulk_startup,
                                                 "Small Size": st.session_state.bulk_small,
                                                 "Mid Size": st.session_state.bulk_mid,
                                                 "MNC/Giant Company": st.session_state.bulk_mnc }
    user["settings"] = user_settings
    users = load_users()
    users[user["email"]] = user
    save_users(users)
    save_user_event(user["email"], "training", {"action": "Model retrained", "industry": industry})

def update_industry_record(industry, model_file, scaler_file, feature_file):
    from datetime import datetime
    record = {"Industry": industry,
              "Model_File": model_file,
              "Scaler_File": scaler_file,
              "Feature_File": feature_file,
              "Training_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
            "lack_female_mentors": "Few female mentors/leaders are available.",
            "rigid_policies": "Policies are too rigid (e.g., no maternity or remote options)."
        },
        "solutions": {
            "lack_female_applicants": "Partner with targeted institutions and emphasize diversity.",
            "lack_female_mentors": "Establish mentorship programs and leadership development.",
            "rigid_policies": "Introduce flexible arrangements and improved benefits."
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are not transparent.",
            "no_mentorship": "There is a lack of mentorship/upskilling opportunities.",
            "bureaucratic_structure": "The structure is overly bureaucratic."
        },
        "solutions": {
            "unclear_criteria": "Publish clear guidelines and KPIs.",
            "no_mentorship": "Launch mentoring programs and offer upskilling sessions.",
            "bureaucratic_structure": "Streamline processes to foster agility."
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job roles or expectations are unclear.",
            "no_feedback": "Continuous feedback is lacking.",
            "skill_gaps": "Training needs are not addressed."
        },
        "solutions": {
            "misaligned_role": "Clarify responsibilities and set SMART goals.",
            "no_feedback": "Implement regular check‑ins and reviews.",
            "skill_gaps": "Offer targeted training and development opportunities."
        }
    }
    # ... (Additional triggers can be added similarly)
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
# Predict attrition using both ML and rule-based methods
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
# Compute trigger counts from a comma-separated column
# ----------------------------------------------------
def compute_trigger_counts(df, column_name):
    triggers_list = []
    for val in df[column_name].dropna():
        if val.strip() != "" and val != "None":
            triggers_list.extend([x.strip() for x in val.split(",") if x.strip()])
    if triggers_list:
        return pd.Series(triggers_list).value_counts()
    else:
        return pd.Series(dtype=int)

# ---------------------------------------
# LOGIN / SIGNUP SYSTEM
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
                        user = {"name": name, "designation": designation, "company": company, "email": email, "password": password, "settings": {}}
                        users[email] = user
                        save_users(users)
                        st.success("Account created and logged in!")
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        safe_rerun()
    if not st.session_state.logged_in:
        st.stop()

# ---------------------------------------
# TOP HEADER: Title, My Account, Logout
# ---------------------------------------
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    safe_rerun()

header_container = st.container()
with header_container:
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("<h1>Employee Attrition Prediction Tool</h1>", unsafe_allow_html=True)
    with col2:
        if st.button("👤 My Account", key="account_button"):
            st.session_state.nav = "My Account"
        if st.button("Logout", key="logout_button"):
            logout()

# ---------------------------------------
# SIDEBAR: Global Settings and Mode Selection
# ---------------------------------------
if st.session_state.nav != "My Account":
    with st.sidebar:
        mode = st.radio("Select Mode", ["Train Mode", "Test Mode"], index=0, key="main_mode")
        disabled_flag = (mode=="Test Mode")
        st.markdown("### Global Settings for Bulk Analysis\n*These settings MUST be filled for bulk analysis*")
        global_avg_age = st.slider("Average Employee Age in Company", 18, 100,
                                   st.session_state.user.get("settings", {}).get("global_avg_age", 35),
                                   key="global_avg_age", disabled=disabled_flag)
        global_female_ratio = st.slider("Women % in Organization", 0, 100,
                                        st.session_state.user.get("settings", {}).get("global_female_ratio", 40),
                                        key="global_female_ratio", disabled=disabled_flag)
        with st.expander("College Tier Retention Settings", expanded=False):
            bulk_tier1 = st.slider("Tier 1 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier1", 60),
                                   key="bulk_tier1", disabled=disabled_flag)
            bulk_tier2 = st.slider("Tier 2 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier2", 50),
                                   key="bulk_tier2", disabled=disabled_flag)
            bulk_tier3 = st.slider("Tier 3 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier3", 40),
                                   key="bulk_tier3", disabled=disabled_flag)
        with st.expander("Industry Retention Settings", expanded=False):
            bulk_industry_retention = {}
            for ind in industry_options:
                default_val = st.session_state.user.get("settings", {}).get("bulk_industry_retention", {}).get(ind, 60 if ind=="Tech" else 50)
                bulk_industry_retention[ind] = st.slider(f"{ind} Retention (%)", 10, 100, default_val,
                                                        key=f"bulk_ind_{ind}", disabled=disabled_flag)
        with st.expander("Company Type Retention Settings", expanded=False):
            bulk_startup = st.slider("Startup Retention (%)", 10, 100,
                                     st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60),
                                     key="bulk_startup", disabled=disabled_flag)
            bulk_small = st.slider("Small Size Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Small Size", 55),
                                   key="bulk_small", disabled=disabled_flag)
            bulk_mid = st.slider("Mid Size Retention (%)", 10, 100,
                                 st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Mid Size", 50),
                                 key="bulk_mid", disabled=disabled_flag)
            bulk_mnc = st.slider("MNC/Giant Company Retention (%)", 10, 100,
                                 st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("MNC/Giant Company", 45),
                                 key="bulk_mnc", disabled=disabled_flag)

# ---------------------------------------
# MAIN NAVIGATION
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
        .tooltip .tooltiptext { visibility: hidden; width: 300px; background-color: #333; color: #ddd; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; top: 125%; left: 50%; margin-left: -150px; box-shadow: 0px 0px 6px rgba(0,0,0,0.2); }
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
            st.download_button("Download Dummy Training File", data=generate_dummy_training_file(), file_name="dummy_training_file.csv", mime="text/csv")
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
            required_cols = ["Name", "Employee Age", "Gender", "Tenure (Months)", "Pulse",
                             "Hasn't been promoted", "Minimum Promotion Cycle", "College Tier",
                             "Industry", "Company Type", "Last Performance Rating", "Compa Ratio"]
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
                        # Remove any old What-If columns if they exist.
                        for col in ["What-If Attrition Score", "What-If Negative Triggers"]:
                            if col in df_bulk.columns:
                                df_bulk.drop(columns=[col], inplace=True)
                        st.session_state.bulk_result = df_bulk.copy()
                        st.session_state.bulk_prediction_complete = True
                        st.session_state.bulk_result["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_user_event(st.session_state.user["email"], "bulk_prediction", {"rows": len(df_bulk)})
                with btn_cols[1]:
                    if st.session_state.bulk_prediction_complete:
                        st.session_state.enable_what_if = st.checkbox("Enable What-If Analysis", key="whatif_toggle")
                
                # -------------------------------
                # DOWNLOAD REPORT BUTTON (placed here so filtered_df is defined)
                # -------------------------------
                if st.session_state.bulk_prediction_complete:
                    # Use the filtered_df from Analysis if available; otherwise, default to bulk_result.
                    if st.session_state.filtered_df is None:
                        st.session_state.filtered_df = st.session_state.bulk_result
                    if st.button("Download Report"):
                        default_charts = []
                        # Recreate default charts using the stored filtered_df.
                        df_for_report = st.session_state.filtered_df.copy()
                        # Refined correlation: remove what-if columns if present.
                        numeric_cols = list(df_for_report.select_dtypes(include=[np.number]).columns)
                        for col in ["What-If Attrition Score", "What-If Negative Triggers"]:
                            if col in numeric_cols:
                                numeric_cols.remove(col)
                        corr = df_for_report[numeric_cols].corr().reset_index().melt(id_vars="index")
                        corr_chart = alt.Chart(corr).mark_rect().encode(
                            x=alt.X("index:N", title=""),
                            y=alt.Y("variable:N", title=""),
                            color=alt.Color("value:Q", scale=alt.Scale(scheme='redblue')),
                            tooltip=["index", "variable", "value"]
                        )
                        default_charts.append(("Correlation Heatmap", corr_chart))
                        scatter_chart = alt.Chart(df_for_report).mark_circle(size=60, color="#4c78a8").encode(
                            x=alt.X("Employee Age:Q", title="Employee Age"),
                            y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                            tooltip=["Name", "Employee Age", "Attrition Score", "Industry"]
                        )
                        default_charts.append(("Employee Age vs Attrition Score", scatter_chart))
                        hist_chart = alt.Chart(df_for_report).mark_bar(color="#e45756").encode(
                            x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=20), title="Attrition Score"),
                            y=alt.Y("count()", title="Frequency")
                        )
                        default_charts.append(("Attrition Score Distribution", hist_chart))
                        box_chart = alt.Chart(df_for_report).mark_boxplot(color="#4c78a8").encode(
                            x=alt.X("Gender:N", title="Gender"),
                            y=alt.Y("Employee Age:Q", title="Employee Age"),
                            tooltip=["Gender", "Employee Age"]
                        )
                        default_charts.append(("Employee Age by Gender", box_chart))
                        compa_chart = alt.Chart(df_for_report).mark_circle(size=60, color="#e45756").encode(
                            x=alt.X("Compa Ratio:Q", title="Compa Ratio"),
                            y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                            tooltip=["Name", "Compa Ratio", "Attrition Score"]
                        )
                        default_charts.append(("Compa Ratio vs Attrition Score", compa_chart))
                        pie_chart = alt.Chart(df_for_report['Industry'].value_counts().reset_index().rename(columns={"index": "Industry", "Industry": "Count"})).mark_arc().encode(
                            theta=alt.Theta(field="Count", type="quantitative"),
                            color=alt.Color(field="Industry", type="nominal"),
                            tooltip=["Industry", "Count"]
                        )
                        default_charts.append(("Industry Distribution", pie_chart))
                        # Build the report HTML.
                        report_html = get_report_html(default_charts, st.session_state.custom_charts, df_for_report)
                        st.download_button("Download Report", report_html, "Bulk_Report.html", "text/html")
                
                # -------------------------------
                # WHAT-IF ANALYSIS SECTION
                # -------------------------------
                if st.session_state.enable_what_if:
                    with st.container():
                        st.markdown("<h3 style='color: white;'>What-If Analysis</h3>", unsafe_allow_html=True)
                        st.info("Adjust parameters below to simulate changes in predicted attrition based on negative triggers.")
                        whatif_params = {}
                        trig_series = compute_trigger_counts(st.session_state.bulk_result, "Negative Triggers")
                        if "Low gender diversity" in trig_series.index:
                            whatif_params["female_ratio"] = st.slider("Women % in Organization", 0, 100, global_female_ratio, key="whatif_female")
                        if "Stagnant promotions" in trig_series.index:
                            default_not_promoted = int(st.session_state.bulk_result["Hasn't been promoted"].mean())
                            default_min_cycle = int(st.session_state.bulk_result["Minimum Promotion Cycle"].mean())
                            whatif_params["not_promoted"] = st.slider("Months Since Last Promotion", 0, 60, default_not_promoted, key="whatif_not_promoted")
                            whatif_params["min_cycle"] = st.slider("Minimum Promotion Cycle", 12, 60, default_min_cycle, key="whatif_min_cycle")
                        if any(x in trig_series.index for x in ["Very low performance rating", "Low performance rating"]):
                            default_rating = int(st.session_state.bulk_result["Last Performance Rating"].mean())
                            default_rating = min(max(default_rating, 1), 5)
                            whatif_params["rating"] = st.selectbox("Last Performance Rating", [1,2,3,4,5], index=default_rating-1, key="whatif_rating")
                        if any(x in trig_series.index for x in ["Low compensation competitiveness", "High compensation ratio"]):
                            default_compa = int(st.session_state.bulk_result["Compa Ratio"].mean())
                            whatif_params["compa_ratio"] = st.slider("Compa Ratio (%)", 50, 150, default_compa, key="whatif_compa")
                        if "Low college tier retention" in trig_series.index:
                            whatif_params["tier1"] = st.slider("Tier 1 Retention (%)", 10, 100, bulk_tier1, key="whatif_tier1")
                            whatif_params["tier2"] = st.slider("Tier 2 Retention (%)", 10, 100, bulk_tier2, key="whatif_tier2")
                            whatif_params["tier3"] = st.slider("Tier 3 Retention (%)", 10, 100, bulk_tier3, key="whatif_tier3")
                        if "Low industry retention" in trig_series.index:
                            avg_ind = int(np.mean(list(bulk_industry_retention.values())))
                            whatif_params["industry_retention"] = st.slider("Industry Retention (%)", 10, 100, avg_ind, key="whatif_industry")
                        if "Low company type retention" in trig_series.index:
                            whatif_params["company_retention"] = st.slider("Company Type Retention (%)", 10, 100, 60, key="whatif_company")
                        if "High dissatisfaction (Pulse)" in trig_series.index:
                            whatif_params["pulse"] = st.selectbox("Pulse", ["High", "Medium", "Low"], index=0, key="whatif_pulse")
                        st.markdown("### Recalculated Predictions with What-If Adjustments")
                        new_scores = []
                        new_triggers_list = []
                        df_bulk_whatif = st.session_state.bulk_result.copy()
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
                        # --- One-to-One Correlations: For every numeric parameter vs What-If Attrition Score
                        st.markdown("<h4 style='color:white;'>One-to-One Correlations with What-If Attrition Score</h4>", unsafe_allow_html=True)
                        numeric_cols = list(df_bulk_whatif.select_dtypes(include=[np.number]).columns)
                        for col in ["What-If Attrition Score", "What-If Negative Triggers", "Prediction Time"]:
                            if col in numeric_cols:
                                numeric_cols.remove(col)
                        for col in numeric_cols:
                            chart = alt.Chart(df_bulk_whatif.dropna(subset=["What-If Attrition Score"])).mark_circle(size=60).encode(
                                x=alt.X(f"{col}:Q", title=col),
                                y=alt.Y("What-If Attrition Score:Q", title="What-If Attrition Score"),
                                tooltip=["Name", col, "What-If Attrition Score"]
                            )
                            st.markdown(graph_header(f"What-If: {col} vs What-If Attrition Score",
                                                     f"A scatter plot showing how {col} correlates with the recalculated What-If Attrition Score."), unsafe_allow_html=True)
                            st.altair_chart(chart, use_container_width=True)
                else:
                    # -------------------------------
                    # Standard Analysis Section
                    # -------------------------------
                    with st.expander("Analysis", expanded=True):
                        analysis_col1, analysis_col2 = st.columns([0.35, 0.65])
                        with analysis_col1:
                            st.subheader("Filters")
                            filter_score_min, filter_score_max = st.slider("Attrition Score Range", 0, 100, (0, 100), key="filter_score")
                            selected_industries = st.multiselect("Filter by Industry",
                                options=st.session_state.bulk_result["Industry"].unique().tolist(),
                                default=st.session_state.bulk_result["Industry"].unique().tolist(), key="filter_ind")
                            selected_company = st.multiselect("Filter by Company Type",
                                options=st.session_state.bulk_result["Company Type"].unique().tolist(),
                                default=st.session_state.bulk_result["Company Type"].unique().tolist(), key="filter_company")
                            filtered_df = st.session_state.bulk_result[
                                (st.session_state.bulk_result["Attrition Score"] >= filter_score_min) &
                                (st.session_state.bulk_result["Attrition Score"] <= filter_score_max) &
                                (st.session_state.bulk_result["Industry"].isin(selected_industries)) &
                                (st.session_state.bulk_result["Company Type"].isin(selected_company))
                            ]
                            st.write("Filtered Bulk Predictions")
                            st.dataframe(filtered_df)
                            st.session_state.filtered_df = filtered_df  # Save for download report.
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
                                for i, custom in enumerate(st.session_state.custom_charts):
                                    display_chart_with_enlarge(custom["chart"], custom["title"], custom["explanation"], key=f"custom_{i}")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df).mark_circle(size=60, color="#4c78a8").encode(
                                    x=alt.X("Employee Age:Q", title="Employee Age"),
                                    y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                    tooltip=["Name", "Employee Age", "Attrition Score", "Industry"]
                                ),
                                "Employee Age vs Attrition Score",
                                "A scatter plot showing how employee age correlates with the predicted attrition risk.",
                                key="scatter1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df).mark_bar(color="#e45756").encode(
                                    x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=20), title="Attrition Score"),
                                    y=alt.Y("count()", title="Frequency")
                                ),
                                "Attrition Score Distribution",
                                "A histogram showing the frequency distribution of predicted attrition risk scores.",
                                key="hist1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df).mark_boxplot(color="#4c78a8").encode(
                                    x=alt.X("Gender:N", title="Gender"),
                                    y=alt.Y("Employee Age:Q", title="Employee Age"),
                                    tooltip=["Gender", "Employee Age"]
                                ),
                                "Employee Age by Gender",
                                "A box plot comparing age distributions across genders.",
                                key="box1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df).mark_circle(size=60, color="#e45756").encode(
                                    x=alt.X("Compa Ratio:Q", title="Compa Ratio"),
                                    y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                    tooltip=["Name", "Compa Ratio", "Attrition Score"]
                                ),
                                "Compa Ratio vs Attrition Score",
                                "A scatter plot exploring the relationship between compensation competitiveness and attrition risk.",
                                key="scatter2")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df[filtered_df.select_dtypes(include=[np.number]).columns.drop(["What-If Attrition Score"], errors="ignore")])
                                .mark_rect().encode(
                                    x=alt.X("index:N", title=""),
                                    y=alt.Y("variable:N", title=""),
                                    color=alt.Color("value:Q", scale=alt.Scale(scheme='redblue')),
                                    tooltip=["index", "variable", "value"]
                                ),
                                "Correlation Heatmap",
                                "A heatmap displaying the correlation among numeric features (unwanted columns removed).",
                                key="heatmap1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df['Industry'].value_counts().reset_index().rename(columns={"index": "Industry", "Industry": "Count"}))
                                .mark_arc().encode(
                                    theta=alt.Theta(field="Count", type="quantitative"),
                                    color=alt.Color(field="Industry", type="nominal"),
                                    tooltip=["Industry", "Count"]
                                ),
                                "Industry Distribution",
                                "A pie chart displaying the distribution of employees across industries.",
                                key="pie1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(compute_trigger_counts(filtered_df, "Negative Triggers").reset_index().rename(columns={"index": "Trigger", 0:"Count"}))
                                .mark_bar(color="#e45756").encode(
                                    x=alt.X("Trigger:N", sort='-y', title="Negative Triggers"),
                                    y=alt.Y("Count:Q", title="Count"),
                                    tooltip=["Trigger", "Count"]
                                ),
                                "Negative Triggers Count",
                                "A bar chart showing how frequently each negative trigger was identified.",
                                key="bar1")
                            
                            display_chart_with_enlarge(
                                alt.Chart(filtered_df).mark_boxplot(color="#4c78a8").encode(
                                    x=alt.X("Industry:N", title="Industry"),
                                    y=alt.Y("Tenure (Months):Q", title="Tenure (Months)"),
                                    tooltip=["Industry", "Tenure (Months)"]
                                ),
                                "Tenure by Industry",
                                "A box plot showing how employee tenure varies across industries.",
                                key="box2")
    show_enlarged_chart()
