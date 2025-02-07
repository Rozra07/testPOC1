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

# Set page configuration to wide (full screen)
st.set_page_config(layout="wide")

# -------------------------------
# Helper functions for safe conversion
# -------------------------------
def safe_float(val, default):
    try:
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)

def safe_int(val, default):
    try:
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)

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
    st.session_state.nav = "Tabs"  # "Tabs" indicates main UI (i.e. not "My Account")
if "user" not in st.session_state:
    st.session_state.user = {}
if "bulk_prediction_complete" not in st.session_state:
    st.session_state.bulk_prediction_complete = False
if "bulk_result" not in st.session_state:
    st.session_state.bulk_result = None
if "enable_what_if" not in st.session_state:
    st.session_state.enable_what_if = False

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
    
    # Save industry record
    update_industry_record(industry, model_filename, scaler_filename, features_filename)
    
    # Save global settings to user record.
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
            df = df.append(record, ignore_index=True)
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
            "lack_female_applicants": "We are not getting enough female applicants",
            "lack_female_mentors": "We have few female mentors or leaders",
            "rigid_policies": "We do not offer flexible policies (e.g., maternity, remote, etc.)"
        },
        "solutions": {
            "lack_female_applicants": (
                "- **Partner with Women’s Universities** or female‑oriented professional groups.\n"
                "- **Highlight DEI** in your recruitment materials."
            ),
            "lack_female_mentors": (
                "- **Implement formal mentorship** programs.\n"
                "- **Sponsor leadership development** for existing female employees."
            ),
            "rigid_policies": (
                "- Introduce **flexible working hours** and remote/hybrid options.\n"
                "- Improve **maternity/paternity benefits** and family‑friendly leave."
            )
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are unclear or inconsistent",
            "no_mentorship": "No proper mentorship or upskilling tracks exist",
            "bureaucratic_structure": "The organization structure is too bureaucratic"
        },
        "solutions": {
            "unclear_criteria": (
                "- **Publish transparent promotion guidelines** linked to clear KPIs.\n"
                "- Provide employees with **regular promotion readiness feedback**."
            ),
            "no_mentorship": (
                "- Launch **formal mentoring** or buddy programs.\n"
                "- Offer **upskilling opportunities** and learning stipends."
            ),
            "bureaucratic_structure": (
                "- **Streamline decision‑making** or reduce hierarchical layers.\n"
                "- Consider more **agile or cross‑functional** teams to encourage skill growth."
            )
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1‑on‑1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        },
        "solutions": {
            "misaligned_role": (
                "- **Clarify job responsibilities** and set SMART goals.\n"
                "- Ensure roles align with employees’ **strengths** and career aspirations."
            ),
            "no_feedback": (
                "- Implement **frequent 1‑on‑1 check‑ins** and agile feedback loops.\n"
                "- Use **performance dashboards** for real‑time updates."
            ),
            "skill_gaps": (
                "- Provide **targeted training** and eLearning modules.\n"
                "- Offer **certification reimbursements** and skill‑building workshops."
            )
        }
    },
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1‑on‑1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        },
        "solutions": {
            "misaligned_role": (
                "- **Clarify job responsibilities** and set SMART goals.\n"
                "- Align roles with employees’ **strengths** and preferences."
            ),
            "no_feedback": (
                "- Implement **regular 1‑on‑1 check‑ins**.\n"
                "- Provide ongoing **coaching and feedback** rather than annual appraisals."
            ),
            "skill_gaps": (
                "- Offer **targeted training** in needed skill areas.\n"
                "- Encourage **peer‑to‑peer learning** or cross‑functional rotations."
            )
        }
    },
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary is below market rates",
            "minimal_bonus": "Bonuses or variable pay are minimal or non‑existent",
            "poor_benefits": "Benefits package is lacking (insurance, retirement, etc.)"
        },
        "solutions": {
            "below_market": (
                "- **Conduct market benchmarking** to adjust salaries to median or above.\n"
                "- Consider **geographic pay differentials** if applicable."
            ),
            "minimal_bonus": (
                "- Introduce **performance‑based incentives** or profit‑sharing.\n"
                "- Evaluate **RSUs (Restricted Stock Units)** or equity grants for retention."
            ),
            "poor_benefits": (
                "- Offer **competitive health insurance**, retirement contributions.\n"
                "- Provide **flexible schedules**, wellness programs, and other perks."
            )
        }
    },
    "Low college tier retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among certain colleges or entry‑level hires",
            "mismatch_culture": "Mismatch between background and company culture",
            "poor_onboarding": "Insufficient onboarding or assimilation for these hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Investigate root causes via **exit interviews**.\n"
                "- Build **campus ambassador** programs to attract the right fit."
            ),
            "mismatch_culture": (
                "- Provide better **orientation** on company culture.\n"
                "- Pair new hires with **mentors** from similar backgrounds."
            ),
            "poor_onboarding": (
                "- Enhance **onboarding programs** with structured check‑ins (30/60/90 days).\n"
                "- Offer a **buddy system** for new graduates."
            )
        }
    },
    "Low industry retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from this industry",
            "mismatch_culture": "Mismatch between industry norms and your company's culture",
            "poor_onboarding": "Insufficient onboarding for these lateral hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Conduct **benchmarking** to see if salaries and roles align with industry standards.\n"
                "- Explore **targeted retention strategies** (mentorship, training)."
            ),
            "mismatch_culture": (
                "- Emphasize **company values** and create inclusive teams.\n"
                "- Have **town halls** or Q&A sessions for lateral hires to assimilate."
            ),
            "poor_onboarding": (
                "- Develop **structured assimilation** for mid‑career folks.\n"
                "- Provide a **transition buddy** who understands both industries."
            )
        }
    },
    "Low company type retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from certain company backgrounds",
            "mismatch_culture": "Mismatch between prior company culture and current environment",
            "poor_onboarding": "Onboarding doesn’t address differences in processes, tools, or structures"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Identify if certain **company backgrounds** always churn quickly.\n"
                "- Adapt your onboarding or project assignments accordingly."
            ),
            "mismatch_culture": (
                "- Provide **culture assimilation** sessions or manager training.\n"
                "- Encourage **peer networking** to help them adapt faster."
            ),
            "poor_onboarding": (
                "- Have a **comprehensive onboarding** covering your processes & tools.\n"
                "- Assign **buddies** who previously transitioned from similar backgrounds."
            )
        }
    },
    "High dissatisfaction (Pulse)": {
        "subproblems": {
            "work_life_imbalance": "Work‑life imbalance or excessive workload",
            "poor_manager_relationships": "Employees feel managers are unsupportive",
            "limited_growth": "Limited growth or recognition opportunities"
        },
        "solutions": {
            "work_life_imbalance": (
                "- Offer **flexible scheduling** and **mental health** resources.\n"
                "- Encourage **healthy boundaries** around work hours."
            ),
            "poor_manager_relationships": (
                "- Train managers on **emotional intelligence** and communication.\n"
                "- Collect **360‑degree feedback** to identify manager blind spots."
            ),
            "limited_growth": (
                "- Implement **career development** paths and internal mobility.\n"
                "- Recognize achievements publicly and **reward** top performers."
            )
        }
    }
}

# ----------------------------------------------------
# Rule-based attrition computation
# Reintroducing the "Hasn't been promoted" and "Minimum Promotion Cycle" factors.
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
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
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
# Function to compute trigger counts from a column with comma-separated triggers
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
    # In Test Mode, start immediately with industry selection after header.
    if st.session_state.main_mode == "Test Mode":
        selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=0, key="test_industry")
        st.markdown("""
        <div class="tooltip">Read Instructions
          <span class="tooltiptext">
            Ensure that you have trained a model in Train Mode.
            <br><br>
            Upload a CSV or Excel file with the following columns:
            <br> - Name
            <br> - Employee Age
            <br> - Gender
            <br> - Tenure (Months)
            <br> - Pulse
            <br> - Hasn't been promoted
            <br> - Minimum Promotion Cycle
            <br> - College Tier
            <br> - Industry
            <br> - Company Type
            <br> - Last Performance Rating
            <br> - Compa Ratio
            <br><br>
            Note: The test data does not require an Attrition column.
          </span>
        </div>
        <style>
        .tooltip {
          position: relative;
          display: inline-block;
          cursor: pointer;
          font-weight: bold;
          color: #0073e6;
        }
        .tooltip .tooltiptext {
          visibility: hidden;
          width: 300px;
          background-color: #f9f9f9;
          color: #333;
          text-align: left;
          border-radius: 6px;
          padding: 10px;
          position: absolute;
          z-index: 1;
          top: 125%;
          left: 50%;
          margin-left: -150px;
          box-shadow: 0px 0px 6px 0px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
          visibility: visible;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        selected_test_industry = None

    if st.session_state.main_mode == "Train Mode":
        st.header("Train Mode")
        selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
        with col2:
            st.markdown("### Detailed Guide for Training File")
            st.markdown("""
            **Your training file must include:**
            - A **target column** (e.g., Attrition – use binary values 0/1, where **0: Active Employee** and **1: Non‑Active Employee**).
            - **Feature columns:**  
              - Employee Age  
              - Gender (e.g., "Male", "Female")  
              - Tenure (Months)  
              - Pulse (e.g., "High", "Medium", "Low")  
              - Hasn't been promoted  
              - Minimum Promotion Cycle  
              - College Tier (e.g., "Tier 1", "Tier 2", "Tier 3")  
              - Industry (e.g., "Tech", "Finance", etc.)  
              - Company Type (e.g., "Startup", "Enterprise", etc.)  
              - Last Performance Rating (e.g., 1 to 5)  
              - Compa Ratio (compensation ratio)
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
            if st.button("Train Model"):
                train_model(train_df, target_column, selected_train_industry)
    else:  # Test Mode - Bulk Analysis Only
        st.header("Bulk Employee Attrition Prediction")
        uploaded_file = st.file_uploader("Upload Bulk Data (CSV or Excel)", type=["csv", "xlsx"], key="bulk_file")
        if uploaded_file is not None:
            try:
                df_bulk = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading the file: {e}")
                st.stop()
            st.write("### Uploaded Data Preview:")
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
                btn_cols = st.columns([1,1])
                with btn_cols[0]:
                    if st.button("🚀 Run Bulk Prediction"):
                        scores = []
                        triggers_list = []
                        names = []
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
                            row_dict["Company Type Retention"] = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60)
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
                        save_user_event(st.session_state.user["email"], "bulk_prediction", {"rows": len(df_bulk)})
                with btn_cols[1]:
                    if st.session_state.bulk_prediction_complete:
                        st.session_state.enable_what_if = st.checkbox("Enable What-If Analysis", key="whatif_toggle", value=False)
                
                if st.session_state.bulk_prediction_complete:
                    df_bulk = st.session_state.bulk_result
                    # Divide screen into two columns: Left (Original Results) and Right (What-If Analysis) only if What-If is enabled.
                    left_col, right_col = st.columns(2)
                    
                    with left_col:
                        st.success("✅ Bulk Prediction Completed!")
                        st.dataframe(df_bulk)
                        high_risk = (df_bulk["Attrition Score"] >= 75).sum()
                        mod_high = ((df_bulk["Attrition Score"] >= 60) & (df_bulk["Attrition Score"] < 75)).sum()
                        moderate = ((df_bulk["Attrition Score"] >= 35) & (df_bulk["Attrition Score"] < 60)).sum()
                        low = (df_bulk["Attrition Score"] < 35).sum()
                        risk_df = pd.DataFrame({
                            "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                            "Count": [high_risk, mod_high, moderate, low]
                        })
                        st.write("### Risk Distribution")
                        st.bar_chart(risk_df.set_index("Risk Category"))
                        
                        trig_series = compute_trigger_counts(df_bulk, "Negative Triggers")
                        if not trig_series.empty:
                            pie_data = pd.DataFrame({"Trigger": trig_series.index, "Count": trig_series.values})
                            pie_data["Percentage"] = (pie_data["Count"] / pie_data["Count"].sum() * 100).round(1)
                            pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field="Count", type="quantitative"),
                                color=alt.Color(field="Trigger", type="nominal",
                                                scale=alt.Scale(range=["#AEC6CF", "#FFD1DC", "#C3B1E1", "#FDFD96", "#77DD77", "#B19CD9", "#FFB347"]),
                                                legend=None)
                            ).properties(width=300, height=300)
                            st.write("### Overall Negative Triggers (Pie Chart)")
                            st.altair_chart(pie_chart, use_container_width=True)
                            
                            # Display legend as a table with colored boxes
                            legend_html = "<table><tr><th>Trigger</th><th>Percentage</th><th>Color</th></tr>"
                            color_scale = ["#AEC6CF", "#FFD1DC", "#C3B1E1", "#FDFD96", "#77DD77", "#B19CD9", "#FFB347"]
                            for i, row in pie_data.iterrows():
                                legend_html += f"<tr><td>{row['Trigger']}</td><td><b>{row['Percentage']}%</b></td><td style='background-color:{color_scale[i]}; width:50px;'>&nbsp;</td></tr>"
                            legend_html += "</table>"
                            st.markdown(legend_html, unsafe_allow_html=True)
                        
                        st.write("### Drill Down into Individual Employee Details")
                        st.markdown("""
                        <style>
                        div[data-baseweb="table"] {
                            min-width: 1000px !important;
                            overflow-x: auto;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        employee_names = df_bulk["Name"].tolist()
                        sel_employee = st.selectbox("Select an Employee (by Name)", employee_names, key="drilldown")
                        if sel_employee:
                            selected_row = df_bulk[df_bulk["Name"] == sel_employee]
                            st.dataframe(selected_row)
                        
                        if not trig_series.empty:
                            selected_trigger = st.selectbox("Select a Negative Trigger for Solutions", options=trig_series.index.tolist())
                            if selected_trigger in TRIGGER_DETAILS:
                                st.markdown(f"### Recommended Solutions for {selected_trigger}")
                                for key, sol in TRIGGER_DETAILS[selected_trigger]["solutions"].items():
                                    st.markdown(sol)
                            else:
                                st.info("No solutions available for the selected trigger.")
                        else:
                            selected_trigger = None
                    
                    if st.session_state.enable_what_if:
                        with right_col:
                            st.markdown(
                                """
                                <div style="background-color:#e6f7ff; padding:20px; border-radius:10px;">
                                <h2 style="color:#005c99;">What-If Analysis</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            with st.container():
                                whatif_params = {}
                                if "Low gender diversity" in trig_series.index:
                                    whatif_params["female_ratio"] = st.slider("Women % in Organization", 0, 100, global_female_ratio, key="whatif_female")
                                if "Stagnant promotions" in trig_series.index:
                                    default_not_promoted = int(df_bulk["Hasn't been promoted"].mean() or 0)
                                    default_min_cycle = int(df_bulk["Minimum Promotion Cycle"].mean() or 24)
                                    whatif_params["not_promoted"] = st.slider("Months Since Last Promotion", 0, 60, default_not_promoted, key="whatif_not_promoted")
                                    whatif_params["min_cycle"] = st.slider("Minimum Promotion Cycle", 12, 60, default_min_cycle, key="whatif_min_cycle")
                                if any(x in trig_series.index for x in ["Very low performance rating", "Low performance rating"]):
                                    default_rating = int(df_bulk["Last Performance Rating"].mean() or 3)
                                    whatif_params["rating"] = st.selectbox("Last Performance Rating", [1, 2, 3, 4, 5], index=max(0, default_rating-1), key="whatif_rating")
                                if any(x in trig_series.index for x in ["Low compensation competitiveness", "High compensation ratio"]):
                                    default_compa = int(df_bulk["Compa Ratio"].mean() or 90)
                                    whatif_params["compa_ratio"] = st.slider("Compa Ratio (%)", 50, 150, default_compa, key="whatif_compa")
                                if "Low college tier retention" in trig_series.index:
                                    whatif_params["tier1"] = st.slider("Tier 1 Retention (%)", 10, 100, bulk_tier1, key="whatif_tier1")
                                    whatif_params["tier2"] = st.slider("Tier 2 Retention (%)", 10, 100, bulk_tier2, key="whatif_tier2")
                                    whatif_params["tier3"] = st.slider("Tier 3 Retention (%)", 10, 100, bulk_tier3, key="whatif_tier3")
                                if "Low industry retention" in trig_series.index:
                                    avg_ind = int(np.mean(list(bulk_industry_retention.values())))
                                    whatif_params["industry_retention"] = st.slider("Industry Retention (%)", 10, 100, avg_ind, key="whatif_industry")
                                if "Low company type retention" in trig_series.index:
                                    default_company = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60)
                                    whatif_params["company_retention"] = st.slider("Company Type Retention (%)", 10, 100, default_company, key="whatif_company")
                                if "High dissatisfaction (Pulse)" in trig_series.index:
                                    whatif_params["pulse"] = st.selectbox("Pulse", ["High", "Medium", "Low"], index=0, key="whatif_pulse")
                                
                                st.write("### Recalculated Predictions with What-If Adjustments")
                                new_scores = []
                                new_triggers_list = []
                                df_bulk_whatif = df_bulk.copy()
                                for idx, row in df_bulk_whatif.iterrows():
                                    new_row = dict(row)
                                    new_row["Average Employee Age"] = safe_float(global_avg_age, global_avg_age)
                                    new_row["Female Employee Ratio"] = safe_float(whatif_params.get("female_ratio", new_row.get("Female Employee Ratio", global_female_ratio)), global_female_ratio)
                                    new_row["Hasn't been promoted"] = safe_float(whatif_params.get("not_promoted", new_row.get("Hasn't been promoted", 0)), 0)
                                    new_row["Minimum Promotion Cycle"] = safe_float(whatif_params.get("min_cycle", new_row.get("Minimum Promotion Cycle", 24)), 24)
                                    new_row["Last Performance Rating"] = safe_int(whatif_params.get("rating", new_row.get("Last Performance Rating", 3)), 3)
                                    new_row["Compa Ratio"] = safe_float(whatif_params.get("compa_ratio", new_row.get("Compa Ratio", 90)), 90)
                                    if new_row.get("College Tier") == "Tier 1":
                                        new_row["College Tier Retention"] = safe_float(whatif_params.get("tier1", new_row.get("College Tier Retention", bulk_tier1)), bulk_tier1)
                                    elif new_row.get("College Tier") == "Tier 2":
                                        new_row["College Tier Retention"] = safe_float(whatif_params.get("tier2", new_row.get("College Tier Retention", bulk_tier2)), bulk_tier2)
                                    elif new_row.get("College Tier") == "Tier 3":
                                        new_row["College Tier Retention"] = safe_float(whatif_params.get("tier3", new_row.get("College Tier Retention", bulk_tier3)), bulk_tier3)
                                    new_row["Industry Retention"] = safe_float(whatif_params.get("industry_retention", new_row.get("Industry Retention", 60)), 60)
                                    new_row["Company Type Retention"] = safe_float(whatif_params.get("company_retention", new_row.get("Company Type Retention", 60)), 60)
                                    new_row["Pulse"] = whatif_params.get("pulse", new_row.get("Pulse"))
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
                                st.write("### What-If Risk Distribution")
                                st.bar_chart(risk_df_w.set_index("Risk Category"))
            else:
                st.info("Please upload a bulk data file to begin analysis.")
