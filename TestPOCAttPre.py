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
import streamlit.components.v1 as components

# Set page configuration to wide (full screen)
st.set_page_config(layout="wide")

# ---------------------------------------
# Helper function for safe rerun
# ---------------------------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except AttributeError:
        st.warning("Your version of Streamlit does not support rerun; please refresh manually.")

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
if "show_scenario_form" not in st.session_state:
    st.session_state.show_scenario_form = False
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = {}  # For editing an existing scenario

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
# Mapping for What-If Scenario inputs
# ----------------------------------------------------
trigger_param_map = {
    "Low gender diversity": {"param": "Female Employee Ratio", "type": "slider", "min": 0, "max": 100, "default": 40},
    "Stagnant promotions": {"param": "Hasn't been promoted", "type": "number", "min": 0, "max": 60, "default": 12},
    "Very low performance rating": {"param": "Last Performance Rating", "type": "slider", "min": 1, "max": 5, "default": 1},
    "Low performance rating": {"param": "Last Performance Rating", "type": "slider", "min": 1, "max": 5, "default": 2},
    "Low compensation competitiveness": {"param": "Compa Ratio", "type": "slider", "min": 50, "max": 150, "default": 80},
    "Low college tier retention": {"param": "College Tier Retention", "type": "slider", "min": 10, "max": 100, "default": 40},
    "Low industry retention": {"param": "Industry Retention", "type": "slider", "min": 10, "max": 100, "default": 50},
    "Low company type retention": {"param": "Company Type Retention", "type": "slider", "min": 10, "max": 100, "default": 60},
    "High dissatisfaction (Pulse)": {"param": "Pulse", "type": "selectbox", "options": ["High", "Medium", "Low"], "default": "High"}
}

# ----------------------------------------------------
# Rule-based Attrition Computation Function
# ----------------------------------------------------
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []
    
    # Gender condition
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")
    
    # Promotion condition
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")
    
    # Performance Rating
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
    
    # Compensation Ratio
    if employee["Compa Ratio"] < 80:
        score += 20
        extreme_factors += 0.8
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] < 70:
        score += 25
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")
    
    # Retention Values (from global sliders)
    college_retention = employee.get("College Tier Retention", 40)
    score += (100 - college_retention) * 0.2  # weighted contribution
    if college_retention < 50:
        extreme_factors += 0.5
        triggers.append("Low college tier retention")
    
    industry_retention = employee.get("Industry Retention", 50)
    score += (100 - industry_retention) * 0.2
    if industry_retention < 50:
        extreme_factors += 0.5
        triggers.append("Low industry retention")
    
    company_retention = employee.get("Company Type Retention", 60)
    score += (100 - company_retention) * 0.2
    if company_retention < 50:
        extreme_factors += 0.5
        triggers.append("Low company type retention")
    
    # Pulse condition
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")
    
    # Extreme factor adjustments (extremities)
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
# Predict Attrition Function (ML + Rule-based)
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
            bulk_company_retention = st.session_state.user.get("settings", {}).get("bulk_company_retention", {})
            bulk_startup = st.slider("Startup Retention (%)", 10, 100,
                                     bulk_company_retention.get("Startup", 60),
                                     key="bulk_startup", disabled=disabled_flag)
            bulk_small = st.slider("Small Size Retention (%)", 10, 100,
                                   bulk_company_retention.get("Small Size", 55),
                                   key="bulk_small", disabled=disabled_flag)
            bulk_mid = st.slider("Mid Size Retention (%)", 10, 100,
                                 bulk_company_retention.get("Mid Size", 50),
                                 key="bulk_mid", disabled=disabled_flag)
            bulk_mnc = st.slider("MNC/Giant Company Retention (%)", 10, 100,
                                 bulk_company_retention.get("MNC/Giant Company", 45),
                                 key="bulk_mnc", disabled=disabled_flag)

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
    # In Test Mode, start with industry selection
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
                btn_cols = st.columns(2)
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
                            # Set College Tier Retention based on the employee's tier and the global slider values.
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
                            # Set Industry Retention from global slider values.
                            ind_val = row_dict.get("Industry")
                            row_dict["Industry Retention"] = bulk_industry_retention.get(ind_val, 50)
                            # Set Company Type Retention from global settings.
                            comp_type = row_dict.get("Company Type")
                            if comp_type in st.session_state.user.get("settings", {}).get("bulk_company_retention", {}):
                                row_dict["Company Type Retention"] = st.session_state.user["settings"]["bulk_company_retention"][comp_type]
                            else:
                                row_dict["Company Type Retention"] = 60
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
                    st.info("After bulk prediction, review the results below.")
                
                if st.session_state.bulk_prediction_complete:
                    # Create two equal-width columns
                    left_col, right_col = st.columns(2)
                    
                    with left_col:
                        st.success("✅ Bulk Prediction Completed!")
                        df_display = st.session_state.bulk_result.copy()
                        st.dataframe(df_display)
                        # Risk Distribution Chart for default predictions
                        high_risk = (df_display["Attrition Score"] >= 75).sum()
                        mod_high = ((df_display["Attrition Score"] >= 60) & (df_display["Attrition Score"] < 75)).sum()
                        moderate = ((df_display["Attrition Score"] >= 35) & (df_display["Attrition Score"] < 60)).sum()
                        low = (df_display["Attrition Score"] < 35).sum()
                        risk_df = pd.DataFrame({
                            "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                            "Count": [high_risk, mod_high, moderate, low]
                        })
                        st.write("Risk Distribution")
                        st.bar_chart(risk_df.set_index("Risk Category"))
                        
                        # Pie Chart for Negative Triggers with percentage labels using a light blue palette.
                        trig_series = compute_trigger_counts(df_display, "Negative Triggers")
                        if not trig_series.empty:
                            pie_data = pd.DataFrame({"Trigger": trig_series.index, "Count": trig_series.values})
                            pie_data["Percentage"] = (pie_data["Count"] / pie_data["Count"].sum() * 100).round(1)
                            pie_data["Label"] = pie_data["Percentage"].apply(lambda x: f"{x}%")
                            # Use your specified light color palette.
                            blue_palette = ["#ACDDDE", "#CAF1DE", "#E1F8DC", "#FEF8DD", "#FFE7C7", "#F7D8BA"]
                            chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field="Count", type="quantitative"),
                                color=alt.Color(field="Trigger", type="nominal",
                                                scale=alt.Scale(range=blue_palette),
                                                legend=None)
                            ).properties(width=300, height=300)
                            text = alt.Chart(pie_data).mark_text(
                                align='center',
                                baseline='middle',
                                fontSize=16,
                                fontWeight='bold',
                                color='white'
                            ).encode(
                                theta=alt.Theta(field="Count", type="quantitative"),
                                radius=alt.value(80),
                                text=alt.Text("Label:N")
                            )
                            final_chart = chart + text
                            st.altair_chart(final_chart, use_container_width=True)
                            st.markdown("<small>Hover over the pie chart for more details.</small>", unsafe_allow_html=True)
                        
                        # Drill Down Section
                        st.write("### Drill Down into Individual Employee Details")
                        employee_names = df_display["Name"].tolist()
                        sel_employee = st.selectbox("Select an Employee (by Name)", employee_names, key="drilldown")
                        if sel_employee:
                            selected_row = df_display[df_display["Name"] == sel_employee]
                            st.dataframe(selected_row)
                    
                    with right_col:
                        st.write("## What-If Analysis")
                        enable_what_if = st.checkbox("Enable What-If Analysis", key="enable_what_if")
                        
                        if enable_what_if:
                            # Create a selectbox for saved scenarios (unique key)
                            if st.session_state.saved_scenarios:
                                scenario_names = [s["scenario_name"] for s in st.session_state.saved_scenarios]
                                selected_saved = st.selectbox("Select Saved Scenario", scenario_names, key="saved_scenario_select")
                                if st.button("Edit Selected Scenario"):
                                    for s in st.session_state.saved_scenarios:
                                        if s["scenario_name"] == selected_saved:
                                            st.session_state.current_scenario = s["scenario_inputs"]
                                            st.session_state.current_scenario["scenario_name"] = s["scenario_name"]
                                            st.session_state.show_scenario_form = True
                                            safe_rerun()
                            # Button to create a new scenario if not editing.
                            if not st.session_state.show_scenario_form:
                                if st.button("Create New Scenario"):
                                    st.session_state.current_scenario = {}  # start fresh
                                    st.session_state.show_scenario_form = True
                            
                            if st.session_state.show_scenario_form:
                                st.write("### Scenario Editor")
                                default_scenario_name = f"Scenario {len(st.session_state.saved_scenarios)+1}"
                                scenario_name = st.text_input("Scenario Name", value=st.session_state.current_scenario.get("scenario_name", default_scenario_name), key="scenario_name")
                                
                                # Compute top 4 negative triggers from the bulk results.
                                trigger_counts = compute_trigger_counts(st.session_state.bulk_result, "Negative Triggers")
                                top_triggers = list(trigger_counts.index[:4])
                                st.write("Top Negative Triggers from Bulk Prediction:", top_triggers)
                                
                                # Arrange parameter inputs in two columns. Use saved values if editing.
                                scenario_inputs = {}
                                cols = st.columns(2)
                                i = 0
                                for trigger in top_triggers:
                                    mapping = trigger_param_map.get(trigger, None)
                                    if mapping is not None:
                                        col = cols[i % 2]
                                        label = f"{trigger} ({mapping['param']})"
                                        init_val = st.session_state.current_scenario.get(mapping["param"], mapping["default"])
                                        if mapping["type"] == "slider":
                                            scenario_inputs[mapping["param"]] = col.slider(label, min_value=mapping["min"], max_value=mapping["max"], value=init_val, key=f"scenario_{trigger}")
                                        elif mapping["type"] == "number":
                                            scenario_inputs[mapping["param"]] = col.number_input(label, min_value=mapping["min"], max_value=mapping["max"], value=init_val, key=f"scenario_{trigger}")
                                        elif mapping["type"] == "selectbox":
                                            scenario_inputs[mapping["param"]] = col.selectbox(label, options=mapping["options"], index=mapping["options"].index(init_val) if init_val in mapping["options"] else 0, key=f"scenario_{trigger}")
                                        i += 1
                                
                                # Live update: compute scenario result from current slider values.
                                live_df = st.session_state.bulk_result.copy()
                                new_scores = []
                                for idx, row in live_df.iterrows():
                                    row_dict = row.to_dict()  # convert to dict for pop()
                                    row_dict.pop("Attrition Score", None)
                                    row_dict.pop("Negative Triggers", None)
                                    for param, value in scenario_inputs.items():
                                        row_dict[param] = value
                                    try:
                                        new_score, new_trigs, _ = predict_attrition(row_dict, selected_test_industry)
                                        if new_score is None:
                                            new_score = compute_weighted_attrition(row_dict)
                                    except Exception as e:
                                        new_score = compute_weighted_attrition(row_dict)
                                    new_scores.append(new_score)
                                live_df["What-If Attrition Score"] = new_scores
                                st.write("### Live Scenario Result")
                                st.dataframe(live_df)
                                
                                # What-If Risk Distribution Chart
                                high_risk_w = (live_df["What-If Attrition Score"] >= 75).sum()
                                mod_high_w = ((live_df["What-If Attrition Score"] >= 60) & (live_df["What-If Attrition Score"] < 75)).sum()
                                moderate_w = ((live_df["What-If Attrition Score"] >= 35) & (live_df["What-If Attrition Score"] < 60)).sum()
                                low_w = (live_df["What-If Attrition Score"] < 35).sum()
                                risk_df_w = pd.DataFrame({
                                    "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                                    "Count": [high_risk_w, mod_high_w, moderate_w, low_w]
                                })
                                st.write("### What-If Risk Distribution")
                                st.bar_chart(risk_df_w.set_index("Risk Category"))
                                
                                if st.button("Apply Scenario"):
                                    scenario_details = {
                                        "scenario_name": scenario_name,
                                        "scenario_inputs": scenario_inputs,
                                        "applied_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "num_employees": len(live_df),
                                        "result_df": live_df.to_dict()
                                    }
                                    # If editing, update the existing scenario; otherwise, append.
                                    found = False
                                    for idx, s in enumerate(st.session_state.saved_scenarios):
                                        if s["scenario_name"] == scenario_name:
                                            st.session_state.saved_scenarios[idx] = scenario_details
                                            found = True
                                            break
                                    if not found:
                                        st.session_state.saved_scenarios.append(scenario_details)
                                    save_user_event(st.session_state.user["email"], "what_if_scenario", scenario_details)
                                    st.success(f"Scenario '{scenario_name}' saved.")
                                    st.session_state.current_scenario = {}
                                    st.session_state.show_scenario_form = False
                            
                            # Display saved scenario details if available.
                            if st.session_state.saved_scenarios:
                                selected_saved = st.selectbox("Select Saved Scenario", [s["scenario_name"] for s in st.session_state.saved_scenarios], key="saved_scenario_display")
                                for s in st.session_state.saved_scenarios:
                                    if s["scenario_name"] == selected_saved:
                                        saved_df = pd.DataFrame(s["result_df"])
                                        st.write(f"### Result for {selected_saved}")
                                        st.dataframe(saved_df)
                                        high_risk_saved = (saved_df["What-If Attrition Score"] >= 75).sum()
                                        mod_high_saved = ((saved_df["What-If Attrition Score"] >= 60) & (saved_df["What-If Attrition Score"] < 75)).sum()
                                        moderate_saved = ((saved_df["What-If Attrition Score"] >= 35) & (saved_df["What-If Attrition Score"] < 60)).sum()
                                        low_saved = (saved_df["What-If Attrition Score"] < 35).sum()
                                        risk_df_saved = pd.DataFrame({
                                            "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                                            "Count": [high_risk_saved, mod_high_saved, moderate_saved, low_saved]
                                        })
                                        st.write("### Saved Scenario Risk Distribution")
                                        st.bar_chart(risk_df_saved.set_index("Risk Category"))
                                        break
                        else:
                            st.info("Enable What-If Analysis to use scenario features.")
        else:
            st.info("Please upload a bulk data file to begin analysis.")
