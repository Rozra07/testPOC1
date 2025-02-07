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
    st.session_state.nav = "Tabs"  # "Tabs" indicates main UI (i.e., not "My Account")
if "user" not in st.session_state:
    st.session_state.user = {}

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
# Functions for model training / saving / loading
# ----------------------------------------------------
def train_model(training_df, target_column, industry):
    st.write("Training on data shape:", training_df.shape)
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]

    # One-hot encoding of categorical variables
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Model training
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    st.write("Model coefficients:", model.coef_)

    # Save model, scaler, features
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

    # Record industry model details
    update_industry_record(industry, model_filename, scaler_filename, features_filename)

    # Save current global settings to user's profile
    user = st.session_state.user
    user_settings = user.get("settings", {})
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3
    user_settings["bulk_industry_retention"] = {
        ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind == "Tech" else 50)
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
# Trigger Details used in Weighted Logic
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
# Weighted Logic for Probability
# ----------------------------------------------------
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []
    
    # Weighted heuristics
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")
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

    if employee["Compa Ratio"] < 70:
        score += 25
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] < 80:
        score += 20
        extreme_factors += 0.8
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")

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

    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")

    # Amplify if multiple extremes
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
# ML + Weighted Hybrid Prediction
# ----------------------------------------------------
def predict_attrition(row_data, industry):
    model, scaler, feature_columns = load_model(industry)
    if model is None:
        return None, None, None

    # One row
    df_input = pd.DataFrame([row_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(row_data, return_triggers=True)

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers, ml_probability

# ----------------------------------------------------
# Sample CSV Generators
# ----------------------------------------------------
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
# Bulk Prediction Helper
# ----------------------------------------------------
def run_bulk_prediction(df, industry, global_params, apply_adjustments=False, adjustments=None):
    df_result = df.copy()

    scores = []
    triggers_list = []
    ml_probs = []

    for _, row in df_result.iterrows():
        row_data = row.to_dict()

        # Insert global references
        row_data["Average Employee Age"] = global_params["global_avg_age"]
        row_data["Female Employee Ratio"] = global_params["global_female_ratio"]

        # College Tier Retention
        c_tier = str(row_data.get("College Tier", "Tier 3"))
        tier_ret = global_params["tier_retention_dict"].get(c_tier, 40)
        row_data["College Tier Retention"] = tier_ret

        # Industry Retention
        ind_val = str(row_data.get("Industry", "Other"))
        ind_ret = global_params["industry_retention_dict"].get(ind_val, 50)
        row_data["Industry Retention"] = ind_ret

        # Company Type Retention
        comp_type = str(row_data.get("Company Type", "Small Size"))
        comp_ret = global_params["company_retention_dict"].get(comp_type, 50)
        row_data["Company Type Retention"] = comp_ret

        # What-If adjustments
        if apply_adjustments and adjustments is not None:
            if "compa_adjust" in adjustments:
                row_data["Compa Ratio"] = row_data["Compa Ratio"] + adjustments["compa_adjust"]
            if "promo_reduction" in adjustments:
                new_val = row_data["Hasn't been promoted"] - adjustments["promo_reduction"]
                row_data["Hasn't been promoted"] = max(0, new_val)
        
        bulk_score, bulk_trigs, ml_prob = predict_attrition(row_data, industry)

        scores.append(bulk_score)
        ml_probs.append(ml_prob if ml_prob is not None else 0)

        neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
        triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
        triggers_list.append(triggers_str)

    df_result["Attrition Score"] = scores
    df_result["ML Probability"] = ml_probs
    df_result["Negative Triggers"] = triggers_list

    return df_result

def show_risk_distribution(df_result):
    high_risk = (df_result["Attrition Score"] >= 75).sum()
    mod_high = ((df_result["Attrition Score"] >= 60) & (df_result["Attrition Score"] < 75)).sum()
    moderate = ((df_result["Attrition Score"] >= 35) & (df_result["Attrition Score"] < 60)).sum()
    low = (df_result["Attrition Score"] < 35).sum()

    risk_df = pd.DataFrame({
        "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
        "Count": [high_risk, mod_high, moderate, low]
    })
    st.bar_chart(risk_df.set_index("Risk Category"))

def show_triggers_distribution(df_result):
    all_trigs = []
    for val in df_result["Negative Triggers"]:
        if pd.notna(val) and val.strip() != "" and val != "None":
            splitted = [x.strip() for x in val.split(",")]
            all_trigs.extend(splitted)
    if all_trigs:
        trig_series = pd.Series(all_trigs).value_counts()
        st.write("### Top Negative Triggers")
        st.bar_chart(trig_series)
    else:
        st.info("No negative triggers found.")

# ----------------------------------------------------
# Login / Sign Up
# ----------------------------------------------------
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
# Top Header with Title, My Account, Logout
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
# Sidebar: Global Settings & Mode Selection
# ---------------------------------------
if st.session_state.nav != "My Account":
    with st.sidebar:
        mode = st.radio("Select Mode", ["Train Mode", "Test Mode"], index=0, key="main_mode")
        
        # Disable global settings if in Test Mode
        disabled_flag = (mode == "Test Mode")
        st.markdown("### Global Settings for Bulk Analysis")
        
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
                default_val = st.session_state.user.get("settings", {}).get("bulk_industry_retention", {}).get(ind, 60 if ind == "Tech" else 50)
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
    # Account / History
    st.markdown("<div style='text-align: center;'><h2>My Account</h2></div>", unsafe_allow_html=True)
    user = st.session_state.user
    
    st.write("### Account Information")
    st.write(f"**Name:** {user.get('name', '')}")
    st.write(f"**Designation:** {user.get('designation', '')}")
    st.write(f"**Company:** {user.get('company', '')}")
    st.write(f"**Email:** {user.get('email', '')}")

    st.write("### Saved Global Settings")
    user_settings = user.get("settings", {})
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
    # ------------ MAIN MODES ------------
    if st.session_state.main_mode == "Train Mode":
        # TRAIN MODE
        st.header("Train Mode")
        selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")

        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
        with col2:
            st.markdown("### Training File Requirements")
            st.markdown("""
            **Your training file must include:**
            - A binary target column (e.g. `Attrition`, 0/1)
            - Feature columns, for example:
              - `Employee Age`
              - `Gender`
              - `Tenure (Months)`
              - `Pulse` ("High", "Medium", or "Low")
              - `Hasn't been promoted`
              - `Minimum Promotion Cycle`
              - `College Tier`
              - `Industry`
              - `Company Type`
              - `Last Performance Rating`
              - `Compa Ratio`
            """)
            st.download_button(
                label="Download Dummy Training File",
                data=generate_dummy_training_file(),
                file_name="dummy_training_file.csv",
                mime="text/csv"
            )

        target_column = st.text_input("Name of Target Column", value="Attrition")

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

    else:
        # TEST MODE (Bulk Only)
        st.header("Bulk Employee Attrition Prediction")
        st.markdown("""
        **Instructions:**
        - Ensure you have trained a model in Train Mode.
        - Upload a CSV or Excel file with columns:
          - `Name`
          - `Employee Age`
          - `Gender`
          - `Tenure (Months)`
          - `Pulse`
          - `Hasn't been promoted`
          - `Minimum Promotion Cycle`
          - `College Tier`
          - `Industry`
          - `Company Type`
          - `Last Performance Rating`
          - `Compa Ratio`
        """)

        # Gather global parameters
        global_params = {
            "global_avg_age": st.session_state.get("global_avg_age", 35),
            "global_female_ratio": st.session_state.get("global_female_ratio", 40),
            "tier_retention_dict": {
                "Tier 1": st.session_state.get("bulk_tier1", 60),
                "Tier 2": st.session_state.get("bulk_tier2", 50),
                "Tier 3": st.session_state.get("bulk_tier3", 40),
            },
            "industry_retention_dict": {
                ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind == "Tech" else 50)
                for ind in industry_options
            },
            "company_retention_dict": {
                "Startup": st.session_state.get("bulk_startup", 60),
                "Small Size": st.session_state.get("bulk_small", 55),
                "Mid Size": st.session_state.get("bulk_mid", 50),
                "MNC/Giant Company": st.session_state.get("bulk_mnc", 45),
                "Enterprise": 60,  # fallback
                "SME": 50          # fallback
            }
        }

        default_test_industry = st.session_state.get("train_industry", industry_options[0])
        selected_test_industry = st.selectbox(
            "Select Your Industry (for prediction)",
            industry_options,
            index=industry_options.index(default_test_industry) if default_test_industry in industry_options else 0,
            key="test_industry"
        )

        uploaded_file = st.file_uploader("Upload Bulk Data (CSV or Excel)", type=["csv", "xlsx"], key="bulk_file")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_bulk = pd.read_csv(uploaded_file)
                else:
                    df_bulk = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading the file: {e}")
                st.stop()

            st.write("### Preview of Uploaded Data")
            st.dataframe(df_bulk.head())

            required_cols = [
                "Name", "Employee Age", "Gender", "Tenure (Months)", "Pulse",
                "Hasn't been promoted", "Minimum Promotion Cycle", "College Tier",
                "Industry", "Company Type", "Last Performance Rating", "Compa Ratio"
            ]
            missing = [c for c in required_cols if c not in df_bulk.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()

            if st.button("Run Bulk Prediction"):
                # Save event
                save_user_event(
                    st.session_state.user["email"],
                    "bulk_test",
                    {"filename": uploaded_file.name, "industry": selected_test_industry}
                )

                df_result = run_bulk_prediction(
                    df_bulk,
                    selected_test_industry,
                    global_params,
                    apply_adjustments=False
                )

                st.success("✅ Bulk Prediction Completed!")
                st.dataframe(df_result)

                st.write("### Risk Distribution")
                show_risk_distribution(df_result)

                show_triggers_distribution(df_result)

                # What-If Analysis
                what_if_mode = st.checkbox("Activate What-If Analysis (Bulk Scenario)")
                if what_if_mode:
                    st.markdown("## What-If Scenario Analysis")
                    col_left, col_right = st.columns([3, 1])

                    with col_right:
                        st.markdown("### Adjust Global Factors")
                        compa_adjust = st.slider("Adjust All Employees' Compa Ratio by (%)", -50, 50, 0)
                        promo_reduction = st.slider("Reduce 'Hasn't been promoted' by (months)", 0, 12, 0)

                        scenario_adjustments = {
                            "compa_adjust": compa_adjust,
                            "promo_reduction": promo_reduction
                        }

                    with col_left:
                        st.markdown("### Scenario-Based Results")
                        df_what_if = run_bulk_prediction(
                            df_bulk,
                            selected_test_industry,
                            global_params,
                            apply_adjustments=True,
                            adjustments=scenario_adjustments
                        )

                        st.dataframe(df_what_if)
                        st.write("#### Scenario Risk Distribution")
                        show_risk_distribution(df_what_if)
                        show_triggers_distribution(df_what_if)

                    st.info("Uncheck the box to revert to original predictions.")
