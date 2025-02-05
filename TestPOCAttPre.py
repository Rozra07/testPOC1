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

#########################################
# Helper functions for user storage
#########################################

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

#########################################
# Global: Expanded Industry Options
#########################################
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]

#########################################
# Functions for model training/prediction
#########################################

def update_aggregated_training_data(industry, new_data_df):
    filename = f"training_data_{industry}.csv"
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
        except Exception as e:
            st.error(f"Error reading aggregated training data: {e}")
            existing_df = pd.DataFrame()
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df
    combined_df.to_csv(filename, index=False)
    return combined_df

def train_model(training_df, target_column, industry):
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)
    
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
    # Compute training accuracy as a proxy for model confidence.
    training_accuracy = model.score(X_scaled, y) * 100
    st.info(f"Training Accuracy (Confidence): {training_accuracy:.2f}%")
    
    update_industry_record(industry, model_filename, scaler_filename, features_filename)
    
    # Save global settings to user record.
    user = st.session_state.user
    # Use safe retrieval for settings:
    user_settings = user.get("settings") or {}
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3
    user_settings["bulk_industry_retention"] = {ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind=="Tech" else 50) for ind in industry_options}
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

#########################################
# Ensure user data exists in session_state
#########################################
if "user" not in st.session_state:
    st.session_state.user = None

#########################################
# Login/Sign Up System
#########################################
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
    if not st.session_state.logged_in:
        st.stop()

#########################################
# Top Header with Title, My Account Icon, and Logout
#########################################
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

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

if "nav" not in st.session_state:
    st.session_state.nav = "Tabs"  # "Tabs" means Train/Test mode

#########################################
# Main Navigation
#########################################
if st.session_state.nav == "My Account":
    st.header("My Account")
    user = st.session_state.user
    st.write("### Account Information")
    st.write(f"**Name:** {user.get('name', '')}")
    st.write(f"**Designation:** {user.get('designation', '')}")
    st.write(f"**Company:** {user.get('company', '')}")
    st.write(f"**Email:** {user.get('email', '')}")
    
    st.write("### Saved Global Settings")
    user_settings = (st.session_state.user.get("settings") or {})
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
    tabs = st.tabs(["Train Mode", "Test Mode"])
    
    with tabs[0]:
        st.header("Train Mode")
        selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
        with col2:
            st.markdown("### Detailed Guide for Training File")
            st.markdown("""
            **Your training file must include:**
            - A **target column** (e.g., `Attrition` – use binary values 0/1, where **0: Active Employee** and **1: Non‑Active Employee**).
            - **Feature columns:**  
              - `Employee Age`  
              - `Gender` (e.g., "Male", "Female")  
              - `Tenure (Months)`  
              - `Pulse` (e.g., "High", "Medium", "Low")  
              - `Hasn't been promoted` (months since last promotion)  
              - `Minimum Promotion Cycle` (in months)  
              - `College Tier` (e.g., "Tier 1", "Tier 2", "Tier 3")  
              - `Industry` (e.g., "Tech", "Finance", etc.)  
              - `Company Type` (e.g., "Startup", "Enterprise", etc.)  
              - `Last Performance Rating` (e.g., 1 to 5)  
              - `Compa Ratio` (compensation ratio)
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
            
            if st.button("Update Aggregated Data and Retrain Model"):
                aggregated_df = update_aggregated_training_data(selected_train_industry, train_df)
                st.write("### Aggregated Training Data Preview")
                st.dataframe(aggregated_df.head())
                train_model(aggregated_df, target_column, selected_train_industry)
    
    with tabs[1]:
        st.header("Test Mode")
        st.markdown("""
        **Instructions for Testing:**
        
        - Ensure that you have trained a model in Train Mode.
        - The industry selection below is pre‑set to your training industry.
        - In Single Employee mode, enter the employee details (including retention percentages).
        - In Bulk Employees mode, upload a CSV or Excel file with the following columns:
             - **Name**
             - **Employee Age**
             - **Gender**
             - **Tenure (Months)**
             - **Pulse**
             - **Hasn't been promoted**
             - **Minimum Promotion Cycle**
             - **College Tier** *(e.g., "Tier 1", "Tier 2", "Tier 3")*
             - **Industry** *(e.g., "Tech", "Finance", etc.)*
             - **Company Type** *(e.g., "Startup", "Small Size", "Mid Size", "MNC/Giant Company")*
             - **Last Performance Rating**
             - **Compa Ratio**
             
             *Note: The global settings you defined in Train Mode (for Average Employee Age, Women % etc.) will be applied automatically to all bulk data.*
        """)
        default_test_industry = st.session_state.get("train_industry", industry_options[0])
        selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=industry_options.index(default_test_industry) if default_test_industry in industry_options else 0, key="test_industry")
        
        test_mode_option = st.selectbox("Select Test Mode", ["Single Employee", "Bulk Employees"])
        
        if test_mode_option == "Single Employee":
            if "prediction_made" not in st.session_state:
                st.session_state.prediction_made = False
            if "score" not in st.session_state:
                st.session_state.score = None
            if "triggers" not in st.session_state:
                st.session_state.triggers = []
            if "employee_data" not in st.session_state:
                st.session_state.employee_data = {}
        
            st.write("### Enter Employee / Company Details")
            with st.form("attrition_form"):
                input_data = {
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
                submit_single = st.form_submit_button("🚀 Predict")
                if submit_single:
                    final_score, triggers, ml_confidence = predict_attrition(input_data, selected_test_industry)
                    st.session_state.score = final_score
                    st.session_state.triggers = triggers
                    st.session_state.prediction_made = True
                    st.session_state.employee_data = input_data
                    save_user_event(st.session_state.user["email"], "test_single", {"input_data": input_data, "result": final_score})
        
            if st.session_state.prediction_made:
                score = st.session_state.score
                triggers = st.session_state.triggers
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
                    st.markdown(f"**Model Confidence:** {ml_confidence:.2f}%")
        
                col_left, col_right = st.columns(2)
                with col_left:
                    st.write("### Key Contributing Factors")
                    negative_triggers = []
                    for t in triggers:
                        if t in TRIGGER_DETAILS:
                            negative_triggers.append(t)
                            st.markdown(f"- **{t}**")
                    if not negative_triggers:
                        st.markdown("*No major negative triggers identified.*")
        
                    st.write("### Sub-Problems Selection")
                    sub_problem_selections = {}
                    for trig in negative_triggers:
                        st.write(f"**{trig}**")
                        subprobs = TRIGGER_DETAILS[trig]["subproblems"]
                        chosen_list = []
                        for sub_key, sub_label in subprobs.items():
                            chk_id = f"{trig}-{sub_key}"
                            if chk_id not in st.session_state:
                                st.session_state[chk_id] = False
                            new_val = st.checkbox(sub_label, key=chk_id)
                            if new_val:
                                chosen_list.append(sub_key)
                        sub_problem_selections[trig] = chosen_list
        
                    if st.button("💡 Show Solutions"):
                        st.write("### Recommended Solutions")
                        any_chosen = False
                        for trig in negative_triggers:
                            chosen_subs = sub_problem_selections[trig]
                            if chosen_subs:
                                any_chosen = True
                                st.write(f"**Trigger:** {trig}")
                                for sub_key in chosen_subs:
                                    solution_text = TRIGGER_DETAILS[trig]["solutions"].get(sub_key, "")
                                    st.markdown(f"**Sub-Problem: {sub_key}**")
                                    st.markdown(f"{solution_text}")
                        if not any_chosen:
                            st.info("No sub-problems selected, so no solutions to display.")
        
                with col_right:
                    st.write("### What-If Scenario Planning")
                    scenario_data = dict(st.session_state.employee_data)
                    scenario_data["Compa Ratio"] = st.slider(
                        "Compa Ratio (%) [Scenario]",
                        50, 150, scenario_data["Compa Ratio"],
                        help="Adjust to see how risk changes if compensation changes."
                    )
                    scenario_data["Last Performance Rating"] = st.slider(
                        "Last Performance Rating [Scenario]",
                        1, 5, scenario_data["Last Performance Rating"],
                        help="What if performance improves (higher rating) or worsens?"
                    )
                    scenario_data["Pulse"] = st.radio(
                        "Pulse (Employee dissatisfaction) [Scenario]",
                        ["High", "Medium", "Low"],
                        index=["High", "Medium", "Low"].index(scenario_data["Pulse"]),
                        horizontal=True
                    )
                    scenario_score, scenario_triggers, _ = predict_attrition(scenario_data, selected_test_industry)
                    st.write(f"**Scenario Attrition Risk:** {scenario_score:.2f}%")
                    diff = scenario_score - score
                    if diff > 0:
                        st.markdown(f"<span style='color:red;'>Risk +{diff:.2f}% higher than original.</span>", unsafe_allow_html=True)
                    elif diff < 0:
                        st.markdown(f"<span style='color:green;'>Risk {diff:.2f}% lower than original.</span>", unsafe_allow_html=True)
                    else:
                        st.write("No change from original risk.")
                    neg_scenario_triggers = [t for t in scenario_triggers if t in TRIGGER_DETAILS]
                    if neg_scenario_triggers:
                        st.write("### Scenario Negative Triggers")
                        for t in neg_scenario_triggers:
                            st.markdown(f"- **{t}**")
                    else:
                        st.markdown("*No negative triggers in this scenario.*")
        
        elif test_mode_option == "Bulk Employees":
            st.header("Bulk Employee Attrition Prediction")
            st.markdown("""
            **Instructions:**
            
            Upload a CSV or Excel file with the following columns:
             - **Name**
             - **Employee Age**
             - **Gender**
             - **Tenure (Months)**
             - **Pulse**
             - **Hasn't been promoted**
             - **Minimum Promotion Cycle**
             - **College Tier** *(e.g., "Tier 1", "Tier 2", "Tier 3")*
             - **Industry** *(e.g., "Tech", "Finance", etc.)*
             - **Company Type** *(e.g., "Startup", "Small Size", "Mid Size", "MNC/Giant Company")*
             - **Last Performance Rating**
             - **Compa Ratio**
             
             *Note: The global settings you defined in Train Mode (for Average Employee Age, Women % etc.) will be applied automatically to all bulk data.*
            """)
            global_avg_age = st.session_state.get("global_avg_age", 35)
            global_female_ratio = st.session_state.get("global_female_ratio", 40)
            bulk_tier1 = st.session_state.get("bulk_tier1", 60)
            bulk_tier2 = st.session_state.get("bulk_tier2", 50)
            bulk_tier3 = st.session_state.get("bulk_tier3", 40)
            bulk_industry_retention = {}
            for ind in industry_options:
                bulk_industry_retention[ind] = st.session_state.get(f"bulk_ind_{ind}", 60 if ind=="Tech" else 50)
            default_company_retention = st.session_state.get("retention_company", 60)
        
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
                    if st.button("🚀 Run Bulk Prediction"):
                        scores = []
                        triggers_list = []
                        names = []
                        for idx, row in df_bulk.iterrows():
                            row_dict = row.to_dict()
                            names.append(row_dict.get("Name"))
                            # Apply global settings for Average Employee Age and Female Employee Ratio:
                            row_dict["Average Employee Age"] = global_avg_age
                            row_dict["Female Employee Ratio"] = global_female_ratio
                            # Use global retention settings:
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
                            comp_type = row_dict.get("Company Type")
                            row_dict["Company Type Retention"] = default_company_retention
        
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
                        all_trigs = []
                        for val in df_bulk["Negative Triggers"]:
                            if pd.notna(val) and val.strip() != "" and val != "None":
                                splitted = [x.strip() for x in val.split(",")]
                                all_trigs.extend(splitted)
                        if all_trigs:
                            trig_series = pd.Series(all_trigs).value_counts()
                            st.write("### Top Negative Triggers")
                            st.bar_chart(trig_series)
                        else:
                            st.info("No negative triggers found across the batch.")
                        st.write("### Drill Down into Individual Rows")
                        df_bulk_reset = df_bulk.reset_index(drop=True)
                        row_options = list(range(len(df_bulk_reset)))
                        sel_row = st.selectbox("Select an Employee (Row Index)", row_options)
                        if sel_row is not None:
                            row_info = df_bulk_reset.loc[sel_row].to_dict()
                            st.write("### Row Data:")
                            st.json(row_info)

elif st.session_state.nav == "My Account":
    st.header("My Account")
    user = st.session_state.user
    st.write("### Account Information")
    st.write(f"**Name:** {user.get('name', '')}")
    st.write(f"**Designation:** {user.get('designation', '')}")
    st.write(f"**Company:** {user.get('company', '')}")
    st.write(f"**Email:** {user.get('email', '')}")
    
    st.write("### Saved Global Settings")
    user_settings = (user.get("settings") or {})
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
