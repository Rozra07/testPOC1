import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# GLOBALS
# =============================================================================
industry_options = ["Tech", "Finance", "Healthcare", "Education", "Manufacturing", "Other"]

# List of available model options for training/analysis.
# The "Logistic Regression (Recommended)" is indicated as the recommended model.
model_options_display = [
    "Logistic Regression (Recommended)",
    "Random Forest",
    "Support Vector Machine",
    "XGBoost"
]

# A mapping from display name to internal model code
model_options_map = {
    "Logistic Regression (Recommended)": "logistic_regression",
    "Random Forest": "random_forest",
    "Support Vector Machine": "svm",
    "XGBoost": "xgboost"
}

# =============================================================================
# MODEL SELECTION UTILITY
# =============================================================================
def get_model(model_type):
    """Return a new model instance based on the model_type string."""
    if model_type == "logistic_regression":
        return LogisticRegression(solver="liblinear", random_state=42)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)
    elif model_type == "svm":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=42)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        st.error("Unknown model type selected.")
        return None

# =============================================================================
# AGGREGATED TRAINING DATA UPDATE
# =============================================================================
def update_aggregated_training_data(industry, new_data_df):
    """
    Updates (or creates) an aggregated training file for the given industry.
    If a file (e.g., training_data_Tech.csv) exists, the new data is appended;
    otherwise, a new file is created.
    Returns the combined aggregated DataFrame.
    """
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

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_model(training_df, target_column, industry, model_type):
    """
    Trains a model on the provided training data.
    The training_df should include the target column (e.g. "Attrition")
    along with all feature columns.
    
    The function one-hot encodes the features, scales the data, trains the model,
    and saves the model, scaler, and feature columns to disk using filenames that
    include the industry.
    """
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Get the chosen model instance and train it
    model = get_model(model_type)
    if model is None:
        return
    model.fit(X_scaled, y)
    
    # Save model artifacts using industry-specific filenames
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
    update_industry_record(industry, model_filename, scaler_filename, features_filename)

def update_industry_record(industry, model_file, scaler_file, feature_file):
    """
    Updates (or creates) a CSV file that records the trained model for each industry.
    Each row includes the Industry, model file names, and the training timestamp.
    """
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

# =============================================================================
# LOADING FUNCTIONS
# =============================================================================
def load_model(industry):
    """
    Loads the saved model, scaler, and feature columns for the given industry.
    Returns None for each if not found.
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

# =============================================================================
# CACHING THE TRAINING FROM AGGREGATED DATA
# =============================================================================
@st.cache(allow_output_mutation=True, show_spinner=False)
def train_model_from_aggregated_data(industry, target_column, model_type):
    """
    Loads the aggregated training data for the industry, trains a model using the selected model type,
    and returns the trained model, scaler, and feature columns.
    """
    filename = f"training_data_{industry}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_encoded = pd.get_dummies(X)
        feature_columns = list(X_encoded.columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        model = get_model(model_type)
        if model is None:
            return None, None, None
        model.fit(X_scaled, y)
        return model, scaler, feature_columns
    else:
        st.error("No aggregated training data found for this industry.")
        return None, None, None

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def predict_attrition_with_custom_model(employee_data, model, scaler, feature_columns):
    """
    Uses the provided model, scaler, and feature columns to predict attrition for a given employee.
    Combines the ML probability with the rule-based score.
    """
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers

# =============================================================================
# TRIGGER DETAILS (Unchanged)
# =============================================================================
TRIGGER_DETAILS = {
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "We are not getting enough female applicants",
            "lack_female_mentors": "We have few female mentors or leaders",
            "rigid_policies": "We do not offer flexible policies (e.g., maternity, remote, etc.)"
        },
        "solutions": {
            "lack_female_applicants": (
                "- **Partner with Women’s Universities** or female-oriented groups.\n"
                "- **Highlight DEI** in your recruitment materials."
            ),
            "lack_female_mentors": (
                "- **Implement formal mentorship** programs.\n"
                "- **Sponsor leadership development** for female employees."
            ),
            "rigid_policies": (
                "- Introduce **flexible working hours** and remote/hybrid options.\n"
                "- Improve **maternity/paternity benefits** and family-friendly leave."
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
                "- **Streamline decision-making** or reduce hierarchical layers.\n"
                "- Consider more **agile or cross-functional** teams."
            )
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1-on-1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        },
        "solutions": {
            "misaligned_role": (
                "- **Clarify job responsibilities** and set SMART goals.\n"
                "- Ensure roles align with employees’ **strengths** and career aspirations."
            ),
            "no_feedback": (
                "- Implement **frequent 1:1 check-ins** and agile feedback loops.\n"
                "- Use **performance dashboards** for real-time updates."
            ),
            "skill_gaps": (
                "- Provide **targeted training** and eLearning modules.\n"
                "- Offer **certification reimbursements** and skill-building workshops."
            )
        }
    },
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1-on-1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        },
        "solutions": {
            "misaligned_role": (
                "- **Clarify job responsibilities** and set SMART goals.\n"
                "- Align roles with employees’ strengths and preferences."
            ),
            "no_feedback": (
                "- Implement **regular 1:1 check-ins**.\n"
                "- Provide ongoing **coaching and feedback** rather than annual appraisals."
            ),
            "skill_gaps": (
                "- Offer **targeted training** in needed skill areas.\n"
                "- Encourage **peer-to-peer learning** or cross-functional rotations."
            )
        }
    },
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary is below market rates",
            "minimal_bonus": "Bonuses or variable pay are minimal or non-existent",
            "poor_benefits": "Benefits package is lacking (insurance, retirement, etc.)"
        },
        "solutions": {
            "below_market": (
                "- **Conduct market benchmarking** to adjust salaries to median or above.\n"
                "- Consider **geographic pay differentials** if applicable."
            ),
            "minimal_bonus": (
                "- Introduce **performance-based incentives** or profit-sharing.\n"
                "- Evaluate **RSUs** or equity grants for retention."
            ),
            "poor_benefits": (
                "- Offer **competitive health insurance** and retirement contributions.\n"
                "- Provide **flexible schedules** and wellness programs."
            )
        }
    },
    "Low college tier retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among certain colleges or entry-level hires",
            "mismatch_culture": "Mismatch between background and company culture",
            "poor_onboarding": "Insufficient onboarding for these hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Investigate root causes via **exit interviews**.\n"
                "- Build **campus ambassador** programs."
            ),
            "mismatch_culture": (
                "- Provide better **orientation** on company culture.\n"
                "- Pair new hires with **mentors**."
            ),
            "poor_onboarding": (
                "- Enhance **onboarding programs** with structured check-ins.\n"
                "- Offer a **buddy system** for new hires."
            )
        }
    },
    "Low industry retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from this industry",
            "mismatch_culture": "Mismatch between industry norms and company culture",
            "poor_onboarding": "Insufficient onboarding for these lateral hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Conduct **benchmarking** to check alignment with industry standards.\n"
                "- Explore **targeted retention strategies**."
            ),
            "mismatch_culture": (
                "- Emphasize **company values** and create inclusive teams.\n"
                "- Hold **town halls** or Q&A sessions."
            ),
            "poor_onboarding": (
                "- Develop **structured assimilation** programs.\n"
                "- Provide a **transition buddy** for lateral hires."
            )
        }
    },
    "Low company type retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among certain company backgrounds",
            "mismatch_culture": "Mismatch between prior and current company culture",
            "poor_onboarding": "Onboarding doesn’t address process differences"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Identify if certain company backgrounds churn quickly.\n"
                "- Adapt onboarding accordingly."
            ),
            "mismatch_culture": (
                "- Provide **culture assimilation** sessions.\n"
                "- Encourage **peer networking**."
            ),
            "poor_onboarding": (
                "- Offer a **comprehensive onboarding** covering processes.\n"
                "- Assign **buddies** who transitioned from similar backgrounds."
            )
        }
    },
    "High dissatisfaction (Pulse)": {
        "subproblems": {
            "work_life_imbalance": "Work-life imbalance or excessive workload",
            "poor_manager_relationships": "Employees feel managers are unsupportive",
            "limited_growth": "Limited growth or recognition opportunities"
        },
        "solutions": {
            "work_life_imbalance": (
                "- Offer **flexible scheduling** and **mental health** resources.\n"
                "- Encourage **healthy work boundaries**."
            ),
            "poor_manager_relationships": (
                "- Train managers on **emotional intelligence**.\n"
                "- Use **360-degree feedback** to identify issues."
            ),
            "limited_growth": (
                "- Implement **career development** paths.\n"
                "- Recognize achievements and **reward** top performers."
            )
        }
    }
}

# =============================================================================
# RULE-BASED SCORING (Unchanged)
# =============================================================================
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

# =============================================================================
# PREDICTION FUNCTION USING CUSTOM MODEL
# =============================================================================
def predict_attrition_with_custom_model(employee_data, model, scaler, feature_columns):
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers

# =============================================================================
# GENERATE SAMPLE CSV (For Bulk Mode) & DUMMY TRAINING FILE
# =============================================================================
def generate_sample_csv():
    sample_csv = pd.DataFrame({
        "Employee Age": [30, 45],
        "Average Employee Age": [35, 40],
        "Gender": ["Male", "Female"],
        "Female Employee Ratio": [50, 10],
        "Tenure (Months)": [36, 48],
        "Pulse": ["Medium", "High"],
        "Hasn't been promoted": [12, 36],
        "Minimum Promotion Cycle": [24, 24],
        "College Tier Retention": [60, 10],
        "Industry Retention": [60, 10],
        "Company Type Retention": [60, 10],
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
        "Average Employee Age": [35, 38, 36],
        "Gender": ["Male", "Female", "Male"],
        "Female Employee Ratio": [50, 20, 45],
        "Tenure (Months)": [36, 48, 24],
        "Pulse": ["Medium", "High", "Low"],
        "Hasn't been promoted": [12, 30, 15],
        "Minimum Promotion Cycle": [24, 24, 24],
        "College Tier Retention": [60, 15, 50],
        "Industry Retention": [60, 20, 55],
        "Company Type Retention": [60, 20, 55],
        "Last Performance Rating": [3, 1, 4],
        "Compa Ratio": [90, 65, 100],
        "Attrition": [0, 1, 0]
    })
    csv_buffer = io.StringIO()
    dummy_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.markdown(
    "<h2 style='text-align: center; color: #141414;'>Employee Attrition Prediction Tool</h2>",
    unsafe_allow_html=True
)

# Create two tabs for Train Mode and Test Mode
tabs = st.tabs(["Train Mode", "Test Mode"])

# -----------------------------------------------------------------------------
# TRAIN MODE TAB
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Train Mode")
    selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
    selected_train_model_display = st.selectbox("Select Model for Training", model_options_display, key="train_model_option")
    selected_train_model = model_options_map[selected_train_model_display]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
    with col2:
        st.markdown("### Detailed Guide for Training File")
        st.markdown("""
        **Your training file must include:**
        - A **target column** (e.g., `Attrition`) with binary values (0/1).
        - **Feature columns:**  
          - `Employee Age`  
          - `Average Employee Age`  
          - `Gender` (e.g., "Male", "Female")  
          - `Female Employee Ratio` (percentage)  
          - `Tenure (Months)`  
          - `Pulse` (e.g., "High", "Medium", "Low")  
          - `Hasn't been promoted` (months since last promotion)  
          - `Minimum Promotion Cycle` (in months)  
          - `College Tier Retention` (percentage)  
          - `Industry Retention` (percentage)  
          - `Company Type Retention` (percentage)  
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
            train_model(aggregated_df, target_column, selected_train_industry, selected_train_model)

# -----------------------------------------------------------------------------
# TEST MODE TAB
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Test Mode")
    st.markdown("""
    **Instructions for Testing:**
    
    - Ensure that you have trained a model in Train Mode.
    - Select your industry (must match the training industry).
    - Choose the model you want to use for analysis from the options below.
      (The recommended model is indicated.)
    - Then select whether to use **Single Employee** or **Bulk Employees** for testing.
      - **Single Employee:** Enter details manually.
      - **Bulk Employees:** Upload a CSV/Excel file with employee data.
    """)
    selected_test_industry = st.selectbox("Select Your Industry", industry_options, key="test_industry")
    selected_test_model_display = st.selectbox("Select Model for Analysis", model_options_display, key="test_model_option")
    selected_test_model = model_options_map[selected_test_model_display]
    
    # Retrain (or load) a model from the aggregated training data using the chosen model type.
    target_column_default = "Attrition"
    model_custom, scaler_custom, feature_columns_custom = train_model_from_aggregated_data(
        selected_test_industry, target_column_default, selected_test_model
    )
    
    test_mode = st.selectbox("Select Test Mode", ["Single Employee", "Bulk Employees"])
    
    # ---------------- Single Employee Mode ----------------
    if test_mode == "Single Employee":
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
                if model_custom is None:
                    st.error("Model could not be trained from aggregated data.")
                else:
                    final_score, triggers = predict_attrition_with_custom_model(input_data, model_custom, scaler_custom, feature_columns_custom)
                    st.session_state.score = final_score
                    st.session_state.triggers = triggers
                    st.session_state.prediction_made = True
                    st.session_state.employee_data = input_data
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
                scenario_data["Compa Ratio"] = st.slider("Compa Ratio (%) [Scenario]", 50, 150, scenario_data["Compa Ratio"], help="Adjust to see how risk changes if compensation changes.")
                scenario_data["Last Performance Rating"] = st.slider("Last Performance Rating [Scenario]", 1, 5, scenario_data["Last Performance Rating"], help="What if performance improves or worsens?")
                scenario_data["Pulse"] = st.radio("Pulse (Employee dissatisfaction) [Scenario]", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(scenario_data["Pulse"]), horizontal=True)
                scenario_score, scenario_triggers = predict_attrition_with_custom_model(scenario_data, model_custom, scaler_custom, feature_columns_custom)
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
                    st.write("**Scenario Negative Triggers**")
                    for t in neg_scenario_triggers:
                        st.markdown(f"- **{t}**")
                else:
                    st.markdown("*No negative triggers in this scenario.*")
    # ---------------- Bulk Employees Mode ----------------
    elif test_mode == "Bulk Employees":
        st.markdown("<h5 style='text-align: center; color: #FF2400;'>[Bulk Mode - Under Development]</h5>", unsafe_allow_html=True)
        st.write("""
        ### 📁 Bulk Employee Attrition Prediction
        Upload a CSV or Excel file with the following columns:
        - **Name**
        - **Employee Age**
        - **Gender**
        - **Chances of employee leaving (High, Medium, Low)**
        - **Hasn't been promoted (in months)**
        - **Min. Promotion cycle (in months)**
        - **Performance rating out of 5**
        - **Compa Ratio**
        - **College Tier (Tier 1, Tier 2, Tier 3)**
        - **Industry**
        - **Company Type**
        """)
        st.sidebar.header("🔧 Set Fixed Attributes and Retention Percentages")
        st.sidebar.subheader("📊 Fixed Attributes for All Employees")
        fixed_attributes = {
            "Average Employee Age": st.sidebar.slider("Average Employee Age", 18, 65, 35, step=1, help="Set the average age."),
            "Female Employee Ratio": st.sidebar.slider("Female Employee Ratio (%)", 0, 100, 40, step=1, help="Set the percentage of female employees.")
        }
        st.sidebar.subheader("🏫 College Tier Retention (%)")
        college_tiers = ["Tier 1", "Tier 2", "Tier 3"]
        college_retention = { tier: st.sidebar.slider(f"{tier} Retention (%)", 10, 100, 60 if tier=="Tier 1" else (50 if tier=="Tier 2" else 40), step=1, help=f"Set retention for {tier} colleges.") for tier in college_tiers }
        st.sidebar.subheader("🏭 Industry Retention (%)")
        industries = ["Tech", "Finance", "Healthcare", "Education", "Manufacturing"]
        industry_retention = { ind: st.sidebar.slider(f"{ind} Retention (%)", 10, 100, 60 if ind=="Tech" else (55 if ind=="Finance" else 50), step=1, help=f"Set retention for {ind}.") for ind in industries }
        st.sidebar.subheader("🏢 Company Type Retention (%)")
        company_types = ["Startup", "Enterprise", "Non-Profit", "SME"]
        company_type_retention = { ct: st.sidebar.slider(f"{ct} Retention (%)", 10, 100, 60 if ct=="Enterprise" else (55 if ct=="Startup" else 50), step=1, help=f"Set retention for {ct}.") for ct in company_types }
        st.sidebar.markdown("---\n**Important:** Adjust these attributes before uploading bulk data.")
        uploaded_file = st.file_uploader("📤 Upload CSV/Excel File", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                df_bulk = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
            st.write("### Bulk Data Preview")
            st.dataframe(df_bulk.head())
            required_cols = [
                "Name", "Employee Age", "Gender", "Chances of employee leaving (High, Medium, Low)",
                "Hasn't been promoted (in months)", "Min. Promotion cycle (in months)",
                "Performance rating out of 5", "Compa Ratio", "College Tier (Tier 1, Tier 2, Tier 3)",
                "Industry", "Company Type"
            ]
            missing = [c for c in required_cols if c not in df_bulk.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("🚀 Run Bulk Prediction"):
                    scores, triggers_list, names = [], [], []
                    for idx, row in df_bulk.iterrows():
                        row_dict = row.to_dict()
                        names.append(row_dict.get("Name"))
                        rename_mapping = {
                            "Chances of employee leaving (High, Medium, Low)": "Pulse",
                            "Hasn't been promoted (in months)": "Hasn't been promoted",
                            "Min. Promotion cycle (in months)": "Minimum Promotion Cycle",
                            "Performance rating out of 5": "Last Performance Rating",
                            "College Tier (Tier 1, Tier 2, Tier 3)": "College Tier"
                        }
                        for desc_col, internal_col in rename_mapping.items():
                            row_dict[internal_col] = row_dict.pop(desc_col, "Medium" if internal_col=="Pulse" else 0)
                        row_dict["Average Employee Age"] = fixed_attributes["Average Employee Age"]
                        row_dict["Female Employee Ratio"] = fixed_attributes["Female Employee Ratio"]
                        ct = row_dict.get("College Tier")
                        row_dict["College Tier Retention"] = college_retention.get(ct, 40)
                        ind_val = row_dict.get("Industry")
                        row_dict["Industry Retention"] = industry_retention.get(ind_val, 50)
                        ct_type = row_dict.get("Company Type")
                        row_dict["Company Type Retention"] = company_type_retention.get(ct_type, 50)
                        try:
                            bulk_score, bulk_trigs = predict_attrition_with_custom_model(row_dict, model_custom, scaler_custom, feature_columns_custom)
                        except Exception as e:
                            st.error(f"Row {idx} prediction failed: {e}")
                            scores.append(None)
                            triggers_list.append("Prediction Failed")
                            continue
                        scores.append(bulk_score)
                        neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
                        triggers_list.append(", ".join(neg_trigs) if neg_trigs else "None")
                    df_bulk["Attrition Score"] = scores
                    df_bulk["Negative Triggers"] = triggers_list
                    df_bulk["Name"] = names
                    st.success("Bulk Prediction Completed!")
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
