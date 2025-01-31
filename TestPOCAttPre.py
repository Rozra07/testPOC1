import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import openai

###############################################################################
# 0. Set Your OpenAI API Key (Replace "xyz" with your real key)
###############################################################################
openai.api_key = "xyz"  # Not recommended for production. Use environment vars or st.secrets.

###############################################################################
# 1. Train and Save Logistic Regression Model (UNCHANGED Weighted Factor + ML)
###############################################################################
def train_and_save_model():
    """
    Creates synthetic data, trains a Logistic Regression model, saves the artifacts.
    """
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

    y = np.random.randint(0, 2, size=n_samples)

    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])
    feature_columns = df_encoded.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)

# Uncomment if only needed once
train_and_save_model()

###############################################################################
# 2. Master Dictionary for Negative Triggers, Sub-Problems, Solutions
###############################################################################
# The same dictionary from your final Phase 1 code, with each negative trigger
# plus the subproblems and solutions you originally defined. For brevity, below
# is a sample. You can copy/paste your full dictionary here.
TRIGGER_DETAILS = {
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "We are not getting enough female applicants",
            "lack_female_mentors": "We have few female mentors or leaders",
            "rigid_policies": "We do not offer flexible policies (e.g., maternity, remote, etc.)"
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are unclear or inconsistent",
            "no_mentorship": "No proper mentorship or upskilling tracks exist",
            "bureaucratic_structure": "The organization structure is too bureaucratic"
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1-on-1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        }
    },
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role or expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback or 1-on-1 sessions",
            "skill_gaps": "Skill gaps or training needs not addressed"
        }
    },
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary is below market rates",
            "minimal_bonus": "Bonuses or variable pay are minimal or non-existent",
            "poor_benefits": "Benefits package is lacking (insurance, retirement, etc.)"
        }
    },
    "Low college tier retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among certain colleges or entry-level hires",
            "mismatch_culture": "Mismatch between background and company culture",
            "poor_onboarding": "Insufficient onboarding or assimilation for these hires"
        }
    },
    "Low industry retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from this industry",
            "mismatch_culture": "Mismatch between industry norms and your company's culture",
            "poor_onboarding": "Insufficient onboarding for these lateral hires"
        }
    },
    "Low company type retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from certain company backgrounds",
            "mismatch_culture": "Mismatch between prior company culture and current environment",
            "poor_onboarding": "Onboarding doesn’t address differences in processes, tools, or structures"
        }
    },
    "High dissatisfaction (Pulse)": {
        "subproblems": {
            "work_life_imbalance": "Work-life imbalance or excessive workload",
            "poor_manager_relationships": "Employees feel managers are unsupportive",
            "limited_growth": "Limited growth or recognition opportunities"
        }
    }
}
# Positive triggers (like "Excellent performance rating", "High compensation ratio",
# and "Low dissatisfaction (Pulse)") are intentionally excluded, as they reduce risk.

###############################################################################
# 3. Weighted Factor Calculation (Exact from Your Code)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []

    # EXACT RULES from your final logic
    # 1. Gender
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # 2. Stagnant promotions
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")

    # 3. Performance Rating
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
        triggers.append("Excellent performance rating")  # positive

    # 4. Compa Ratio
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")  # positive

    # 5. Retention
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

    # 6. Pulse
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")  # positive

    # EXTREME FACTOR SCALING
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
# 4. ML + Weighted Factor = Final Combined Score
###############################################################################
def predict_attrition(employee_data):
    # Load model
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Prepare input
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    # ML Probability
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100

    # Rule-Based Weighted Factor
    rule_score, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    # Combined
    combined_score = 0.75 * rule_score + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# 5. LLM Explanation Function (AI-Powered)
###############################################################################
def get_llm_explanation(trigger, chosen_subproblems, industry):
    """
    Calls OpenAI to generate dynamic, context-aware solutions.
    """
    # Build subproblem text
    sub_text = "\n".join([f"- {s}" for s in chosen_subproblems])
    prompt = f"""
You are an HR consultant helping a {industry} industry user reduce attrition.
They have identified the trigger: '{trigger}' with these sub-problems:
{sub_text}

Explain why this trigger increases attrition and provide actionable solutions.
Keep your response around 150-200 words.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling LLM: {e}"

###############################################################################
# 6. Streamlit UI with Side-by-Side, Full-Width Risk Box, and Live Scenarios
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>⭐ Employee Attrition Prediction (LLM Enhanced) ⭐</h2>", unsafe_allow_html=True)

# Session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "score" not in st.session_state:
    st.session_state.score = None
if "triggers" not in st.session_state:
    st.session_state.triggers = []
if "employee_data" not in st.session_state:
    st.session_state.employee_data = {}
if "industry" not in st.session_state:
    st.session_state.industry = "Others"

# --- A) Input Form
with st.form("attrition_form"):
    st.write("#### 1. Enter Employee / Company Details")
    employee_data_input = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Average Employee Age", 18, 65, 35),
        "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
        "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Pulse": st.radio("Pulse (Employee dissatisfaction)", ["High", "Medium", "Low"], horizontal=True),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }

    industry = st.selectbox("Which Industry?", ["Manufacturing", "BFSI", "IT/ITES", "Retail", "Others"], index=4)
    if st.form_submit_button("🚀 Predict"):
        final_score, triggers = predict_attrition(employee_data_input)
        st.session_state.score = final_score
        st.session_state.triggers = triggers
        st.session_state.prediction_made = True
        st.session_state.employee_data = employee_data_input
        st.session_state.industry = industry

# --- B) If Predicted
if st.session_state.prediction_made:
    score = st.session_state.score
    triggers = st.session_state.triggers
    industry = st.session_state.industry

    # Full-width color-coded risk box
    with st.container():
        if score >= 75:
            bg_color = "#ff4d4d"
            label_html = f"⚠️ HIGH Attrition Risk<br>{score:.2f}% 🚨"
        elif 60 <= score < 75:
            bg_color = "#ff9933"
            label_html = f"⚠️ Moderate to High Risk<br>{score:.2f}% ⚡"
        elif 35 <= score < 60:
            bg_color = "#ffd700"
            label_html = f"⚖️ Moderate Attrition Risk<br>{score:.2f}% 📉"
        else:
            bg_color = "#28a745"
            label_html = f"✅ SAFE! Low Attrition Risk<br>{score:.2f}% 🌱"

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; color:white; padding:15px; border-radius:10px; 
                        text-align:center; font-size:24px; font-weight:bold;">
                {label_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Two columns
    col_left, col_right = st.columns(2)

    # ---- LEFT: Negative triggers + sub-problem selection
    with col_left:
        st.write("### 2. Key Contributing Factors")
        negative_triggers = []
        for trig in triggers:
            if trig in TRIGGER_DETAILS:  # negative
                negative_triggers.append(trig)
                st.markdown(f"- **{trig}**")
        if not negative_triggers:
            st.markdown("*No major negative triggers identified.*")

        st.write("### Select Sub-Problems (if any)")
        sub_problem_selections = {}
        for trig in negative_triggers:
            st.write(f"**{trig}**")
            subprobs_dict = TRIGGER_DETAILS[trig]["subproblems"]
            chosen_subs = []
            for sub_key, sub_label in subprobs_dict.items():
                chk_id = f"{trig}-{sub_key}"
                # store the checkbox in session_state to preserve on re-runs
                if chk_id not in st.session_state:
                    st.session_state[chk_id] = False
                if st.checkbox(sub_label, key=chk_id):
                    chosen_subs.append(sub_label)
            sub_problem_selections[trig] = chosen_subs

        # Button to get LLM-based solutions
        if st.button("💡 AI-Powered Solutions"):
            st.write("## LLM Recommendations / Explanations")
            for trig in negative_triggers:
                chosen_list = sub_problem_selections[trig]
                if chosen_list:
                    llm_response = get_llm_explanation(trig, chosen_list, industry)
                    st.markdown(f"### Trigger: {trig}")
                    st.markdown(llm_response)

    # ---- RIGHT: Live Scenario Planning
    with col_right:
        st.write("### 3. Live What-If Scenario")
        scenario_data = dict(st.session_state.employee_data)  # copy original

        # We'll let the user dynamically tweak these fields
        scenario_data["Compa Ratio"] = st.slider(
            "Compa Ratio (%) [Scenario]",
            50, 150, scenario_data["Compa Ratio"]
        )
        scenario_data["Last Performance Rating"] = st.slider(
            "Last Performance Rating [Scenario]",
            1, 5, scenario_data["Last Performance Rating"]
        )
        scenario_data["Pulse"] = st.radio(
            "Pulse [Scenario]",
            ["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(scenario_data["Pulse"]),
            horizontal=True
        )

        # Auto-recompute scenario
        scenario_score, scenario_trigs = predict_attrition(scenario_data)
        st.markdown(f"**Scenario Attrition Risk:** {scenario_score:.2f}%")

        diff = scenario_score - score
        if diff > 0:
            st.markdown(f"<span style='color:red;'>Risk +{diff:.2f}% higher than original.</span>", unsafe_allow_html=True)
        elif diff < 0:
            st.markdown(f"<span style='color:green;'>Risk {diff:.2f}% lower than original.</span>", unsafe_allow_html=True)
        else:
            st.write("No change from original risk.")

        # Show negative triggers in scenario
        neg_scenario_trigs = [t for t in scenario_trigs if t in TRIGGER_DETAILS]
        if neg_scenario_trigs:
            st.write("**Scenario Negative Triggers**")
            for t in neg_scenario_trigs:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No negative triggers in the scenario.*")
