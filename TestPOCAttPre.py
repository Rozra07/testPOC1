import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1) Import openai
import openai

###############################################################################
# Step 0: Set your OpenAI API Key
###############################################################################
# In a secure environment, load from st.secrets or an environment variable.
# For demonstration, we are hardcoding the key here (not recommended for production).
openai.api_key = "xyz"   # <-- Replace "xyz" with your real key

###############################################################################
# Step 1: Train and Save a Dummy Logistic Regression Model
###############################################################################
def train_and_save_model():
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

train_and_save_model()

###############################################################################
# Step 2: Dictionary for Negative Triggers (Example)
###############################################################################
TRIGGER_DETAILS = {
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "Not enough female applicants",
            "lack_female_mentors": "Few female mentors/leaders",
            "rigid_policies": "Rigid policies (no flexible work, limited maternity/paternity)"
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are unclear",
            "no_mentorship": "No mentorship/upskilling programs",
            "bureaucratic_structure": "Too many hierarchical layers"
        }
    },
    # Add more triggers as needed...
}

###############################################################################
# Step 3: Rule-Based Scoring
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []

    # Example condition
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")

    # Add other conditions as in your previous code...

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
# Step 4: Combine ML + Rule Score
###############################################################################
def predict_attrition(employee_data):
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# Step 5: LLM Explanation
###############################################################################
def get_llm_explanation(trigger, chosen_subproblems, industry):
    """
    Call OpenAI ChatCompletion API to generate dynamic HR solutions.
    """
    # Build prompt
    subproblem_text = "\n".join([f"- {sub}" for sub in chosen_subproblems])

    prompt = f"""
You are an HR consultant. The user is from the {industry} industry.
They have identified the trigger: '{trigger}' with these sub-problems:
{subproblem_text}

Explain why this causes attrition risk, and provide actionable strategies (150-200 words).
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        reply = response["choices"][0]["message"]["content"]
        return reply.strip()

    except Exception as e:
        return f"Error calling LLM: {str(e)}"

###############################################################################
# Step 6: Streamlit UI
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Tool (OpenAI Integrated) 🚀</h2>", unsafe_allow_html=True)

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

# --- A) Input Form ---
with st.form("attrition_form"):
    st.write("#### Enter Employee / Company Details")
    input_data = {
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
        final_score, triggers = predict_attrition(input_data)
        st.session_state.score = final_score
        st.session_state.triggers = triggers
        st.session_state.prediction_made = True
        st.session_state.employee_data = input_data
        st.session_state.industry = industry

# --- B) Show Results if Predicted ---
if st.session_state.prediction_made:
    score = st.session_state.score
    triggers = st.session_state.triggers
    industry = st.session_state.industry

    # Full-width risk box
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
            <div style="background-color:{bg_color}; color:white; padding:15px; 
                        border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">
                {msg_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    col_left, col_right = st.columns(2)

    # LEFT: Negative triggers + sub-problems
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
            sub_dict = TRIGGER_DETAILS[trig]["subproblems"]
            chosen_subs = []
            for sub_key, sub_label in sub_dict.items():
                check_val = st.checkbox(sub_label, key=f"{trig}-{sub_key}")
                if check_val:
                    chosen_subs.append(sub_label)
            sub_problem_selections[trig] = chosen_subs

        # Generate AI solutions
        if st.button("💡 Generate AI-Powered Recommendations"):
            st.write("## Dynamic LLM Explanations")
            for trig in negative_triggers:
                chosen = sub_problem_selections[trig]
                if chosen:
                    explanation = get_llm_explanation(trig, chosen, industry)
                    st.markdown(f"**Trigger:** {trig}")
                    st.markdown(explanation)

    # RIGHT: Scenario Planning (optional or omitted for brevity)
    with col_right:
        st.write("### What-If Scenario Planning")
        scenario_data = dict(st.session_state.employee_data)

        scenario_data["Compa Ratio"] = st.slider("Compa Ratio (%) [Scenario]", 50, 150, scenario_data["Compa Ratio"])
        scenario_data["Last Performance Rating"] = st.slider("Last Performance Rating [Scenario]", 1, 5, scenario_data["Last Performance Rating"])

        scenario_score, scenario_trigs = predict_attrition(scenario_data)
        st.write(f"**Scenario Risk:** {scenario_score:.2f}%")
        delta = scenario_score - score
        if delta > 0:
            st.error(f"Risk +{delta:.2f}% higher.")
        elif delta < 0:
            st.success(f"Risk {delta:.2f}% lower.")
        else:
            st.info("No change.")
