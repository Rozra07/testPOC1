import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: Train and Save Logistic Regression Model (Same as Before)
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

# Comment out if you only want to train once
train_and_save_model()

###############################################################################
# Step 2: Trigger Dictionary (Same as Before)
###############################################################################
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
                "- **Highlight DEI** in recruitment materials."
            ),
            "lack_female_mentors": (
                "- **Implement formal mentorship** programs.\n"
                "- **Sponsor leadership development** for female employees."
            ),
            "rigid_policies": (
                "- Offer **flexible working hours** and remote/hybrid options.\n"
                "- Improve **maternity/paternity** benefits."
            )
        }
    },
    # ... other triggers omitted for brevity (same structure) ...
}

###############################################################################
# Step 3: Rule-Based Scoring (Unchanged)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []

    # Example triggers
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
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")

    # ... other conditions omitted for brevity ...

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
# Step 4: ML Prediction (Unchanged, Just Returns Score & Triggers)
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
# Step 5: Streamlit UI with Full-Width Risk Box & Live Scenario Updates
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>", unsafe_allow_html=True)

# Store user input in session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "score" not in st.session_state:
    st.session_state.score = None
if "triggers" not in st.session_state:
    st.session_state.triggers = []
if "employee_data" not in st.session_state:
    st.session_state.employee_data = {}

# --- A. Collect Employee Data ---
with st.form("attrition_form", clear_on_submit=False):
    st.write("#### Enter Employee / Company Details")
    user_data = {
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

    if st.form_submit_button("🚀 Predict"):
        combined_score, triggers = predict_attrition(user_data)
        st.session_state.score = combined_score
        st.session_state.triggers = triggers
        st.session_state.prediction_made = True
        st.session_state.employee_data = user_data

# --- B. If Predict is Clicked, Show the Full-Width Risk Box ---
if st.session_state.prediction_made:
    score = st.session_state.score
    triggers = st.session_state.triggers

    # Container for the big color-coded risk box, full width
    with st.container():
        if score >= 75:
            color_html = "#ff4d4d"
            label_html = f"⚠️ HIGH Attrition Risk<br>{score:.2f}% 🚨"
        elif score >= 60:
            color_html = "#ff9933"
            label_html = f"⚠️ Moderate to High Risk<br>{score:.2f}% ⚡"
        elif score >= 35:
            color_html = "#ffd700"
            label_html = f"⚖️ Moderate Attrition Risk<br>{score:.2f}% 📉"
        else:
            color_html = "#28a745"
            label_html = f"✅ SAFE! Low Attrition Risk<br>{score:.2f}% 🌱"

        # Display a full-width colored box
        st.markdown(
            f"""
            <div style="background-color:{color_html}; color:white; padding:15px; 
                        border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">
                {label_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create two columns below the full-width risk box
    col_left, col_right = st.columns(2)

    # ---------------- LEFT COLUMN: Original triggers + sub-problems ---------------
    with col_left:
        st.write("### 1. Key Contributing Factors")
        if triggers:
            for t in triggers:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No major negative triggers identified.*")

        st.write("### 2. Sub-Problems Selection")
        sub_problem_selections = {}
        for trig in triggers:
            if trig not in TRIGGER_DETAILS:
                continue
            st.write(f"**{trig}**")
            subprobs = TRIGGER_DETAILS[trig]["subproblems"]
            chosen = []
            for sub_key, sub_label in subprobs.items():
                # We'll store these checks in session state so they persist on re-run
                chk_id = f"{trig}-{sub_key}"
                if chk_id not in st.session_state:
                    st.session_state[chk_id] = False
                new_val = st.checkbox(sub_label, key=chk_id)
                if new_val:
                    chosen.append(sub_key)
            sub_problem_selections[trig] = chosen

        # Show Solutions On-Demand
        if st.button("💡 Show Customized Solutions"):
            st.write("### Recommended Solutions / Action Points")
            any_selected = False
            for trig in triggers:
                if trig not in TRIGGER_DETAILS:
                    continue
                chosen_subproblems = sub_problem_selections[trig]
                if chosen_subproblems:
                    any_selected = True
                    st.write(f"**Trigger:** {trig}")
                    for sub_key in chosen_subproblems:
                        solution_text = TRIGGER_DETAILS[trig]["solutions"].get(sub_key, "")
                        st.markdown(f"- **{sub_key}**: {solution_text}")
            if not any_selected:
                st.info("No sub-problems were selected; no solutions to display.")

    # ---------------- RIGHT COLUMN: Live Scenario Planning ---------------
    with col_right:
        st.write("### 3. Live What-If Scenario")
        st.write("Adjust these factors to see how the risk changes instantly.")

        scenario_data = dict(st.session_state.employee_data)  # copy original

        # We'll re-run scenario predict automatically as the user moves the sliders
        # So we do NOT use a button—just retrieve slider values directly.
        scenario_data["Compa Ratio"] = st.slider(
            "Compa Ratio (%) [Scenario]",
            50, 150, scenario_data["Compa Ratio"],
            help="Try raising or lowering compensation to see effect."
        )
        scenario_data["Last Performance Rating"] = st.slider(
            "Last Performance Rating [Scenario]",
            1, 5, scenario_data["Last Performance Rating"],
            help="What if performance improves or worsens?"
        )
        scenario_data["Pulse"] = st.radio(
            "Pulse (Employee dissatisfaction) [Scenario]",
            ["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(scenario_data["Pulse"]),
            horizontal=True
        )

        # Now we recalc scenario risk on-the-fly
        scenario_score, scenario_triggers = predict_attrition(scenario_data)
        st.markdown(f"**Scenario Attrition Risk:** {scenario_score:.2f}%")

        # Compare scenario risk to original
        diff = scenario_score - st.session_state.score
        if diff > 0:
            st.markdown(f"<span style='color:red;'>(+{diff:.2f}%) Higher than original</span>", unsafe_allow_html=True)
        elif diff < 0:
            st.markdown(f"<span style='color:green;'>({diff:.2f}%) Lower than original</span>", unsafe_allow_html=True)
        else:
            st.write("No change from the original scenario.")

        # Display scenario triggers
        if scenario_triggers:
            st.write("#### Scenario Triggers")
            for t in scenario_triggers:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No major negative triggers in this scenario.*")
