import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: Train and Save a Logistic Regression Model
###############################################################################
def train_and_save_model():
    """
    Creates a dummy dataset, trains a logistic regression model, and saves the
    model, scaler, and feature columns for later use. This is unchanged logic.
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

    # Dummy binary target
    y = np.random.randint(0, 2, size=n_samples)

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])
    feature_columns = df_encoded.columns

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # Train logistic regression
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    # Save artifacts
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)

# (Uncomment next line if you only want to train the model once)
train_and_save_model()

###############################################################################
# Step 2: TRIGGER_DETAILS for Negative Triggers
###############################################################################
# NOTE: Only include triggers that *increase* risk. Positive triggers won't have sub-problems.
TRIGGER_DETAILS = {
    # 1. "Low gender diversity"
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "Not enough female applicants",
            "lack_female_mentors": "Few female mentors/leaders",
            "rigid_policies": "Rigid policies (no flexible working, limited maternity/paternity)"
        },
        "solutions": {
            "lack_female_applicants": (
                "- **Partner with Women’s Universities** or women-focused professional networks.\n"
                "- **Highlight DEI** in your job postings and employer branding."
            ),
            "lack_female_mentors": (
                "- **Implement formal mentorship** programs.\n"
                "- **Sponsor leadership development** for existing female employees."
            ),
            "rigid_policies": (
                "- Introduce **flexible/hybrid** work schedules.\n"
                "- Enhance **maternity/paternity benefits** and on-site childcare if possible."
            )
        }
    },

    # 2. "Stagnant promotions"
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are unclear or inconsistent",
            "no_mentorship": "No mentorship/upskilling programs",
            "bureaucratic_structure": "Too many hierarchical layers/bureaucracy"
        },
        "solutions": {
            "unclear_criteria": (
                "- **Publish transparent criteria** linked to clear KPIs.\n"
                "- Provide **regular feedback** on promotion readiness."
            ),
            "no_mentorship": (
                "- Launch **formal mentoring** or buddy programs.\n"
                "- Offer **upskilling** and learning stipends."
            ),
            "bureaucratic_structure": (
                "- **Streamline decision-making** or reduce hierarchy.\n"
                "- Adopt more **agile, cross-functional** teams."
            )
        }
    },

    # 3. "Very low performance rating"
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job role/expectations are unclear or mismatched",
            "no_feedback": "Lack of continuous feedback / 1-on-1s",
            "skill_gaps": "Skill gaps or training needs not addressed"
        },
        "solutions": {
            "misaligned_role": (
                "- **Clarify job responsibilities** and set SMART goals.\n"
                "- Align tasks with employees’ **strengths**."
            ),
            "no_feedback": (
                "- Implement **regular 1:1 check-ins** and agile feedback loops.\n"
                "- Provide **real-time dashboards** or frequent reviews."
            ),
            "skill_gaps": (
                "- Offer **targeted training** and eLearning modules.\n"
                "- Provide **certification reimbursements** and skill workshops."
            )
        }
    },

    # 4. "Low performance rating"
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Role or expectations unclear/mismatched",
            "no_feedback": "Lack of feedback or performance discussions",
            "skill_gaps": "Employee lacks key skills/training"
        },
        "solutions": {
            "misaligned_role": (
                "- **Re-assess job responsibilities** to ensure good fit.\n"
                "- Set **SMART** performance goals and accountability."
            ),
            "no_feedback": (
                "- **Frequent check-ins** with manager.\n"
                "- Peer coaching or **weekly sprints** for feedback."
            ),
            "skill_gaps": (
                "- Implement **upskilling** or cross-functional training.\n"
                "- Use a **mentorship** or buddy system for knowledge-sharing."
            )
        }
    },

    # 5. "Low compensation competitiveness"
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary below market benchmarks",
            "minimal_bonus": "Bonuses/variable pay are minimal",
            "poor_benefits": "Benefits package lacking (insurance, retirement, etc.)"
        },
        "solutions": {
            "below_market": (
                "- **Benchmark** salaries and adjust to median or above.\n"
                "- Consider **geo-based pay** if relevant."
            ),
            "minimal_bonus": (
                "- Introduce **performance-based incentives** or profit-sharing.\n"
                "- Evaluate **RSUs** or equity for retention."
            ),
            "poor_benefits": (
                "- Offer **competitive health & retirement** benefits.\n"
                "- Provide **flexible schedules** and wellness programs."
            )
        }
    },

    # 6. "Low college tier retention"
    "Low college tier retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among certain colleges / new grads",
            "mismatch_culture": "Mismatch between background & company culture",
            "poor_onboarding": "Onboarding not tailored for entry-level hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Investigate root causes via **exit interviews**.\n"
                "- Build **campus ambassador** programs for targeted hiring."
            ),
            "mismatch_culture": (
                "- Offer better **orientation** on company culture.\n"
                "- **Mentorship** for new grads from similar backgrounds."
            ),
            "poor_onboarding": (
                "- Enhance **onboarding** with 30/60/90 check-ins.\n"
                "- Create a **buddy system** for new grads."
            )
        }
    },

    # 7. "Low industry retention"
    "Low industry retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover from employees with certain industry backgrounds",
            "mismatch_culture": "Mismatch between industry norms & your org culture",
            "poor_onboarding": "Onboarding insufficient for lateral hires"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- **Benchmark** comp & roles vs. industry.\n"
                "- Provide **targeted retention** (mentorship, training)."
            ),
            "mismatch_culture": (
                "- Emphasize **company values** & inclusive culture.\n"
                "- Host **town halls** or Q&A sessions for lateral hires."
            ),
            "poor_onboarding": (
                "- Structure **assimilation** for mid-career folks.\n"
                "- Provide a **transition buddy** with industry experience."
            )
        }
    },

    # 8. "Low company type retention"
    "Low company type retention": {
        "subproblems": {
            "high_turnover_talent_pools": "High turnover among employees from certain prior companies",
            "mismatch_culture": "Mismatch between previous company culture & current environment",
            "poor_onboarding": "Onboarding doesn’t address new processes/tools"
        },
        "solutions": {
            "high_turnover_talent_pools": (
                "- Identify if certain backgrounds churn quickly.\n"
                "- Adapt projects / roles for better alignment."
            ),
            "mismatch_culture": (
                "- Provide **culture assimilation** sessions.\n"
                "- Encourage **peer networking** for new hires."
            ),
            "poor_onboarding": (
                "- Have a **comprehensive onboarding** about processes/tools.\n"
                "- **Pair** with employees who made similar transitions."
            )
        }
    },

    # 9. "High dissatisfaction (Pulse)"
    "High dissatisfaction (Pulse)": {
        "subproblems": {
            "work_life_imbalance": "Employees overworked or no work-life balance",
            "poor_manager_relationships": "Unsupportive or distant managers",
            "limited_growth": "Limited career growth or recognition"
        },
        "solutions": {
            "work_life_imbalance": (
                "- Offer **flexible hours** / remote options.\n"
                "- Provide **mental health** resources and reduce after-hours calls."
            ),
            "poor_manager_relationships": (
                "- Train managers in **emotional intelligence**.\n"
                "- Collect **360-degree feedback** on leadership."
            ),
            "limited_growth": (
                "- Establish **career development** paths.\n"
                "- Recognize achievements publicly & frequently."
            )
        }
    }
}

# Positive triggers (these reduce the score):
#   - "Excellent performance rating"
#   - "High compensation ratio"
#   - "Low dissatisfaction (Pulse)"
# We skip them in TRIGGER_DETAILS (no sub-problems needed).

###############################################################################
# Step 3: Rule-Based Scoring
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    """
    Computes a 0-100 rule-based score. Returns (score, triggers) if return_triggers=True.
    """
    score = 0
    extreme_factors = 0
    triggers = []

    # Gender Diversity
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # Stagnant Promotions
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
        triggers.append("Excellent performance rating")  # positive

    # Compensation
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")  # positive

    # Retention
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

    # Pulse
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")  # positive

    # Extreme Factors
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
# Step 4: Machine Learning Combination
###############################################################################
def predict_attrition(employee_data):
    """
    Loads the saved logistic regression model, transforms the data,
    and combines ML probability with rule-based score.
    Returns (combined_score, triggers).
    """
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
# Step 5: Streamlit UI - Side-by-Side & Full-Width Risk Box
###############################################################################
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>",
    unsafe_allow_html=True
)

# Session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "score" not in st.session_state:
    st.session_state.score = None
if "triggers" not in st.session_state:
    st.session_state.triggers = []
if "employee_data" not in st.session_state:
    st.session_state.employee_data = {}

# --- A) Input Form ---
with st.form("attrition_form"):
    st.write("#### Enter Employee / Company Details")
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

    if st.form_submit_button("🚀 Predict"):
        final_score, triggers = predict_attrition(input_data)
        st.session_state.score = final_score
        st.session_state.triggers = triggers
        st.session_state.prediction_made = True
        st.session_state.employee_data = input_data

# --- B) Show Results if Predicted ---
if st.session_state.prediction_made:
    score = st.session_state.score
    triggers = st.session_state.triggers

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
            <div style="background-color:{bg_color}; color:white; padding:15px; border-radius:10px; 
                        text-align:center; font-size:24px; font-weight:bold;">
                {msg_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Two columns below
    col_left, col_right = st.columns(2)

    # ---------- LEFT: Triggers + Sub-Problems ----------
    with col_left:
        st.write("### Key Contributing Factors")
        negative_triggers = []
        for t in triggers:
            # We only list negative triggers here
            if t in TRIGGER_DETAILS:
                negative_triggers.append(t)
                st.markdown(f"- **{t}**")
            else:
                # It's a positive or unrecognized trigger
                # e.g. "Excellent performance rating" => do not show sub-problems
                pass

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

    # ---------- RIGHT: Live What-If Scenario -------------
    with col_right:
        st.write("### What-If Scenario Planning")
        scenario_data = dict(st.session_state.employee_data)

        # Let’s pick a few changeable fields
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

        # Recompute scenario risk every time user changes
        scenario_score, scenario_triggers = predict_attrition(scenario_data)
        st.write(f"**Scenario Attrition Risk:** {scenario_score:.2f}%")

        # Compare to original
        diff = scenario_score - score
        if diff > 0:
            st.markdown(f"<span style='color:red;'>Risk +{diff:.2f}% higher than original.</span>", unsafe_allow_html=True)
        elif diff < 0:
            st.markdown(f"<span style='color:green;'>Risk {diff:.2f}% lower than original.</span>", unsafe_allow_html=True)
        else:
            st.write("No change from original risk.")

        # Show triggers for scenario
        neg_scenario_triggers = [t for t in scenario_triggers if t in TRIGGER_DETAILS]
        if neg_scenario_triggers:
            st.write("**Scenario Negative Triggers**")
            for t in neg_scenario_triggers:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No negative triggers in the scenario.*")
