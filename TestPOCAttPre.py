import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################
# Step 1: Train and Save a Dummy Logistic Regression Model (UNCHANGED)
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

# Uncomment if you only want to train the model once, otherwise it re-trains every run.
train_and_save_model()

###############################################################################
# Step 2: Define the Master Dictionary for Triggers, Sub-Problems, and Solutions
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
                "- **Partner with Women’s Universities** or female-oriented professional groups.\n"
                "- **Highlight DEI** (Diversity, Equity, Inclusion) in your recruitment materials."
            ),
            "lack_female_mentors": (
                "- **Implement formal mentorship** programs.\n"
                "- **Sponsor leadership development** for existing female employees."
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
                "- Consider more **agile or cross-functional** teams to encourage skill growth."
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
            "high_turnover_talent_pools": "High turnover among certain colleges or entry-level hires",
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
                "- Enhance **onboarding programs** with structured check-ins (30/60/90 days).\n"
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
                "- Develop **structured assimilation** for mid-career folks.\n"
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
            "work_life_imbalance": "Work-life imbalance or excessive workload",
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
                "- Collect **360-degree feedback** to identify manager blind spots."
            ),
            "limited_growth": (
                "- Implement **career development** paths and internal mobility.\n"
                "- Recognize achievements publicly and **reward** top performers."
            )
        }
    }
}

###############################################################################
# Step 3: Rule-Based Scoring
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []

    # 1. Gender Diversity
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # 2. Stagnant Promotions
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
        triggers.append("Excellent performance rating")

    # 4. Compa Ratio
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")

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
        triggers.append("Low dissatisfaction (Pulse)")

    # Extreme Factor Scaling
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
# Step 4: ML Prediction - Combines ML Probability + Rule-Based Score
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
# Step 5: Streamlit UI
###############################################################################
# We use session_state to handle a multi-step process:
# 1. Collect employee data -> Display base prediction.
# 2. Show triggers -> user picks sub-problems.
# 3. Show solutions.
# 4. (New) Show "What-If" scenario sliders to see updated risk if certain features change.

st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>", unsafe_allow_html=True)

# ------------- Step A: Collect Employee Data Form -------------
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "score" not in st.session_state:
    st.session_state.score = None
if "triggers" not in st.session_state:
    st.session_state.triggers = []

with st.form("attrition_form", clear_on_submit=False):
    st.write("### 1. Enter Employee/Company Details")
    employee_data = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Avg Employee Age", 18, 65, 35),
        "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
        "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Pulse": st.radio("Employee dissatisfaction according to Pulse", ["High", "Medium", "Low"], horizontal=True),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }

    industry = st.selectbox("Which Industry Are You From?", ["Manufacturing", "BFSI", "IT/ITES", "Retail", "Others"], index=4)
    submit_button = st.form_submit_button("🚀 Predict")

    if submit_button:
        score, triggers = predict_attrition(employee_data)
        st.session_state.score = score
        st.session_state.triggers = triggers
        st.session_state.prediction_made = True

# ------------- Step B: If Prediction is made, show results -------------
if st.session_state.prediction_made:
    score = st.session_state.score
    triggers = st.session_state.triggers

    # 1. Show the Risk Category
    if score >= 75:
        st.markdown(
            f'<div style="background-color:#ff4d4d; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ HIGH Attrition Risk! <br> {score:.2f}% 🚨</div>',
            unsafe_allow_html=True
        )
    elif 60 <= score < 75:
        st.markdown(
            f'<div style="background-color:#ff9933; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ Moderate to High Risk <br> {score:.2f}% ⚡</div>',
            unsafe_allow_html=True
        )
    elif 35 <= score < 60:
        st.markdown(
            f'<div style="background-color:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚖️ Moderate Attrition Risk <br> {score:.2f}% 📉</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color:#28a745; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'✅ SAFE! Low Attrition Risk <br> {score:.2f}% 🌱</div>',
            unsafe_allow_html=True
        )

    # 2. List the triggers
    st.write("### 2. Key Contributing Factors")
    if triggers:
        for t in triggers:
            st.markdown(f"- **{t}**")
    else:
        st.markdown("*No major negative triggers identified.*")

    # 3. Ask the user to pick sub-problems for each trigger
    st.write("---")
    st.write("### 3. Select Sub-Problems That Apply to Your Organization")
    sub_problem_selections = {}
    for trig in triggers:
        # If the trigger is a "positive" factor or not in TRIGGER_DETAILS, skip
        if trig not in TRIGGER_DETAILS:
            continue
        st.write(f"**{trig}**: Which of these sub-problems do you see in your organization?")
        
        # Display checkboxes
        subproblem_dict = TRIGGER_DETAILS[trig]["subproblems"]
        chosen = []
        for sub_key, sub_label in subproblem_dict.items():
            check_val = st.checkbox(f"{sub_label}", key=f"{trig}-{sub_key}")
            if check_val:
                chosen.append(sub_key)
        sub_problem_selections[trig] = chosen

    # Button to "Show me solutions"
    if st.button("💡 Show Customized Solutions"):
        st.write("### 4. Customized Recommendations and Action Points")
        any_selection = False

        for trig in triggers:
            if trig not in TRIGGER_DETAILS:
                continue

            chosen_subproblems = sub_problem_selections[trig]
            if chosen_subproblems:
                any_selection = True
                st.write(f"**For Trigger: {trig}**")
                for sub_key in chosen_subproblems:
                    solution_text = TRIGGER_DETAILS[trig]["solutions"].get(sub_key, "")
                    sub_label = TRIGGER_DETAILS[trig]["subproblems"][sub_key]
                    st.markdown(f"**Sub-Problem:** {sub_label}")
                    st.markdown(f"**Suggested Approach:**\n{solution_text}\n")

        if not any_selection:
            st.info("No sub-problems were selected. Hence, no additional solutions to display.")

    # ------------- NEW Step C: “What-If” Scenario Planning -------------
    st.write("---")
    st.write("### 5. What-If Scenario Planning")
    st.write("Adjust certain factors to see how they could reduce or increase the attrition risk.")

    # Create a copy of employee_data for scenario simulation
    scenario_data = employee_data.copy()

    # Only a subset of features might realistically be changed by HR or can vary quickly.
    # For demonstration, let's pick a few key ones: Compa Ratio, Last Performance Rating, Pulse
    scenario_data["Compa Ratio"] = st.slider(
        "Scenario: Compa Ratio (%)", 50, 150, employee_data["Compa Ratio"]
    )
    scenario_data["Last Performance Rating"] = st.slider(
        "Scenario: Last Performance Rating", 1, 5, employee_data["Last Performance Rating"]
    )
    scenario_data["Pulse"] = st.radio(
        "Scenario: Pulse (Employee dissatisfaction)",
        ["High", "Medium", "Low"],
        index=["High", "Medium", "Low"].index(employee_data["Pulse"]),
        horizontal=True
    )

    # Button to "Recalculate" scenario-based risk
    if st.button("Recalculate Risk for Scenario"):
        scenario_score, scenario_triggers = predict_attrition(scenario_data)
        st.write("**Scenario Attrition Risk:** {:.2f}%".format(scenario_score))

        # Show difference from original
        score_diff = scenario_score - score
        if score_diff > 0:
            st.markdown(f"<p style='color:red;'>Risk increased by +{score_diff:.2f}% compared to original.</p>", unsafe_allow_html=True)
        elif score_diff < 0:
            st.markdown(f"<p style='color:green;'>Risk decreased by {score_diff:.2f}% compared to original.</p>", unsafe_allow_html=True)
        else:
            st.write("No change in risk compared to original scenario.")

        # Show new triggers
        if scenario_triggers:
            st.write("#### Scenario Triggers:")
            for t in scenario_triggers:
                st.markdown(f"- **{t}**")
        else:
            st.markdown("*No major negative triggers identified in this scenario.*")
