import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
pip install xgboost

###############################################################################
# Step 1: Train and Save a Logistic Regression Model (UNCHANGED)
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

# (Uncomment if you only want to train the model once)
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
# Step 3: Rule-Based Scoring (No Changes)
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
# Step 4: Machine Learning Combination (No Changes)
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
# Generate Sample CSV in-memory for Bulk
###############################################################################
def generate_sample_csv():
    """
    Returns a string of CSV data containing the required columns 
    with 2 example rows.
    """
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

###############################################################################
# Step 5: Streamlit UI - Original Single Code + Bulk Code with Mode Switch
###############################################################################
st.markdown(
    "<h2 style='text-align: center; color: #141414;'>Employee Attrition Prediction Tool</h2>",
    unsafe_allow_html=True
)

# Mode Switch
mode = st.selectbox("Select Mode", ["Single Employee", "Bulk Employees"])

# Only show sample CSV download in Bulk Mode
if mode == "Bulk Employees":
    st.write("**Download a Sample Bulk CSV** if you want to see the required columns:")
    sample_csv = generate_sample_csv()
    st.download_button(
        label="Download Sample Bulk CSV",
        data=sample_csv,
        file_name="sample_bulk_data.csv",
        mime="text/csv"
    )
# ==================== SINGLE EMPLOYEE MODE (Original Code) ==================== #
if mode == "Single Employee":

    # No changes to single-employee logic or UI
    # EXACT code you shared, with the same form, sub-problem logic, scenario, etc.

    # Step 5: Streamlit UI - Single Employee
    # (We'll keep your existing code. Just placed inside 'if mode == "Single Employee":')
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
            final_score, triggers = predict_attrition(input_data)
            st.session_state.score = final_score
            st.session_state.triggers = triggers
            st.session_state.prediction_made = True
            st.session_state.employee_data = input_data

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

        col_left, col_right = st.columns(2)

        # ---------- LEFT: Triggers + Sub-Problems ----------
        with col_left:
            st.write("### Key Contributing Factors")
            negative_triggers = []
            for t in triggers:
                if t in TRIGGER_DETAILS:
                    negative_triggers.append(t)
                    st.markdown(f"- **{t}**")
                else:
                    # It's a positive or unrecognized trigger
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

            scenario_score, scenario_triggers = predict_attrition(scenario_data)
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


# ========================= BULK EMPLOYEE MODE =========================== #
elif mode == "Bulk Employees":
    st.markdown(
    "<h5 style='text-align: center; color: #FF2400;'>[NOT READY YET]</h5>",
    unsafe_allow_html=True
)
    st.write("""
    ### 📁 Bulk Employee Attrition Prediction
    Upload a **CSV or Excel** file with the following columns:
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

    # 1. Introduce Inputs for Fixed Attributes and Tier-Based Retention Percentages
    st.sidebar.header("🔧 Set Fixed Attributes and Retention Percentages")

    # a. Fixed Attributes Sliders
    st.sidebar.subheader("📊 Fixed Attributes for All Employees")
    fixed_attributes = {
        "Average Employee Age": st.sidebar.slider(
            "Average Employee Age",
            min_value=18,
            max_value=65,
            value=35,
            step=1,
            help="Set the average age of employees across the company."
        ),
        "Female Employee Ratio": st.sidebar.slider(
            "Female Employee Ratio (%)",
            min_value=0,
            max_value=100,
            value=40,
            step=1,
            help="Set the percentage of female employees in the company."
        )
    }

    # b. College Tier Retention Percentages
    st.sidebar.subheader("🏫 College Tier Retention (%)")
    college_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    college_retention = {}
    for tier in college_tiers:
        default_value = 60 if tier == "Tier 1" else (50 if tier == "Tier 2" else 40)
        college_retention[tier] = st.sidebar.slider(
            f"{tier} Retention (%)",
            min_value=10,
            max_value=100,
            value=default_value,
            step=1,
            help=f"Set the retention rate for employees from {tier} colleges."
        )

    # c. Industry Retention Percentages
    st.sidebar.subheader("🏭 Industry Retention (%)")
    # Define your actual industry categories here
    industries = ["Tech", "Finance", "Healthcare", "Education", "Manufacturing"]
    industry_retention = {}
    for industry in industries:
        default_value = 60 if industry == "Tech" else (55 if industry == "Finance" else 50)
        industry_retention[industry] = st.sidebar.slider(
            f"{industry} Retention (%)",
            min_value=10,
            max_value=100,
            value=default_value,
            step=1,
            help=f"Set the retention rate for employees from the {industry} industry."
        )

    # d. Company Type Retention Percentages
    st.sidebar.subheader("🏢 Company Type Retention (%)")
    # Define your actual company type categories here
    company_types = ["Startup", "Enterprise", "Non-Profit", "SME"]
    company_type_retention = {}
    for ctype in company_types:
        default_value = 60 if ctype == "Enterprise" else (55 if ctype == "Startup" else 50)
        company_type_retention[ctype] = st.sidebar.slider(
            f"{ctype} Retention (%)",
            min_value=10,
            max_value=100,
            value=default_value,
            step=1,
            help=f"Set the retention rate for employees from {ctype} companies."
        )

    # 2. User Guidance
    st.sidebar.markdown("""
    ---
    **🔔 Important:**  
    Please adjust the fixed attributes and retention percentages according to your organization's specifics **before** uploading and processing the bulk data.
    """)

    # 3. File Uploader
    uploaded_file = st.file_uploader("📤 Upload CSV/Excel File", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # 4. Read the Uploaded File
        try:
            if uploaded_file.name.endswith(".csv"):
                df_bulk = pd.read_csv(uploaded_file)
            else:
                df_bulk = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading the file: {e}")
            st.stop()

        st.write("**📊 Uploaded Data Preview:**")
        st.dataframe(df_bulk.head())

        # 5. Validate Columns
        required_cols = [
            "Name",
            "Employee Age",
            "Gender",
            "Chances of employee leaving (High, Medium, Low)",
            "Hasn't been promoted (in months)",
            "Min. Promotion cycle (in months)",
            "Performance rating out of 5",
            "Compa Ratio",
            "College Tier (Tier 1, Tier 2, Tier 3)",
            "Industry",
            "Company Type"
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
                    # Convert row to dictionary
                    row_dict = row.to_dict()

                    # a. Extract and store the Name
                    employee_name = row_dict.get("Name")
                    names.append(employee_name)

                    # 6. Rename Descriptive Columns to Internal Names
                    rename_mapping = {
                        "Chances of employee leaving (High, Medium, Low)": "Pulse",
                        "Hasn't been promoted (in months)": "Hasn't been promoted",
                        "Min. Promotion cycle (in months)": "Minimum Promotion Cycle",
                        "Performance rating out of 5": "Last Performance Rating",
                        "College Tier (Tier 1, Tier 2, Tier 3)": "College Tier"
                        # "Industry" and "Company Type" remain the same
                    }

                    for desc_col, internal_col in rename_mapping.items():
                        if desc_col in row_dict:
                            row_dict[internal_col] = row_dict.pop(desc_col)
                        else:
                            st.warning(f"Row {idx}: Missing column '{desc_col}'. Using default value.")
                            # Assign a default or skip processing
                            # Here, assigning a default value
                            if internal_col == "Pulse":
                                row_dict[internal_col] = "Medium"  # Default Pulse
                            elif internal_col == "Hasn't been promoted":
                                row_dict[internal_col] = 0  # Default months since promotion
                            elif internal_col == "Minimum Promotion Cycle":
                                row_dict[internal_col] = 12  # Default promotion cycle
                            elif internal_col == "Last Performance Rating":
                                row_dict[internal_col] = 3  # Default rating
                            elif internal_col == "College Tier":
                                row_dict[internal_col] = "Tier 3"  # Default tier

                    # 7. Apply Fixed Attributes from Sliders
                    row_dict["Average Employee Age"] = fixed_attributes["Average Employee Age"]
                    row_dict["Female Employee Ratio"] = fixed_attributes["Female Employee Ratio"]

                    # 8. Apply Tier-Based Retention Percentages
                    # a. College Tier Retention
                    college_tier = row_dict.get("College Tier")
                    if college_tier in college_retention:
                        row_dict["College Tier Retention"] = college_retention[college_tier]
                    else:
                        st.warning(f"Row {idx}: Unknown College Tier '{college_tier}'. Using default retention of 40%.")
                        row_dict["College Tier Retention"] = 40  # Default value

                    # b. Industry Retention
                    industry = row_dict.get("Industry")
                    if industry in industry_retention:
                        row_dict["Industry Retention"] = industry_retention[industry]
                    else:
                        st.warning(f"Row {idx}: Unknown Industry '{industry}'. Using default retention of 50%.")
                        row_dict["Industry Retention"] = 50  # Default value

                    # c. Company Type Retention
                    company_type = row_dict.get("Company Type")
                    if company_type in company_type_retention:
                        row_dict["Company Type Retention"] = company_type_retention[company_type]
                    else:
                        st.warning(f"Row {idx}: Unknown Company Type '{company_type}'. Using default retention of 50%.")
                        row_dict["Company Type Retention"] = 50  # Default value

                    # 9. Predict Attrition
                    try:
                        bulk_score, bulk_trigs = predict_attrition(row_dict)
                    except Exception as e:
                        st.error(f"Row {idx}: Prediction failed due to {e}. Skipping this row.")
                        scores.append(None)
                        triggers_list.append("Prediction Failed")
                        continue

                    scores.append(bulk_score)

                    # Filter negative triggers
                    neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
                    triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
                    triggers_list.append(triggers_str)

                # 10. Append Results to DataFrame
                df_bulk["Attrition Score"] = scores
                df_bulk["Negative Triggers"] = triggers_list

                # 11. Retain and Display the Name Column
                df_bulk["Name"] = names

                st.success("✅ Bulk Prediction Completed!")
                st.dataframe(df_bulk)

                # 12. Basic Insights
                st.write("### 📈 Aggregate Insights")

                high_risk = (df_bulk["Attrition Score"] >= 75).sum()
                mod_high = ((df_bulk["Attrition Score"] >= 60) & (df_bulk["Attrition Score"] < 75)).sum()
                moderate = ((df_bulk["Attrition Score"] >= 35) & (df_bulk["Attrition Score"] < 60)).sum()
                low = (df_bulk["Attrition Score"] < 35).sum()

                risk_df = pd.DataFrame({
                    "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                    "Count": [high_risk, mod_high, moderate, low]
                })
                st.write("**📊 Risk Distribution**")
                st.bar_chart(risk_df.set_index("Risk Category"))

                # Negative Triggers Frequency
                all_trigs = []
                for val in df_bulk["Negative Triggers"]:
                    if pd.notna(val) and val.strip() != "" and val != "None":
                        splitted = [x.strip() for x in val.split(",")]
                        all_trigs.extend(splitted)

                if all_trigs:
                    trig_series = pd.Series(all_trigs).value_counts()
                    st.write("**🔻 Top Negative Triggers**")
                    st.bar_chart(trig_series)
                else:
                    st.info("ℹ️ No negative triggers found across the batch.")

                # Drill-down into Individual Rows
                st.write("### 🔍 Drill Down into Individual Rows")
                df_bulk_reset = df_bulk.reset_index(drop=True)
                row_options = list(range(len(df_bulk_reset)))
                sel_row = st.selectbox("📌 Select an Employee (Row Index)", row_options)

                if sel_row is not None:
                    row_info = df_bulk_reset.loc[sel_row].to_dict()
                    st.write("**📄 Row Data:**")
                    st.json(row_info)
                    st.write("""
                    *You can expand this section to include more detailed analysis or individual scenario planning for each employee.*
                    """)

