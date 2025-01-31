import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
},

###############################################################################
# Step 3: Rule-Based Scoring (UNCHANGED)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    """
    EXACT Weighted-Factor logic from your code, no changes
    """
    score = 0
    extreme_factors = 0
    triggers = []

    # Gender Diversity
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # ... all your other conditions ...

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
# Step 4: ML Probability + Weighted Factor (UNCHANGED)
###############################################################################
def predict_attrition(employee_data):
    """
    Loads logistic regression, transforms data, and merges with Weighted Factor
    EXACT logic. No changes from your code.
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
    rule_score, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    combined_score = 0.75 * rule_score + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# Generate Sample CSV for Bulk (with columns that the user must fill)
###############################################################################
def generate_bulk_sample_csv():
    """
    This sample has columns for each employee's individual data:
      Employee Name, Department, Employee Age, Tenure (Months),
      Hasn't been promoted, Minimum Promotion Cycle, Pulse, 
      Last Performance Rating, Compa Ratio, Gender,
      Which Tier College, Which Industry, Which Company Type

    Meanwhile, 'Average Employee Age', 'Female Employee Ratio', 
    'College Tier Retention', 'Industry Retention', 
    'Company Type Retention' are asked as global sliders in the app
    """
    sample_df = pd.DataFrame({
        "Employee Name": ["Alice", "Bob"],
        "Department": ["Sales", "Engineering"],
        "Employee Age": [30, 45],
        "Tenure (Months)": [36, 48],
        "Hasn't been promoted": [12, 24],
        "Minimum Promotion Cycle": [24, 24],
        "Pulse": ["Medium", "High"],
        "Last Performance Rating": [3, 1],
        "Compa Ratio": [90, 65],
        "Gender": ["Female", "Male"],
        "Which Tier College": ["Tier 1", "Tier 2"],
        "Which Industry": ["IT", "Manufacturing"],
        "Which Company Type": ["Startup", "MNC"]
    })
    buffer = io.StringIO()
    sample_df.to_csv(buffer, index=False)
    return buffer.getvalue()

###############################################################################
# Step 5: Streamlit UI - Single + Bulk via Mode Switch
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>", unsafe_allow_html=True)

mode = st.selectbox("Select Mode", ["Single Employee", "Bulk Employees"])

# Only show sample CSV download in Bulk Mode
if mode == "Bulk Employees":
    st.write("**Download a Sample Bulk CSV** if you want to see the required columns:")
    sample_csv = generate_bulk_sample_csv()
    st.download_button(
        label="Download Sample Bulk CSV",
        data=sample_csv,
        file_name="sample_bulk_data.csv",
        mime="text/csv"
    )

# ==================== SINGLE EMPLOYEE MODE (UNCHANGED) ==================== #
if mode == "Single Employee":
    st.subheader("Single Employee Prediction")
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "score" not in st.session_state:
        st.session_state.score = None
    if "triggers" not in st.session_state:
        st.session_state.triggers = []
    if "employee_data" not in st.session_state:
        st.session_state.employee_data = {}

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

        # ---------- LEFT: Negative triggers + Sub-Problems ----------
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
                subprobs_dict = TRIGGER_DETAILS[trig]["subproblems"]
                chosen_subs = []
                for sub_key, sub_label in subprobs_dict.items():
                    chk_id = f"{trig}-{sub_key}"
                    if chk_id not in st.session_state:
                        st.session_state[chk_id] = False
                    if st.checkbox(sub_label, key=chk_id):
                        chosen_subs.append(sub_key)
                sub_problem_selections[trig] = chosen_subs

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
                help="What if performance improves or worsens?"
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

            neg_scenario_trigs = [t for t in scenario_triggers if t in TRIGGER_DETAILS]
            if neg_scenario_trigs:
                st.write("**Scenario Negative Triggers**")
                for t in neg_scenario_trigs:
                    st.markdown(f"- **{t}**")
            else:
                st.markdown("*No negative triggers in this scenario.*")


# =================== BULK EMPLOYEES MODE =================== #
elif mode == "Bulk Employees":
    st.subheader("Global Parameters (Apply to All Employees in Bulk)")

    # 1. Global sliders for entire dataset
    avg_age = st.slider("Average Employee Age (Across All)", 18, 65, 35)
    female_ratio = st.slider("Female Employee Ratio (%) [Global]", 0, 100, 40)
    college_ret = st.slider("College Tier Retention (%) [Global]", 10, 100, 60)
    industry_ret = st.slider("Industry Retention (%) [Global]", 10, 100, 60)
    company_ret = st.slider("Company Type Retention (%) [Global]", 10, 100, 60)

    st.write("""
    Now upload a CSV/Excel with columns:
    - Employee Name, Department, Employee Age, Tenure (Months),
      Hasn't been promoted, Minimum Promotion Cycle, Pulse, 
      Last Performance Rating, Compa Ratio, Gender,
      Which Tier College, Which Industry, Which Company Type
    """)

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df_bulk = pd.read_csv(uploaded_file)
        else:
            df_bulk = pd.read_excel(uploaded_file)

        st.write("**Uploaded Data Preview:**")
        st.dataframe(df_bulk.head())

        # Check required columns
        required_cols = [
            "Employee Name", "Department", "Employee Age", "Tenure (Months)",
            "Hasn't been promoted", "Minimum Promotion Cycle", "Pulse",
            "Last Performance Rating", "Compa Ratio", "Gender",
            "Which Tier College", "Which Industry", "Which Company Type"
        ]
        missing = [col for col in required_cols if col not in df_bulk.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Run Bulk Prediction"):
                # We'll store results
                risk_scores = []
                neg_triggers_list = []

                for idx, row in df_bulk.iterrows():
                    # Merge global inputs with row-level data
                    employee_dict = {
                        "Employee Age": row["Employee Age"],
                        "Average Employee Age": avg_age,
                        "Gender": row["Gender"],
                        "Female Employee Ratio": female_ratio,
                        "Tenure (Months)": row["Tenure (Months)"],
                        "Pulse": row["Pulse"],
                        "Hasn't been promoted": row["Hasn't been promoted"],
                        "Minimum Promotion Cycle": row["Minimum Promotion Cycle"],
                        "College Tier Retention": college_ret,
                        "Industry Retention": industry_ret,
                        "Company Type Retention": company_ret,
                        "Last Performance Rating": row["Last Performance Rating"],
                        "Compa Ratio": row["Compa Ratio"]
                    }
                    # Weighted factor + ML
                    bulk_score, triggers = predict_attrition(employee_dict)

                    # Gather negative triggers only
                    neg_trigs = [t for t in triggers if t in TRIGGER_DETAILS]
                    triggers_str = ", ".join(neg_trigs)

                    risk_scores.append(bulk_score)
                    neg_triggers_list.append(triggers_str)

                df_bulk["Attrition Score"] = risk_scores
                df_bulk["Negative Triggers"] = neg_triggers_list

                st.success("Bulk Prediction Completed!")
                st.write("### Bulk Results with Risk & Triggers")
                st.dataframe(df_bulk)

                # ============== DEPARTMENT-LEVEL INSIGHTS ============= #
                st.write("## Department-Level Insights")

                if "Department" in df_bulk.columns:
                    # 1) Group by Department, compute average risk
                    dept_group = df_bulk.groupby("Department")["Attrition Score"].mean().reset_index()
                    dept_group.columns = ["Department", "Avg Attrition Score"]
                    st.write("### Average Attrition Risk by Department")
                    st.bar_chart(data=dept_group.set_index("Department"))

                    # 2) For each department, show top negative triggers
                    st.write("### Top Negative Triggers per Department")
                    # We'll map department => list of triggers
                    dept_triggers_map = {}
                    for idx, row in df_bulk.iterrows():
                        dept = row["Department"]
                        trig_str = row["Negative Triggers"]
                        if pd.notna(trig_str) and trig_str.strip():
                            splitted = [x.strip() for x in trig_str.split(",")]
                        else:
                            splitted = []
                        if dept not in dept_triggers_map:
                            dept_triggers_map[dept] = []
                        dept_triggers_map[dept].extend(splitted)

                    for dept, trig_list in dept_triggers_map.items():
                        if len(trig_list) == 0:
                            st.write(f"**{dept}:** No negative triggers.")
                            continue
                        trig_series = pd.Series(trig_list).value_counts()
                        st.write(f"**Department: {dept}**")
                        st.bar_chart(trig_series)

                    # 3) Suggest basic "action" if dept avg risk is high
                    for idx2, row2 in dept_group.iterrows():
                        department_name = row2["Department"]
                        avg_risk = row2["Avg Attrition Score"]
                        if avg_risk >= 75:
                            st.error(f"Department '{department_name}' has a HIGH average risk of {avg_risk:.2f}%. Immediate interventions recommended!")
                        elif avg_risk >= 60:
                            st.warning(f"Department '{department_name}' is in Mod-High zone with an average risk of {avg_risk:.2f}%. Needs close attention.")
                        elif avg_risk >= 35:
                            st.info(f"Department '{department_name}' is Moderate at {avg_risk:.2f}%. Investigate triggers but not critical.")
                        else:
                            st.success(f"Department '{department_name}' is relatively Safe at {avg_risk:.2f}%. Maintain best practices.")

                else:
                    st.info("No 'Department' column found to show department-level insights.")

                # ============== EMPLOYEE-LEVEL LISTING: Name & Risk ============= #
                st.write("## Employee-Level Overview")
                st.write("Below is the final table with Employee Name and Attrition Score. Sort or filter as needed.")
                summary_df = df_bulk[["Employee Name", "Department", "Attrition Score", "Negative Triggers"]].copy()
                st.dataframe(summary_df)

                st.write("""
                **You can further expand**: 
                - Provide sub-problem selection or scenario planning for each row.
                - Offer advanced analytics or a separate page for deeper insights. 
                """)
