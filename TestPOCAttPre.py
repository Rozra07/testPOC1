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
# Step 2: TRIGGER_DETAILS for Negative Triggers (UNCHANGED from your code)
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
                "**Recruitment Outreach & Employer Branding**\n\n"
                "1. **Dedicated Female-Focused Campus Drives**: Partner with women's universities, community colleges, or professional groups to actively recruit female talent. Showcase success stories of women in your organization.\n"
                "2. **Scholarships & Sponsorships**: Offer scholarships or sponsorship for certification programs targeting women in technical or leadership fields. This builds a talent pipeline.\n"
                "3. **Inclusive Employer Branding**: Feature female employees in your marketing and recruitment materials; highlight flexible policies, leadership opportunities, and mentorship programs."
            ),
            "lack_female_mentors": (
                "**Leadership Development & Mentoring Programs**\n\n"
                "1. **Formal Mentorship Framework**: Pair new female hires or mid-level employees with senior leaders who provide career guidance, skill development, and networking opportunities.\n"
                "2. **Female Leadership Initiatives**: Create targeted leadership tracks or development courses that help high-potential women gain visibility and executive skills.\n"
                "3. **Peer Circles & ERGs (Employee Resource Groups)**: Encourage female employees to form supportive communities. Sponsor regular meetups, workshops, or knowledge-sharing sessions to promote solidarity."
            ),
            "rigid_policies": (
                "**Flexible & Family-Friendly Policies**\n\n"
                "1. **Flexible Working Hours**: Offer part-time, remote, or hybrid models to accommodate different life stages, including childcare or eldercare responsibilities.\n"
                "2. **Enhanced Parental Leave**: Extend maternity and paternity leaves, and ensure re-entry support for returning parents, such as transitional part-time options.\n"
                "3. **On-Site or Subsidized Childcare**: If feasible, provide in-house daycare or partner with local childcare centers. This significantly improves retention for working parents."
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
                "**Transparent Promotion Framework**\n\n"
                "1. **Objective KPIs**: Define a clear set of metrics (e.g., revenue impact, project success rates, leadership traits) so employees know exactly what’s needed for promotion.\n"
                "2. **Promotion Review Panels**: Form cross-functional panels to mitigate bias. Publicize the panel’s membership and how decisions are reached.\n"
                "3. **Continuous Feedback Mechanisms**: Avoid once-a-year evaluations. Instead, provide quarterly or monthly check-ins on promotion readiness."
            ),
            "no_mentorship": (
                "**Mentorship & Upskilling Pathways**\n\n"
                "1. **Formal Mentoring Program**: Assign each new or mid-career employee a seasoned mentor who can guide them in career progression.\n"
                "2. **Upskilling Initiatives**: Offer internal courses, eLearning subscriptions, or skill certifications. Tie these to real promotion opportunities.\n"
                "3. **Reverse Mentoring**: Pair senior leaders with junior staff to exchange fresh ideas (tech-savviness) and institutional knowledge (strategic thinking). This fosters mutual learning."
            ),
            "bureaucratic_structure": (
                "**Streamline Organizational Hierarchy**\n\n"
                "1. **Flatten Org Layers**: Consolidate overlapping departments or reduce hierarchical tiers to speed decision-making.\n"
                "2. **Empower Frontline Managers**: Grant more autonomy for promotion recommendations at the local/department level.\n"
                "3. **Agile or Cross-Functional Teams**: Adopt agile frameworks where employees can move up based on skill mastery rather than waiting for openings in a rigid org chart."
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
                "**Role Alignment & Expectation Management**\n\n"
                "1. **Detailed JD & Goals**: Provide a clear job description and define specific, measurable objectives aligned with business goals.\n"
                "2. **Job Realignment**: If an employee’s strengths are better suited elsewhere, consider an internal transfer. Encourage managers to spot potential role mismatches early.\n"
                "3. **Regular Pulse Checks**: Schedule monthly or quarterly touchpoints to confirm that the role still fits the employee’s evolving interests and competencies."
            ),
            "no_feedback": (
                "**Frequent 1-on-1 Sessions & Real-Time Feedback**\n\n"
                "1. **Weekly or Bi-Weekly 1-on-1s**: Ensure managers discuss performance, challenges, and goals. Provide immediate course corrections or praise.\n"
                "2. **Performance Dashboards**: Implement a real-time metric or scoreboard that employees can view to track their KPIs.\n"
                "3. **Peer Feedback Loops**: Encourage peer reviews or 360° feedback sessions to give employees a well-rounded perspective of their performance."
            ),
            "skill_gaps": (
                "**Targeted Training & Growth Plans**\n\n"
                "1. **Skill Matrix Assessment**: Identify critical skill gaps through structured testing or observation. Align training modules with these needs.\n"
                "2. **Sponsored Certifications**: Cover costs for professional certifications related to the employee’s role. Offer time off for study.\n"
                "3. **Buddy or Mentorship**: Pair the employee with a more experienced colleague to provide day-to-day skill guidance and coaching."
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
                "**Role Optimization & Clear Objectives**\n\n"
                "1. **Reevaluate Role Fit**: Conduct a mini-audit of responsibilities to ensure the employee’s core strengths align with tasks.\n"
                "2. **SMART Goals**: (Specific, Measurable, Achievable, Relevant, Time-Bound) for each quarter. Track progress in a transparent system.\n"
                "3. **Collaborative Task Assignment**: Let employees volunteer for projects that interest them. This often boosts engagement and performance."
            ),
            "no_feedback": (
                "**Structured Feedback & Coaching**\n\n"
                "1. **Regular 1:1 Coaching**: Institute weekly or bi-weekly sessions where managers discuss current work, roadblocks, and improvements.\n"
                "2. **Instant Recognition Tools**: Acknowledge small wins or correct issues in real time (e.g., Slack kudos, short manager check-ins).\n"
                "3. **360-Degree Reviews**: Expand beyond manager feedback to peers, direct reports (if any), and cross-functional teams to get a holistic view."
            ),
            "skill_gaps": (
                "**Learning & Development Interventions**\n\n"
                "1. **Needs Assessment**: Use employee surveys or manager feedback to pinpoint exact skills the employee lacks.\n"
                "2. **Microlearning Modules**: Provide short, focused eLearning segments employees can complete during breaks or off-peak hours.\n"
                "3. **Mentorship & Cross-Training**: Rotate employees through different roles or departments to broaden their competence."
            )
        }
    },

    # (You continue with the rest of your negative triggers dictionary here...)
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
    sample_data = pd.DataFrame({
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
    sample_data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

###############################################################################
# Step 5: Streamlit UI - Original Single Code + Bulk Code with Mode Switch
###############################################################################
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>",
    unsafe_allow_html=True
)

# Mode Switch
mode = st.selectbox("Select Mode", ["Single Employee", "Bulk Employees"])

# Provide download for sample bulk CSV
st.write("**Download a Sample Bulk CSV** if you want to see the required format:")
sample_csv = generate_sample_csv()
st.download_button("Download Sample CSV", sample_csv, file_name="sample_bulk_data.csv", mime="text/csv")

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
    st.write("""
    Upload a **CSV or Excel** file with the exact columns:
    - Employee Age
    - Average Employee Age
    - Gender
    - Female Employee Ratio
    - Tenure (Months)
    - Pulse
    - Hasn't been promoted
    - Minimum Promotion Cycle
    - College Tier Retention
    - Industry Retention
    - Company Type Retention
    - Last Performance Rating
    - Compa Ratio
    """)

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # 1. Read file
        if uploaded_file.name.endswith(".csv"):
            df_bulk = pd.read_csv(uploaded_file)
        else:
            df_bulk = pd.read_excel(uploaded_file)

        st.write("**Uploaded Data Preview:**")
        st.dataframe(df_bulk.head())

        # 2. Validate columns
        required_cols = [
            "Employee Age", "Average Employee Age", "Gender", "Female Employee Ratio",
            "Tenure (Months)", "Pulse", "Hasn't been promoted", "Minimum Promotion Cycle",
            "College Tier Retention", "Industry Retention", "Company Type Retention",
            "Last Performance Rating", "Compa Ratio"
        ]
        missing = [c for c in required_cols if c not in df_bulk.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Run Bulk Prediction"):
                scores = []
                triggers_list = []

                for idx, row in df_bulk.iterrows():
                    row_dict = row.to_dict()
                    bulk_score, bulk_trigs = predict_attrition(row_dict)
                    scores.append(bulk_score)
                    neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
                    triggers_str = ", ".join(neg_trigs)
                    triggers_list.append(triggers_str)

                df_bulk["Attrition Score"] = scores
                df_bulk["Negative Triggers"] = triggers_list

                st.success("Bulk Prediction Completed!")
                st.dataframe(df_bulk)

                # Basic insights
                st.write("### Aggregate Insights")

                high_risk = (df_bulk["Attrition Score"] >= 75).sum()
                mod_high = ((df_bulk["Attrition Score"] >= 60) & (df_bulk["Attrition Score"] < 75)).sum()
                moderate = ((df_bulk["Attrition Score"] >= 35) & (df_bulk["Attrition Score"] < 60)).sum()
                low = (df_bulk["Attrition Score"] < 35).sum()

                risk_df = pd.DataFrame({
                    "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                    "Count": [high_risk, mod_high, moderate, low]
                })
                st.write("**Risk Distribution**")
                st.bar_chart(risk_df.set_index("Risk Category"))

                # Negative triggers frequency
                all_trigs = []
                for val in df_bulk["Negative Triggers"]:
                    if pd.notna(val) and val.strip() != "":
                        splitted = [x.strip() for x in val.split(",")]
                        all_trigs.extend(splitted)

                if all_trigs:
                    trig_series = pd.Series(all_trigs).value_counts()
                    st.write("**Top Negative Triggers**")
                    st.bar_chart(trig_series)
                else:
                    st.info("No negative triggers found across the batch.")

                # Drill-down
                st.write("### Drill Down into Individual Row")
                df_bulk_reset = df_bulk.reset_index(drop=True)
                row_options = list(range(len(df_bulk_reset)))
                sel_row = st.selectbox("Pick an Employee (Row Index)", row_options)

                if sel_row is not None:
                    row_info = df_bulk_reset.loc[sel_row].to_dict()
                    st.write("**Row Data**:")
                    st.json(row_info)
                    st.write("""
                    *You could expand to let the user see sub-problem selections for each row
                    or scenario planning in a batch context, but this is a basic overview.*
                    """)

