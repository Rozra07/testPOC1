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

# Train the model (comment out if you don't want it running every time)
train_and_save_model()

###############################################################################
# Step 2: TRIGGER_DETAILS for Negative Triggers (Huge Detailed Dictionary)
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
                "1. **Dedicated Female-Focused Campus Drives**: Partner with women's universities, community colleges, or professional groups to actively recruit female talent.\n"
                "2. **Scholarships & Sponsorships**: Offer scholarships or sponsorship for certification programs targeting women in technical or leadership fields.\n"
                "3. **Inclusive Employer Branding**: Feature female employees in marketing materials; highlight flexible policies, leadership opportunities, and mentorship programs."
            ),
            "lack_female_mentors": (
                "**Leadership Development & Mentoring Programs**\n\n"
                "1. **Formal Mentorship Framework**: Pair new female hires or mid-level employees with senior leaders.\n"
                "2. **Female Leadership Initiatives**: Create targeted leadership tracks for high-potential women.\n"
                "3. **Peer Circles & ERGs**: Encourage female employees to form supportive communities with meetups/workshops."
            ),
            "rigid_policies": (
                "**Flexible & Family-Friendly Policies**\n\n"
                "1. **Flexible Working Hours**: Offer part-time, remote, or hybrid models.\n"
                "2. **Enhanced Parental Leave**: Extend maternity/paternity leaves, ensure re-entry support.\n"
                "3. **On-Site/Subsidized Childcare**: If feasible, provide or partner with local childcare centers."
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
                "1. **Objective KPIs**: Define a clear set of metrics.\n"
                "2. **Promotion Review Panels**: Form cross-functional panels to mitigate bias.\n"
                "3. **Continuous Feedback**: Provide quarterly or monthly check-ins on promotion readiness."
            ),
            "no_mentorship": (
                "**Mentorship & Upskilling Pathways**\n\n"
                "1. **Formal Mentoring Program**: Assign mentors for career progression.\n"
                "2. **Upskilling Initiatives**: Offer internal courses, eLearning, or skill certifications.\n"
                "3. **Reverse Mentoring**: Pair senior leaders with junior staff for fresh ideas."
            ),
            "bureaucratic_structure": (
                "**Streamline Organizational Hierarchy**\n\n"
                "1. **Flatten Org Layers**: Consolidate overlapping departments.\n"
                "2. **Empower Frontline Managers**: Grant autonomy for promotion recommendations.\n"
                "3. **Agile Teams**: Adopt agile frameworks to help employees move up based on skill mastery."
            )
        }
    },
    # ... More triggers (Low performance rating, etc.) go here ...
    # For brevity, we've omitted them, but you can copy your entire dictionary.
}

###############################################################################
# Step 3: Rule-Based Scoring (No Changes)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
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
# Step 4: ML Probability + Weighted Factor
###############################################################################
def predict_attrition(employee_data):
    """
    Loads logistic regression model, transforms data, and combines ML probability
    with the rule-based score. Returns (combined_score, triggers).
    """
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

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_score, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    combined_score = 0.75 * rule_score + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# Step 5: Streamlit UI - Single + Bulk
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>", unsafe_allow_html=True)

### SINGLE EMPLOYEE SECTION ###

# Session state for single
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "score" not in st.session_state:
    st.session_state.score = None
if "triggers" not in st.session_state:
    st.session_state.triggers = []
if "employee_data" not in st.session_state:
    st.session_state.employee_data = {}

st.subheader("Single Employee Prediction")

with st.form("attrition_form"):
    st.write("#### Enter Single Employee Details")

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

    submit_single = st.form_submit_button("🚀 Predict Single")
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

    # LEFT: Negative triggers + sub-problems
    with col_left:
        st.write("### Key Contributing Factors (Single)")
        negative_triggers = []
        for t in triggers:
            if t in TRIGGER_DETAILS:
                negative_triggers.append(t)
                st.markdown(f"- **{t}**")
        if not negative_triggers:
            st.markdown("*No major negative triggers identified.*")

        st.write("### Sub-Problems Selection (Single)")
        sub_problem_selections = {}
        for trig in negative_triggers:
            st.write(f"**{trig}**")
            subprobs = TRIGGER_DETAILS[trig]["subproblems"]
            chosen_list = []
            for sub_key, sub_label in subprobs.items():
                chk_id = f"{trig}-{sub_key}-single"
                if st.checkbox(sub_label, key=chk_id):
                    chosen_list.append(sub_key)
            sub_problem_selections[trig] = chosen_list

        if st.button("💡 Show Solutions (Single)"):
            st.write("### Recommended Solutions (Single)")
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
                st.info("No sub-problems selected for single employee. No solutions displayed.")

    # RIGHT: What-If Scenario for single employee
    with col_right:
        st.write("### Live What-If Scenario (Single)")
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


# =============================================================================
# =========================== BULK PREDICTION SECTION ==========================
# =============================================================================

st.markdown("---")
st.subheader("Bulk/Multiple Employee Prediction")

st.write("""
Upload a **CSV or Excel** file with **exact** columns:
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

(Refer to the sample_bulk_data.csv for format)
""")

uploaded_file = st.file_uploader("Upload CSV/Excel for Bulk Attrition", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 1. Read the file
    if uploaded_file.name.endswith(".csv"):
        df_bulk = pd.read_csv(uploaded_file)
    else:
        df_bulk = pd.read_excel(uploaded_file)

    st.write("**Uploaded Data Preview:**")
    st.dataframe(df_bulk.head())

    # 2. Validate required columns
    required_cols = [
        "Employee Age", "Average Employee Age", "Gender", "Female Employee Ratio",
        "Tenure (Months)", "Pulse", "Hasn't been promoted", "Minimum Promotion Cycle",
        "College Tier Retention", "Industry Retention", "Company Type Retention",
        "Last Performance Rating", "Compa Ratio"
    ]
    missing = [col for col in required_cols if col not in df_bulk.columns]
    if missing:
        st.error(f"Missing Columns: {missing}")
    else:
        # 3. Run Bulk Prediction
        if st.button("Run Bulk Prediction"):
            scores = []
            triggers_list = []

            # For each row, run predict_attrition
            for idx, row in df_bulk.iterrows():
                emp_dict = row.to_dict()
                final_score, tlist = predict_attrition(emp_dict)
                scores.append(final_score)
                # Convert triggers to a comma-separated string
                negative_tlist = [t for t in tlist if t in TRIGGER_DETAILS]  # only negative triggers
                triggers_str = ", ".join(negative_tlist)
                triggers_list.append(triggers_str)

            df_bulk["Attrition Score"] = scores
            df_bulk["Negative Triggers"] = triggers_list

            st.success("Bulk Prediction Completed!")
            st.dataframe(df_bulk)

            # ================== CREATE SOME VISUALIZATIONS ================== #

            st.write("### Aggregate Insights")

            # Example 1: Distribution of Risk
            high_risk_count = (df_bulk["Attrition Score"] >= 75).sum()
            moderate_high_count = ((df_bulk["Attrition Score"] >= 60) & (df_bulk["Attrition Score"] < 75)).sum()
            moderate_count = ((df_bulk["Attrition Score"] >= 35) & (df_bulk["Attrition Score"] < 60)).sum()
            low_count = (df_bulk["Attrition Score"] < 35).sum()

            risk_df = pd.DataFrame({
                "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                "Count": [high_risk_count, moderate_high_count, moderate_count, low_count]
            })

            st.write("**Risk Distribution**")
            st.bar_chart(risk_df.set_index("Risk Category"))

            # Example 2: Frequency of Negative Triggers
            # Split triggers by comma, flatten them, count frequencies
            all_trigs = []
            for val in df_bulk["Negative Triggers"]:
                if pd.notna(val) and val != "":
                    splitted = [x.strip() for x in val.split(",")]
                    all_trigs.extend(splitted)

            if len(all_trigs) > 0:
                trig_series = pd.Series(all_trigs).value_counts()
                st.write("**Top Negative Triggers Across All Employees**")
                st.bar_chart(trig_series)
            else:
                st.info("No negative triggers found across the batch.")

            # =========== Drill Down: Individual Employee Analysis ============ #
            st.write("### Drill Down into Individual Results")
            # We can let user pick a row from the data
            df_bulk_reset = df_bulk.reset_index(drop=True)
            row_options = list(range(len(df_bulk_reset)))
            selected_row = st.selectbox("Select an Employee (Row Index)", row_options)

            if selected_row is not None:
                row_data = df_bulk_reset.loc[selected_row].to_dict()
                st.write("**Selected Employee’s Data**")
                st.json(row_data)

                # Show a mini re-run if we want triggers
                st.write("**Negative Triggers:**", row_data["Negative Triggers"])
                st.write("**Attrition Score:**", row_data["Attrition Score"])

                st.write("""
                *You could expand this to show sub-problems, etc. 
                But for a quick drill-down, we just display the row’s existing triggers & score.*
                """)

