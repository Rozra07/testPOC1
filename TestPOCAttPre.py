import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import io  # for creating in-memory CSV/Excel

###############################################################################
# Step 1: Train and Save a Logistic Regression Model (unchanged)
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
# Step 2: TRIGGER_DETAILS (Large Dictionary)
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
                "1. **Dedicated Female-Focused Campus Drives**: ...\n"
                "2. **Scholarships & Sponsorships**: ...\n"
                "3. **Inclusive Employer Branding**: ..."
            ),
            "lack_female_mentors": (
                "**Leadership Development & Mentoring Programs**\n\n"
                "1. **Formal Mentorship Framework**: ...\n"
                "2. **Female Leadership Initiatives**: ...\n"
                "3. **Peer Circles & ERGs**: ..."
            ),
            "rigid_policies": (
                "**Flexible & Family-Friendly Policies**\n\n"
                "1. **Flexible Working Hours**: ...\n"
                "2. **Enhanced Parental Leave**: ...\n"
                "3. **On-Site or Subsidized Childcare**: ..."
            )
        }
    },
    # ... Include your entire dictionary for other triggers ...
}

###############################################################################
# Step 3: Rule-Based Scoring
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []

    # Gender
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # ... (Your entire logic for other triggers) ...

    final_score = min(100, max(0, score))
    if return_triggers:
        return final_score, triggers
    else:
        return final_score

###############################################################################
# Step 4: ML Probability + Weighted Factor
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
    rule_score, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    combined_score = 0.75 * rule_score + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# Helper: Create a sample CSV in memory for the user to download
###############################################################################
def generate_sample_csv():
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
# Step 5: Streamlit UI with Mode Selector
###############################################################################
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>", unsafe_allow_html=True)

mode = st.selectbox("Select Mode", ["Single Employee", "Bulk Employees"])

# Provide a download button for the sample CSV
st.write("**Download the Sample Bulk File**")
sample_csv = generate_sample_csv()
st.download_button(
    label="Download Sample CSV",
    data=sample_csv,
    file_name="sample_bulk_data.csv",
    mime="text/csv"
)

# -------------------- SINGLE EMPLOYEE MODE -------------------- #
if mode == "Single Employee":
    st.subheader("Single Employee Prediction")

    # single session states
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "score" not in st.session_state:
        st.session_state.score = None
    if "triggers" not in st.session_state:
        st.session_state.triggers = []
    if "employee_data" not in st.session_state:
        st.session_state.employee_data = {}

    with st.form("single_form"):
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

        # RIGHT: What-If Scenario
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

            scenario_score, scenario_trigs = predict_attrition(scenario_data)
            st.write(f"**Scenario Attrition Risk:** {scenario_score:.2f}%")

            diff = scenario_score - score
            if diff > 0:
                st.markdown(f"<span style='color:red;'>Risk +{diff:.2f}% higher than original.</span>", unsafe_allow_html=True)
            elif diff < 0:
                st.markdown(f"<span style='color:green;'>Risk {diff:.2f}% lower than original.</span>", unsafe_allow_html=True)
            else:
                st.write("No change from original risk.")

            neg_scenario_trigs = [t for t in scenario_trigs if t in TRIGGER_DETAILS]
            if neg_scenario_trigs:
                st.write("**Scenario Negative Triggers**")
                for t in neg_scenario_trigs:
                    st.markdown(f"- **{t}**")
            else:
                st.markdown("*No negative triggers in this scenario.*")


# -------------------- BULK EMPLOYEES MODE -------------------- #
elif mode == "Bulk Employees":
    st.subheader("Bulk/Multiple Employee Prediction")

    st.write("Upload a **CSV or Excel** file with **exact** columns listed above (or see the sample).")

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
                    negative_tlist = [t for t in tlist if t in TRIGGER_DETAILS]
                    triggers_str = ", ".join(negative_tlist)
                    triggers_list.append(triggers_str)

                df_bulk["Attrition Score"] = scores
                df_bulk["Negative Triggers"] = triggers_list

                st.success("Bulk Prediction Completed!")
                st.dataframe(df_bulk)

                # ============= CREATE SOME VISUALIZATIONS ============== #
                st.write("### Aggregate Insights")

                # Risk Distribution
                high_risk_count = (df_bulk["Attrition Score"] >= 75).sum()
                mod_high_count = ((df_bulk["Attrition Score"] >= 60) & (df_bulk["Attrition Score"] < 75)).sum()
                mod_count = ((df_bulk["Attrition Score"] >= 35) & (df_bulk["Attrition Score"] < 60)).sum()
                low_count = (df_bulk["Attrition Score"] < 35).sum()

                risk_df = pd.DataFrame({
                    "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                    "Count": [high_risk_count, mod_high_count, mod_count, low_count]
                })

                st.write("**Risk Distribution**")
                st.bar_chart(risk_df.set_index("Risk Category"))

                # Negative Triggers Frequency
                all_trigs = []
                for val in df_bulk["Negative Triggers"]:
                    if pd.notna(val) and val.strip() != "":
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
                df_bulk_reset = df_bulk.reset_index(drop=True)
                row_options = list(range(len(df_bulk_reset)))
                selected_row = st.selectbox("Select an Employee (Row Index)", row_options)

                if selected_row is not None:
                    row_data = df_bulk_reset.loc[selected_row].to_dict()
                    st.write("**Selected Employee’s Data**:")
                    st.json(row_data)

                    st.write("**Negative Triggers:**", row_data["Negative Triggers"])
                    st.write("**Attrition Score:**", row_data["Attrition Score"])

                    st.write("""
                    You could extend this to show sub-problems / scenario planning for each row, 
                    but for now, we just display basic info.
                    """)
