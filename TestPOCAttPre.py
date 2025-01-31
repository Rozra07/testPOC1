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
    # Same code as your original
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

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=["Gender", "Pulse"])
    feature_columns = df_encoded.columns

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # Train logistic regression
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    # Save model, scaler, and feature columns
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(feature_columns), f)

# Run training once (you can comment this out after the first run to avoid overwriting)
train_and_save_model()

###############################################################################
# Step 2: Rule-Based Scoring (Including All Conditions & Trigger Collection)
###############################################################################
def compute_weighted_attrition(employee, return_triggers=False):
    """
    Computes the rule-based score and optionally returns triggers (factors)
    that increased or decreased the score.
    """
    score = 0
    extreme_factors = 0
    triggers = []  # Will store text explanations for each triggered condition

    # -------------------------------------------------------------------------
    # Condition 1: Gender Diversity
    # -------------------------------------------------------------------------
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30
        extreme_factors += 1
        triggers.append("Low gender diversity")

    # -------------------------------------------------------------------------
    # Condition 2: Stagnant Promotions
    # -------------------------------------------------------------------------
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30
        extreme_factors += 1
        triggers.append("Stagnant promotions")

    # -------------------------------------------------------------------------
    # Condition 3: Performance Ratings
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Condition 4: Compensation Ratio
    # -------------------------------------------------------------------------
    if employee["Compa Ratio"] < 70:
        score += 30
        extreme_factors += 1
        triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15
        extreme_factors -= 0.5
        triggers.append("High compensation ratio")

    # -------------------------------------------------------------------------
    # Condition 5: Retention Metrics
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Condition 6: Pulse
    # -------------------------------------------------------------------------
    if employee["Pulse"] == "High":
        score += 20
        extreme_factors += 0.5
        triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20
        extreme_factors -= 0.5
        triggers.append("Low dissatisfaction (Pulse)")

    # -------------------------------------------------------------------------
    # Extreme Factor Scaling
    # -------------------------------------------------------------------------
    if extreme_factors == 2:
        score = min(100, score * 1.3)
    elif extreme_factors == 3:
        score = min(100, score * 1.6)
    elif extreme_factors >= 4:
        score = min(100, score * 2)

    # Final Clamping
    final_score = min(100, max(0, score))

    # Return triggers if requested
    if return_triggers:
        return final_score, triggers
    else:
        return final_score

###############################################################################
# Step 3: Machine Learning Prediction (UNCHANGED, except we now extract triggers)
###############################################################################
def predict_attrition(employee_data):
    """
    Returns the combined attrition score (ML + Rule-based) and
    also the triggers that contributed to the rule-based risk.
    """
    # Load model artifacts
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Prepare input for ML
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)

    # ML Probability
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100

    # Rule-based Score + triggers
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)

    # Combine Score
    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, triggers

###############################################################################
# Step 4: Explanation / Recommendation Logic
###############################################################################
def provide_explanations(triggers, industry):
    """
    For each triggered condition, provide tailored recommendations or insights.
    We'll do it systematically by checking if each condition is in triggers,
    then referencing the selected 'industry'.
    """
    # -- Low gender diversity -------------------------------------------------
    if "Low gender diversity" in triggers:
        if industry == "Manufacturing":
            st.write(
                "**Low Gender Diversity in Manufacturing:**\n"
                "- Historically, manufacturing has underrepresented female employees.\n"
                "- Consider targeted recruitment, mentorship programs, and improved workplace amenities.\n"
                "- Example: XYZ Corp increased female hires by 20% via specialized campus programs."
            )
        elif industry == "IT/ITES":
            st.write(
                "**Low Gender Diversity in IT/ITES:**\n"
                "- Tech roles often skew male, especially in senior engineering.\n"
                "- Sponsor or partner with programs that support women in tech (e.g., Women Who Code).\n"
                "- Encourage leadership training for mid-level female employees."
            )
        elif industry == "BFSI":
            st.write(
                "**Low Gender Diversity in BFSI:**\n"
                "- Financial services can benefit from bridging the female leadership gap.\n"
                "- Offer flexible policies for working mothers, sponsor women leadership forums.\n"
                "- Case study: ABC Bank improved gender ratio by introducing flexible hours + childcare."
            )
        elif industry == "Retail":
            st.write(
                "**Low Gender Diversity in Retail:**\n"
                "- Retail frontline roles can be balanced, but back-office/leadership may be skewed.\n"
                "- Consider leadership development tracks for female employees.\n"
                "- Mentorship programs and clear promotion paths help retention."
            )
        else:
            st.write(
                "**Low Gender Diversity (General):**\n"
                "- Focus on inclusive hiring, mentorship, and flexible work arrangements.\n"
                "- Engage with diversity consultancies and track retention metrics by gender.\n"
            )

    # -- Stagnant promotions --------------------------------------------------
    if "Stagnant promotions" in triggers:
        if industry == "Manufacturing":
            st.write(
                "**Stagnant Promotions in Manufacturing:**\n"
                "- Rotational programs can help employees gain cross-functional skills.\n"
                "- Recognize high performers with on-floor responsibilities or team lead roles.\n"
            )
        elif industry == "IT/ITES":
            st.write(
                "**Stagnant Promotions in IT/ITES:**\n"
                "- Consider shorter promotion cycles in agile environments.\n"
                "- Provide certification/training credits to encourage skill growth.\n"
            )
        elif industry == "BFSI":
            st.write(
                "**Stagnant Promotions in BFSI:**\n"
                "- Implement transparent promotion criteria tied to clear KPIs.\n"
                "- Offer cross-training in different financial products for skill variety.\n"
            )
        elif industry == "Retail":
            st.write(
                "**Stagnant Promotions in Retail:**\n"
                "- Identify and fast-track high-potential employees for store or regional management.\n"
                "- Introduce clear leadership track with quarterly reviews.\n"
            )
        else:
            st.write(
                "**Stagnant Promotions (General):**\n"
                "- Publish clear promotion guidelines.\n"
                "- Offer mentorship and skill development to pave a clear path upward.\n"
            )

    # -- Very low performance rating / Low performance rating -----------------
    if "Very low performance rating" in triggers or "Low performance rating" in triggers:
        if industry == "Manufacturing":
            st.write(
                "**Performance Improvement in Manufacturing:**\n"
                "- Use frequent on-floor observations & skill-based training.\n"
                "- Implement buddy systems for new hires and continuous improvement programs.\n"
            )
        elif industry == "IT/ITES":
            st.write(
                "**Performance Improvement in IT/ITES:**\n"
                "- Leverage agile methodology for short feedback loops.\n"
                "- Provide upskilling or certifications in trending tech domains.\n"
            )
        elif industry == "BFSI":
            st.write(
                "**Performance Improvement in BFSI:**\n"
                "- Ensure employees have clarity on key financial targets.\n"
                "- Provide product & compliance training to reduce errors and increase confidence.\n"
            )
        elif industry == "Retail":
            st.write(
                "**Performance Improvement in Retail:**\n"
                "- Regularly measure customer service metrics.\n"
                "- Coach employees on sales techniques and product knowledge.\n"
            )
        else:
            st.write(
                "**Performance Improvement (General):**\n"
                "- Implement frequent 1:1 check-ins and set SMART goals.\n"
                "- Offer personal development plans and targeted training programs.\n"
            )

    # -- Excellent performance rating (positive factor) -----------------------
    if "Excellent performance rating" in triggers:
        st.write(
            "**Excellent Performance Rating:**\n"
            "- This factor *reduces* the attrition risk by showing strong engagement.\n"
            "- Continue recognition and career growth opportunities to retain top performers.\n"
        )

    # -- Low compensation competitiveness / High compensation ratio -----------
    if "Low compensation competitiveness" in triggers:
        if industry == "Manufacturing":
            st.write(
                "**Low Compensation in Manufacturing:**\n"
                "- Conduct market benchmarking to ensure wages are competitive.\n"
                "- Introduce productivity-based bonuses or skill allowances.\n"
            )
        elif industry == "IT/ITES":
            st.write(
                "**Low Compensation in IT/ITES:**\n"
                "- Tech roles are highly competitive.\n"
                "- Offer stock options, flexible hours, or remote work perks if direct salary hikes aren't feasible.\n"
            )
        elif industry == "BFSI":
            st.write(
                "**Low Compensation in BFSI:**\n"
                "- Provide performance-based incentives or profit-sharing.\n"
                "- Benchmark roles against industry peers for fairness.\n"
            )
        elif industry == "Retail":
            st.write(
                "**Low Compensation in Retail:**\n"
                "- Retail tends to have high turnover if wages are below market.\n"
                "- Offer bonus structures tied to store/individual performance.\n"
            )
        else:
            st.write(
                "**Low Compensation Competitiveness (General):**\n"
                "- Ensure compensation aligns with market rates.\n"
                "- Consider total rewards approach (benefits, bonuses, flexibility) beyond base pay.\n"
            )

    if "High compensation ratio" in triggers:
        st.write(
            "**High Compensation Ratio:**\n"
            "- This *reduces* attrition risk, as employees are well-paid compared to market.\n"
            "- Maintain fairness internally, ensuring no major pay gaps.\n"
        )

    # -- Low college tier / industry / company type retention -----------------
    if "Low college tier retention" in triggers:
        st.write(
            "**Low College Tier Retention:**\n"
            "- Possibly indicates that hires from certain colleges leave quickly.\n"
            "- Investigate if those employees face skill or culture gaps.\n"
            "- Offer targeted onboarding or mentorship for new grads.\n"
        )

    if "Low industry retention" in triggers:
        st.write(
            "**Low Industry Retention:**\n"
            "- Suggests employees from your industry have shorter tenures.\n"
            "- Evaluate if there's intense competition or frequent poaching.\n"
            "- Provide career paths, skill enhancement, and leadership opportunities to retain them.\n"
        )

    if "Low company type retention" in triggers:
        st.write(
            "**Low Company Type Retention:**\n"
            "- Possibly employees from certain corporate backgrounds exit sooner.\n"
            "- Revisit your company's onboarding, culture assimilation, or role clarity.\n"
            "- Pair them with mentors who understand that specific background.\n"
        )

    # -- High dissatisfaction (Pulse) / Low dissatisfaction (Pulse) ----------
    if "High dissatisfaction (Pulse)" in triggers:
        st.write(
            "**High Dissatisfaction (Pulse):**\n"
            "- Pulse surveys indicate employees are unhappy.\n"
            "- Explore open-ended feedback, 1-on-1 check-ins, and quick-win changes.\n"
            "- Employee recognition programs or mental health support can help.\n"
        )

    if "Low dissatisfaction (Pulse)" in triggers:
        st.write(
            "**Low Dissatisfaction (Pulse):**\n"
            "- This factor *reduces* attrition risk.\n"
            "- Maintain good practices: feedback loops, engagement activities, and recognition.\n"
        )

###############################################################################
# Step 5: Streamlit UI
###############################################################################
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🌟 Employee Attrition Prediction Tool 🚀</h2>",
    unsafe_allow_html=True
)

with st.form("attrition_form"):
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

    # Add a new field for Industry
    industry = st.selectbox(
        "Which Industry Are You From?",
        ["Manufacturing", "BFSI", "IT/ITES", "Retail", "Others"],
        index=4
    )

    submit_button = st.form_submit_button("🚀 Predict")

if submit_button:
    # 1. Make the Prediction
    prediction, triggers = predict_attrition(employee_data)

    # 2. Display the Risk Category
    if prediction >= 75:
        st.markdown(
            f'<div style="background-color:#ff4d4d; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ HIGH Attrition Risk! <br> {prediction:.2f}% 🚨</div>',
            unsafe_allow_html=True
        )
    elif 60 <= prediction < 75:
        st.markdown(
            f'<div style="background-color:#ff9933; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚠️ Moderate to High Risk <br> {prediction:.2f}% ⚡</div>',
            unsafe_allow_html=True
        )
    elif 35 <= prediction < 60:
        st.markdown(
            f'<div style="background-color:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'⚖️ Moderate Attrition Risk <br> {prediction:.2f}% 📉</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color:#28a745; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;">'
            f'✅ SAFE! Low Attrition Risk <br> {prediction:.2f}% 🌱</div>',
            unsafe_allow_html=True
        )

    # 3. Show Triggers (i.e., 'Reasons' for the Score)
    st.subheader("Key Contributing Factors to Attrition Risk")
    if triggers:
        for t in triggers:
            st.markdown(f"- **{t}**")
    else:
        st.markdown("*No major negative triggers identified.*")

    # 4. Provide Explanations and Recommendations Based on Those Triggers
    st.subheader("Customized Explanations & Recommendations")
    provide_explanations(triggers, industry)
