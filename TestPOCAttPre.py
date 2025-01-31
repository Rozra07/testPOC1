import streamlit as st
    X_scaled = scaler.transform(df_input)

    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, reasons = compute_weighted_attrition(employee_data)

    combined_score = 0.75 * rule_probability + 0.25 * ml_probability
    return combined_score, reasons

###############################################################################
# Streamlit UI
###############################################################################
st.title("Employee Attrition Prediction Tool")
st.markdown("### Identify potential attrition risks and gain insights into employee retention.")

with st.form("attrition_form"):
    employee_data = {
        "Employee Age": st.slider("Employee Age", 18, 65, 30),
        "Average Employee Age": st.slider("Avg Employee Age", 18, 65, 35),
        "Gender": st.radio("Gender", ["Male", "Female"], horizontal=True),
        "Female Employee Ratio": st.slider("Female Employee Ratio (%)", 0, 100, 40),
        "Tenure (Months)": st.slider("Tenure (Months)", 0, 240, 36),
        "Hasn't been promoted": st.slider("Months Since Last Promotion", 0, 60, 12),
        "Minimum Promotion Cycle": st.slider("Min Promotion Cycle (Months)", 12, 60, 24),
        "College Tier Retention": st.slider("College Tier Retention (%)", 10, 100, 60),
        "Industry Retention": st.slider("Industry Retention (%)", 10, 100, 60),
        "Company Type Retention": st.slider("Company Type Retention (%)", 10, 100, 60),
        "Last Performance Rating": st.slider("Last Performance Rating", 1, 5, 3),
        "Compa Ratio": st.slider("Compa Ratio (%)", 50, 150, 100)
    }
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    prediction, reasons = predict_attrition(employee_data)
    
    st.markdown("---")
    
    if prediction > 70:
        st.markdown(f"## ⚠️ High Attrition Risk: {prediction:.2f}%")
        st.write("This employee has a high likelihood of leaving the organization. Below are some potential reasons:")
        for reason in reasons:
            st.markdown(f"- {reason}")
        
        st.markdown("### 🤔 Why do you think this is happening?")
        employer_response = st.text_area("Your thoughts on the reasons above:")
        
        if st.button("Get AI Suggestions"):
            if employer_response.strip():
                st.write("### AI-Driven Suggestions for Improvement")
                st.write("Based on your insights, here are some ways to mitigate attrition risk:")
                
                if "promotion" in employer_response.lower():
                    st.markdown("- **Review Promotion Timelines:** Ensure employees see a clear career path with timely promotions.")
                if "compensation" in employer_response.lower():
                    st.markdown("- **Salary Benchmarking:** Conduct market salary analysis to ensure competitive pay.")
                if "gender" in employer_response.lower():
                    st.markdown("- **Diversity & Inclusion Programs:** Improve workplace culture to support underrepresented groups.")
                if "performance" in employer_response.lower():
                    st.markdown("- **Performance Management:** Provide regular feedback and training to improve ratings.")
                if "retention" in employer_response.lower():
                    st.markdown("- **Engagement Surveys:** Conduct employee engagement surveys to understand concerns better.")
            else:
                st.write("### Please share your thoughts above for AI-driven suggestions!")
    else:
        st.markdown(f"## ✅ Low Attrition Risk: {prediction:.2f}%")
        st.write("This employee has a low likelihood of leaving the organization. Keep up the good work!")
