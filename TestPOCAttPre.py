import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import altair as alt
import matplotlib.pyplot as plt

# =============================
# Define and enable a dark Altair theme
# =============================
def dark_theme():
    return {
        "config": {
            "background": "black",
            "view": {"fill": "black"},
            "title": {"color": "white"},
            "axis": {
                "domainColor": "white",
                "gridColor": "#444444",
                "labelColor": "white",
                "titleColor": "white"
            },
            "legend": {"labelColor": "white", "titleColor": "white"}
        }
    }
alt.themes.register("dark_theme", dark_theme)
alt.themes.enable("dark_theme")

# =============================
# Page configuration
# =============================
st.set_page_config(layout="wide")

# =============================
# Custom CSS for an engaging design
# =============================
st.markdown("""
    <style>
      .cluster-card {
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 8px;
          background-color: #222;
          color: white;
          margin: 5px;
      }
      .custom-cluster-card {
          border: 2px solid #4c78a8;
          padding: 10px;
          border-radius: 8px;
          background-color: #333;
          color: white;
      }
      .header-title {
          text-align: center;
          color: white;
      }
    </style>
    """, unsafe_allow_html=True)

# =============================
# Helper Functions
# =============================
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Refresh functionality is not available. Please update Streamlit (>=0.65.0).")

def colored_metric(value, threshold, higher_better=True, is_lower=False):
    if is_lower:
        color = "green" if value <= threshold else "red"
    else:
        color = "green" if value >= threshold else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.2f}</span>'

# =============================
# User Storage: Load/Save Users and Events
# =============================
USERS_FILE = "users.json"
USER_DATA_DIR = "user_data"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def save_user_event(email, event_type, event_data):
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR)
    history_file = os.path.join(USER_DATA_DIR, f"{email}_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "event_data": event_data
    }
    history.append(event)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

def load_user_history(email):
    history_file = os.path.join(USER_DATA_DIR, f"{email}_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    else:
        return []

# =============================
# Global Variables & Options
# =============================
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]

# =============================
# Automatic Cluster Generation
# =============================
def generate_automatic_clusters(df):
    if df.empty:
        return pd.DataFrame()
    df_auto = df.copy()
    # Create Age Group using bins
    bins = [18, 30, 40, 50, 60, 100]
    labels = ["18-30", "31-40", "41-50", "51-60", "61+"]
    df_auto["Age Group"] = pd.cut(df_auto["Employee Age"], bins=bins, labels=labels, right=False)
    
    # Extract primary negative trigger
    def get_primary(trigger_str):
        if pd.isna(trigger_str) or trigger_str.strip() == "" or trigger_str.strip().lower() == "none":
            return np.nan
        else:
            return trigger_str.split(",")[0].strip()
    df_auto["Primary Trigger"] = df_auto["Negative Triggers"].apply(get_primary)
    df_auto = df_auto.dropna(subset=["Primary Trigger"])
    
    # Group by Gender, Age Group, and Primary Trigger
    clusters = df_auto.groupby(["Gender", "Age Group", "Primary Trigger"]).agg({"Attrition Score": "mean", "Name": "count"}).reset_index()
    clusters.rename(columns={"Name": "Count"}, inplace=True)
    
    # Filter and sort clusters
    clusters = clusters[(clusters["Count"] >= 3) & (clusters["Attrition Score"] >= 60)]
    clusters = clusters.sort_values(by="Attrition Score", ascending=False).head(9)
    return clusters

# =============================
# Model Training and Persistence
# =============================
def train_model(training_df, target_column, industry):
    st.write("Training on data shape:", training_df.shape)
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)
    st.write("Model coefficients:", model.coef_)
    
    # Model evaluation
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, log_loss, average_precision_score
    preds = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, model.predict(X_scaled))
    report = classification_report(y, model.predict(X_scaled), output_dict=True)
    
    st.subheader("Model Evaluation Metrics")
    st.write(f"**ROC AUC:** {roc_auc:.2f}")
    
    fig, ax = plt.subplots(facecolor='black')
    ax.set_facecolor('black')
    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})", color='cyan')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate", color='white')
    ax.set_ylabel("True Positive Rate", color='white')
    ax.set_title("ROC Curve", color='white')
    ax.legend(loc="best", facecolor='black', edgecolor='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    
    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    
    st.write("**Classification Report:**")
    st.json(report)
    
    # Trustworthiness Metrics
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp) if (tn+fp) > 0 else 0
    pr_auc = average_precision_score(y, preds)
    logloss = log_loss(y, model.predict_proba(X_scaled))
    
    st.session_state.trust_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "logloss": logloss
    }
    
    trust_table = f"""
    <table style="width:100%; border: 1px solid white; border-collapse: collapse;">
      <tr>
        <th style="border: 1px solid white; padding: 8px;">Metric</th>
        <th style="border: 1px solid white; padding: 8px;">Value</th>
        <th style="border: 1px solid white; padding: 8px;">Ideal</th>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Accuracy</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(accuracy, 0.8)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.8</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Precision</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(precision, 0.7)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Recall (Sensitivity)</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(recall, 0.7)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Specificity</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(specificity, 0.7)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">F1 Score</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(f1, 0.7)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">ROC AUC</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(roc_auc, 0.8)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.8</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Precision-Recall AUC</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(pr_auc, 0.7)}</td>
        <td style="border: 1px solid white; padding: 8px;">&gt;= 0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid white; padding: 8px;">Log Loss</td>
        <td style="border: 1px solid white; padding: 8px;">{colored_metric(logloss, 0.5, is_lower=True)}</td>
        <td style="border: 1px solid white; padding: 8px;">&lt;= 0.5</td>
      </tr>
    </table>
    """
    st.markdown("### Trustworthiness of Model")
    st.markdown(trust_table, unsafe_allow_html=True)
    
    # Save model, scaler, and feature columns
    model_filename = f"{industry}_model.pkl"
    scaler_filename = f"{industry}_scaler.pkl"
    features_filename = f"{industry}_feature_columns.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_filename, "wb") as f:
        pickle.dump(feature_columns, f)
    st.success("Model trained and saved successfully!")
    training_accuracy = model.score(X_scaled, y) * 100
    st.info(f"Training Accuracy (Confidence): {training_accuracy:.2f}%")
    
    update_industry_record(industry, model_filename, scaler_filename, features_filename)
    
    # Save global settings to user record
    user = st.session_state.user
    user_settings = user.get("settings") or {}
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3
    user_settings["bulk_industry_retention"] = {
        ind: st.session_state.get(f"bulk_ind_{ind}", 60 if ind=="Tech" else 50)
        for ind in industry_options
    }
    user_settings["bulk_company_retention"] = {
        "Startup": st.session_state.bulk_startup,
        "Small Size": st.session_state.bulk_small,
        "Mid Size": st.session_state.bulk_mid,
        "MNC/Giant Company": st.session_state.bulk_mnc
    }
    user["settings"] = user_settings
    users = load_users()
    users[user["email"]] = user
    save_users(users)
    save_user_event(user["email"], "training", {"action": "Model retrained", "industry": industry})

def update_industry_record(industry, model_file, scaler_file, feature_file):
    record = {
        "Industry": industry,
        "Model_File": model_file,
        "Scaler_File": scaler_file,
        "Feature_File": feature_file,
        "Training_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    csv_filename = "industry_models.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if industry in df["Industry"].values:
            df.loc[df["Industry"] == industry, ["Model_File", "Scaler_File", "Feature_File", "Training_Date"]] = \
                [model_file, scaler_file, feature_file, record["Training_Date"]]
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(csv_filename, index=False)

def load_model(industry):
    model_filename = f"{industry}_model.pkl"
    scaler_filename = f"{industry}_scaler.pkl"
    features_filename = f"{industry}_feature_columns.pkl"
    if os.path.exists(model_filename) and os.path.exists(scaler_filename) and os.path.exists(features_filename):
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        with open(scaler_filename, "rb") as f:
            scaler = pickle.load(f)
        with open(features_filename, "rb") as f:
            feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    else:
        st.error("No trained model found for the selected industry. Please train your model in Train Mode first.")
        return None, None, None

# =============================
# Rule-Based Attrition Computation & Prediction
# =============================
TRIGGER_DETAILS = {
    "Low gender diversity": {
        "subproblems": {
            "lack_female_applicants": "Not enough female applicants are applying.",
            "lack_female_mentors": "There are few female mentors or leaders available.",
            "rigid_policies": "The policies are too rigid (e.g., no maternity or remote options)."
        },
        "solutions": {
            "lack_female_applicants": "Partner with women’s universities or female‑oriented professional groups and emphasize diversity in recruitment.",
            "lack_female_mentors": "Implement formal mentorship programs and sponsor leadership development for female employees.",
            "rigid_policies": "Introduce flexible working hours, remote/hybrid work options, and enhance family‑friendly benefits."
        }
    },
    "Stagnant promotions": {
        "subproblems": {
            "unclear_criteria": "Promotion criteria are not transparent.",
            "no_mentorship": "There is a lack of mentorship or upskilling tracks.",
            "bureaucratic_structure": "The organizational structure is overly bureaucratic."
        },
        "solutions": {
            "unclear_criteria": "Publish clear promotion guidelines with KPIs and provide regular feedback.",
            "no_mentorship": "Launch mentoring programs and provide upskilling opportunities.",
            "bureaucratic_structure": "Streamline decision‑making processes or reduce hierarchical layers to foster agility."
        }
    },
    "Very low performance rating": {
        "subproblems": {
            "misaligned_role": "Job roles or expectations are unclear or mismatched.",
            "no_feedback": "There is a lack of continuous feedback or one‑on‑one sessions.",
            "skill_gaps": "Training needs are not being addressed."
        },
        "solutions": {
            "misaligned_role": "Clarify job responsibilities, set SMART goals, and align roles with employees’ strengths.",
            "no_feedback": "Implement frequent one‑on‑one check‑ins and real‑time performance dashboards.",
            "skill_gaps": "Offer targeted training, certification reimbursements, and peer‑to‑peer learning opportunities."
        }
    },
    "Low performance rating": {
        "subproblems": {
            "misaligned_role": "Job roles or expectations are unclear or mismatched.",
            "no_feedback": "Continuous feedback is lacking.",
            "skill_gaps": "Training needs are not addressed."
        },
        "solutions": {
            "misaligned_role": "Clarify job responsibilities and ensure roles align with employees’ strengths.",
            "no_feedback": "Implement regular one‑on‑one check‑ins and provide ongoing coaching.",
            "skill_gaps": "Offer targeted training sessions and promote cross‑functional learning."
        }
    },
    "Low compensation competitiveness": {
        "subproblems": {
            "below_market": "Base salary is below market rates.",
            "minimal_bonus": "Bonuses or variable pay are minimal or nonexistent.",
            "poor_benefits": "The benefits package is insufficient."
        },
        "solutions": {
            "below_market": "Conduct market benchmarking to adjust salaries to at least median levels.",
            "minimal_bonus": "Introduce performance‑based incentives or profit‑sharing schemes.",
            "poor_benefits": "Offer competitive benefits including health insurance and retirement plans."
        }
    }
}

def compute_weighted_attrition(employee, return_triggers=False):
    score = 0
    extreme_factors = 0
    triggers = []
    
    if employee["Gender"] == "Female" and employee["Female Employee Ratio"] <= 15:
        score += 30; extreme_factors += 1; triggers.append("Low gender diversity")
    if employee["Hasn't been promoted"] >= 2 * employee["Minimum Promotion Cycle"]:
        score += 30; extreme_factors += 1; triggers.append("Stagnant promotions")
    if employee["Last Performance Rating"] == 1:
        score += 25; extreme_factors += 1; triggers.append("Very low performance rating")
    elif employee["Last Performance Rating"] == 2:
        score += 15; extreme_factors += 0.5; triggers.append("Low performance rating")
    elif employee["Last Performance Rating"] == 5:
        score -= 15; extreme_factors -= 0.5; triggers.append("Excellent performance rating")
    if employee["Compa Ratio"] < 80:
        score += 20; extreme_factors += 0.8; triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] < 70:
        score += 25; extreme_factors += 1; triggers.append("Low compensation competitiveness")
    elif employee["Compa Ratio"] > 110:
        score -= 15; extreme_factors -= 0.5; triggers.append("High compensation ratio")
    if employee["College Tier Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low college tier retention")
    if employee["Industry Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low industry retention")
    if employee["Company Type Retention"] < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low company type retention")
    if employee["Pulse"] == "High":
        score += 20; extreme_factors += 0.5; triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"] == "Low":
        score -= 20; extreme_factors -= 0.5; triggers.append("Low dissatisfaction (Pulse)")
    
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

def predict_attrition(employee_data, industry):
    model, scaler, feature_columns = load_model(industry)
    if model is None:
        return None, None, None
    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_input)
    ml_probability = model.predict_proba(X_scaled)[:, 1][0] * 100
    rule_probability, triggers = compute_weighted_attrition(employee_data, return_triggers=True)
    combined_score = 0.5 * rule_probability + 0.5 * ml_probability
    return combined_score, triggers, ml_probability

def generate_sample_csv():
    sample_csv = pd.DataFrame({
        "Employee Age": [30, 45],
        "Gender": ["Male", "Female"],
        "Tenure (Months)": [36, 48],
        "Pulse": ["Medium", "High"],
        "Hasn't been promoted": [12, 36],
        "Minimum Promotion Cycle": [24, 24],
        "College Tier": ["Tier 1", "Tier 2"],
        "Industry": ["Tech", "Finance"],
        "Company Type": ["Startup", "Enterprise"],
        "Last Performance Rating": [3, 1],
        "Compa Ratio": [90, 65]
    })
    csv_buffer = io.StringIO()
    sample_csv.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_dummy_training_file():
    dummy_df = pd.DataFrame({
        "Name": ["Example 1", "Example 2", "Example 3"],
        "Employee Age": [30, 40, 35],
        "Gender": ["Male", "Female", "Male"],
        "Tenure (Months)": [36, 48, 24],
        "Pulse": ["Medium", "High", "Low"],
        "Hasn't been promoted": [12, 30, 15],
        "Minimum Promotion Cycle": [24, 24, 24],
        "College Tier": ["Tier 1", "Tier 2", "Tier 3"],
        "Industry": ["Tech", "Finance", "Healthcare"],
        "Company Type": ["Startup", "Enterprise", "SME"],
        "Last Performance Rating": [3, 1, 4],
        "Compa Ratio": [90, 65, 100],
        "Attrition": [0, 1, 0]
    })
    csv_buffer = io.StringIO()
    dummy_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# =============================
# Data Visualization Helpers
# =============================
def compute_trigger_counts(df, column_name):
    triggers_list = []
    for val in df[column_name].dropna():
        if val.strip() != "" and val != "None":
            triggers_list.extend([x.strip() for x in val.split(",") if x.strip()])
    if triggers_list:
        return pd.Series(triggers_list).value_counts()
    else:
        return pd.Series(dtype=int)

def graph_header(title, explanation):
    return f'<h4 style="color: white;">{title} <span title="{explanation}" style="cursor: help; color: #ccc;">&#9432;</span></h4>'

def generate_custom_chart(config, data):
    x_axis = config.get("x_axis")
    y_axis = config.get("y_axis")
    missing_cols = [col for col in [x_axis, y_axis] if col not in data.columns]
    if missing_cols:
        error_msg = f"Column(s) not found: {', '.join(missing_cols)}"
        return alt.Chart(pd.DataFrame({'Error': [error_msg]})).mark_text(
            align='center',
            baseline='middle',
            color='red'
        ).encode(text='Error:N')
    
    if x_axis == "Negative Triggers" or y_axis == "Negative Triggers":
        ct = compute_trigger_counts(data, "Negative Triggers").reset_index()
        ct.columns = ["Trigger", "Count"]
        chart = alt.Chart(ct).mark_bar(color="#e45756").encode(
            x=alt.X("Trigger:N", title="Negative Triggers"),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=["Trigger", "Count"]
        )
    else:
        x_is_numeric = pd.api.types.is_numeric_dtype(data[x_axis])
        y_is_numeric = pd.api.types.is_numeric_dtype(data[y_axis])
        if x_is_numeric and y_is_numeric:
            chart = alt.Chart(data).mark_circle(size=60, color="#4c78a8").encode(
                x=alt.X(f"{x_axis}:Q", title=x_axis),
                y=alt.Y(f"{y_axis}:Q", title=y_axis),
                tooltip=[x_axis, y_axis]
            )
        elif not x_is_numeric and y_is_numeric:
            chart = alt.Chart(data).mark_boxplot(color="#e45756").encode(
                x=alt.X(f"{x_axis}:N", title=x_axis),
                y=alt.Y(f"{y_axis}:Q", title=y_axis),
                tooltip=[x_axis, y_axis]
            )
        elif x_is_numeric and not y_is_numeric:
            chart = alt.Chart(data).mark_boxplot(color="#e45756").encode(
                x=alt.X(f"{y_axis}:N", title=y_axis),
                y=alt.Y(f"{x_axis}:Q", title=x_axis),
                tooltip=[x_axis, y_axis]
            )
        else:
            chart = alt.Chart(data).mark_bar(color="#4c78a8").encode(
                x=alt.X(f"{x_axis}:N", title=x_axis),
                y=alt.Y("count()", title="Count"),
                tooltip=[x_axis]
            )
    return chart

def horizontal_filters(df):
    filter_values = {}
    st.markdown(
        """
        <style>
        div[data-testid="column"] {
             min-width: 200px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    num_custom = len(st.session_state.custom_filters)
    total_filters = 1 + num_custom
    cols = st.columns(total_filters)
    
    with cols[0]:
        score_range = st.slider("Attrition Score", 0, 100, (0, 100), key="filter_attrition_score")
        filter_values["Attrition Score"] = score_range

    for i, filter_id in enumerate(st.session_state.custom_filters):
        with cols[i+1]:
            possible_cols = [col for col in df.columns if col not in ["Name", "Attrition Score", "What-If Attrition Score", "What-If Negative Triggers", "Prediction Time"]]
            col_selected = st.selectbox("Select column", options=possible_cols, key=f"custom_filter_col_{filter_id}")
            filter_values[f"custom_filter_col_{filter_id}"] = col_selected
            if st.button("Remove", key=f"remove_{filter_id}"):
                st.session_state.custom_filters.remove(filter_id)
                safe_rerun()
            if pd.api.types.is_numeric_dtype(df[col_selected]):
                min_val = float(df[col_selected].min())
                max_val = float(df[col_selected].max())
                selected_range = st.slider("Range", min_val, max_val, (min_val, max_val), key=f"custom_filter_range_{filter_id}")
                filter_values[f"custom_filter_range_{filter_id}"] = selected_range
            else:
                unique_vals = list(df[col_selected].dropna().unique())
                selected_vals = st.multiselect("Values", options=unique_vals, default=unique_vals, key=f"custom_filter_vals_{filter_id}")
                filter_values[f"custom_filter_vals_{filter_id}"] = selected_vals

    if st.button("Add Custom Filter", key="add_custom_filter"):
        new_id = str(datetime.now().timestamp())
        st.session_state.custom_filters.append(new_id)
        safe_rerun()
        
    return filter_values

def apply_filters(df, filter_values):
    filtered_df = df.copy()
    if "Attrition Score" in filter_values:
        low, high = filter_values["Attrition Score"]
        filtered_df = filtered_df[(filtered_df["Attrition Score"] >= low) & (filtered_df["Attrition Score"] <= high)]
    for key in filter_values:
        if key.startswith("custom_filter_col_"):
            filter_id = key.replace("custom_filter_col_", "")
            col = filter_values[key]
            range_key = f"custom_filter_range_{filter_id}"
            if range_key in filter_values:
                low_val, high_val = filter_values[range_key]
                filtered_df = filtered_df[(filtered_df[col] >= low_val) & (filtered_df[col] <= high_val)]
            else:
                vals_key = f"custom_filter_vals_{filter_id}"
                if vals_key in filter_values:
                    selected_vals = filter_values[vals_key]
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    return filtered_df

# Initialize custom filters if not already present
if "custom_filters" not in st.session_state:
    st.session_state.custom_filters = []

# =============================
# Login/Sign Up System
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Employee Attrition Prediction Tool - Login / Sign Up")
    auth_mode = st.radio("Select Mode", ["Login", "Sign Up"], index=0)
    if auth_mode == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                users = load_users()
                if email in users and users[email]["password"] == password:
                    st.success(f"Welcome back, {users[email]['name']}!")
                    st.session_state.user = users[email]
                    st.session_state.logged_in = True
                    safe_rerun()
                else:
                    st.error("Invalid email or password.")
    else:
        with st.form("signup_form"):
            name = st.text_input("Name")
            designation = st.text_input("Designation")
            company = st.text_input("Company Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    users = load_users()
                    if email in users:
                        st.error("Email already exists. Please log in.")
                    else:
                        user = {
                            "name": name,
                            "designation": designation,
                            "company": company,
                            "email": email,
                            "password": password,
                            "settings": {}
                        }
                        users[email] = user
                        save_users(users)
                        st.success("Account created and logged in!")
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        safe_rerun()
    if not st.session_state.logged_in:
        st.stop()

# =============================
# Top Header with Title, My Account, and Logout
# =============================
header_container = st.container()
with header_container:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>Employee Attrition Prediction Tool</h1>", unsafe_allow_html=True)
    with col2:
        if st.button("👤 My Account", key="account_button"):
            st.session_state.nav = "My Account"
        if st.button("Logout", key="logout_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            safe_rerun()

# =============================
# Sidebar Global Settings & Navigation
# =============================
# Global Settings for Bulk Analysis
if "global_avg_age" not in st.session_state:
    st.session_state.global_avg_age = st.session_state.user.get("settings", {}).get("global_avg_age", 35)
if "global_female_ratio" not in st.session_state:
    st.session_state.global_female_ratio = st.session_state.user.get("settings", {}).get("global_female_ratio", 40)
if "bulk_tier1" not in st.session_state:
    st.session_state.bulk_tier1 = st.session_state.user.get("settings", {}).get("bulk_tier1", 60)
if "bulk_tier2" not in st.session_state:
    st.session_state.bulk_tier2 = st.session_state.user.get("settings", {}).get("bulk_tier2", 50)
if "bulk_tier3" not in st.session_state:
    st.session_state.bulk_tier3 = st.session_state.user.get("settings", {}).get("bulk_tier3", 40)
if "bulk_startup" not in st.session_state:
    st.session_state.bulk_startup = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60)
if "bulk_small" not in st.session_state:
    st.session_state.bulk_small = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Small Size", 55)
if "bulk_mid" not in st.session_state:
    st.session_state.bulk_mid = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Mid Size", 50)
if "bulk_mnc" not in st.session_state:
    st.session_state.bulk_mnc = st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("MNC/Giant Company", 45)

st.sidebar.markdown("### Global Settings for Bulk Analysis")
st.session_state.global_avg_age = st.sidebar.slider(
    "Average Employee Age in Company", 18, 100,
    st.session_state.global_avg_age,
    key="global_avg_age"
)
st.session_state.global_female_ratio = st.sidebar.slider(
    "Women % in Organization", 0, 100,
    st.session_state.global_female_ratio,
    key="global_female_ratio"
)
with st.sidebar.expander("College Tier Retention Settings"):
    st.session_state.bulk_tier1 = st.slider(
        "Tier 1 Retention (%)", 10, 100,
        st.session_state.bulk_tier1,
        key="bulk_tier1"
    )
    st.session_state.bulk_tier2 = st.slider(
        "Tier 2 Retention (%)", 10, 100,
        st.session_state.bulk_tier2,
        key="bulk_tier2"
    )
    st.session_state.bulk_tier3 = st.slider(
        "Tier 3 Retention (%)", 10, 100,
        st.session_state.bulk_tier3,
        key="bulk_tier3"
    )
with st.sidebar.expander("Industry Retention Settings"):
    bulk_industry_retention = {}
    for ind in industry_options:
        default_val = st.session_state.user.get("settings", {}).get("bulk_industry_retention", {}).get(ind, 60 if ind=="Tech" else 50)
        bulk_industry_retention[ind] = st.slider(
            f"{ind} Retention (%)", 10, 100, default_val,
            key=f"bulk_ind_{ind}"
        )
with st.sidebar.expander("Company Type Retention Settings"):
    st.session_state.bulk_startup = st.slider(
        "Startup Retention (%)", 10, 100,
        st.session_state.bulk_startup,
        key="bulk_startup"
    )
    st.session_state.bulk_small = st.slider(
        "Small Size Retention (%)", 10, 100,
        st.session_state.bulk_small,
        key="bulk_small"
    )
    st.session_state.bulk_mid = st.slider(
        "Mid Size Retention (%)", 10, 100,
        st.session_state.bulk_mid,
        key="bulk_mid"
    )
    st.session_state.bulk_mnc = st.slider(
        "MNC/Giant Company Retention (%)", 10, 100,
        st.session_state.bulk_mnc,
        key="bulk_mnc"
    )

# =============================
# Sidebar Navigation
# =============================
nav_options = ["Dashboard", "Train Mode", "Test Mode", "Cluster Analysis", "Additional Features", "My Account"]
selected_nav = st.sidebar.radio("Navigation", nav_options, index=0)
st.session_state.nav = selected_nav

# =============================
# Navigation Pages
# =============================
if st.session_state.nav == "Dashboard":
    st.header("Dashboard")
    st.write("Welcome to the Employee Attrition Prediction Dashboard!")
    # Add aggregated metrics or summaries here.

elif st.session_state.nav == "Train Mode":
    st.header("Train Mode")
    selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
    with col2:
        st.markdown("### Training File Guide")
        st.markdown("""
        Your training file must include:
        - A target column (e.g., Attrition; use 0 for active, 1 for non‑active).
        - Feature columns: Employee Age, Gender, Tenure (Months), Pulse, Hasn't been promoted, Minimum Promotion Cycle, College Tier, Industry, Company Type, Last Performance Rating, Compa Ratio.
        """)
        st.download_button(
            label="Download Dummy Training File",
            data=generate_dummy_training_file(),
            file_name="dummy_training_file.csv",
            mime="text/csv"
        )
    target_column = st.text_input("Enter the target column name", value="Attrition")
    if uploaded_train is not None:
        try:
            train_df = pd.read_csv(uploaded_train) if uploaded_train.name.endswith(".csv") else pd.read_excel(uploaded_train)
            st.write("### Training Data Preview:")
            st.dataframe(train_df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
        if st.button("Train Model"):
            train_model(train_df, target_column, selected_train_industry)
            st.session_state.training_data = train_df.copy()

elif st.session_state.nav == "Test Mode":
    st.header("Test Mode")
    selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=0, key="test_industry")
    st.markdown("""
    <div class="tooltip">Read Instructions
      <span class="tooltiptext">
        Ensure you have trained a model in Train Mode.
        <br><br>
        Upload a CSV/Excel file with columns: Name, Employee Age, Gender, Tenure (Months), Pulse, Hasn't been promoted, Minimum Promotion Cycle, College Tier, Industry, Company Type, Last Performance Rating, Compa Ratio.
        <br><br>
        (No Attrition column needed for testing.)
      </span>
    </div>
    <style>
    .tooltip { position: relative; display: inline-block; cursor: pointer; font-weight: bold; color: #0073e6; }
    .tooltip .tooltiptext { visibility: hidden; width: 300px; background-color: #333; color: #ddd; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; top: 125%; left: 50%; margin-left: -150px; box-shadow: 0px 0px 6px 0px rgba(0,0,0,0.2); }
    .tooltip:hover .tooltiptext { visibility: visible; }
    </style>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Bulk Data (CSV or Excel)", type=["csv", "xlsx"], key="bulk_file")
    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
        st.write("### Bulk Data Preview:")
        st.dataframe(df_bulk.head())
        required_cols = [
            "Name", "Employee Age", "Gender", "Tenure (Months)", "Pulse",
            "Hasn't been promoted", "Minimum Promotion Cycle", "College Tier",
            "Industry", "Company Type", "Last Performance Rating", "Compa Ratio"
        ]
        missing = [c for c in required_cols if c not in df_bulk.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Run Bulk Prediction"):
                scores, triggers_list, names = [], [], []
                for idx, row in df_bulk.iterrows():
                    row_dict = row.to_dict()
                    names.append(row_dict.get("Name"))
                    row_dict["Average Employee Age"] = st.session_state.global_avg_age
                    row_dict["Female Employee Ratio"] = st.session_state.global_female_ratio
                    college_tier = row_dict.get("College Tier")
                    if college_tier == "Tier 1":
                        row_dict["College Tier Retention"] = st.session_state.bulk_tier1
                    elif college_tier == "Tier 2":
                        row_dict["College Tier Retention"] = st.session_state.bulk_tier2
                    elif college_tier == "Tier 3":
                        row_dict["College Tier Retention"] = st.session_state.bulk_tier3
                    else:
                        st.warning(f"Row {idx}: Unknown College Tier '{college_tier}'. Using default 40%.")
                        row_dict["College Tier Retention"] = 40
                    ind_val = row_dict.get("Industry")
                    row_dict["Industry Retention"] = bulk_industry_retention.get(ind_val, 50)
                    ctype_val = row_dict.get("Company Type", "Startup")
                    if ctype_val.lower() == "startup":
                        row_dict["Company Type Retention"] = st.session_state.bulk_startup
                    elif "small" in ctype_val.lower():
                        row_dict["Company Type Retention"] = st.session_state.bulk_small
                    elif "mid" in ctype_val.lower():
                        row_dict["Company Type Retention"] = st.session_state.bulk_mid
                    elif "mnc" in ctype_val.lower() or "giant" in ctype_val.lower():
                        row_dict["Company Type Retention"] = st.session_state.bulk_mnc
                    else:
                        row_dict["Company Type Retention"] = 50
                    try:
                        bulk_score, bulk_trigs, _ = predict_attrition(row_dict, selected_test_industry)
                    except Exception as e:
                        st.error(f"Row {idx}: Prediction failed due to {e}. Skipping this row.")
                        scores.append(None)
                        triggers_list.append("Prediction Failed")
                        continue
                    scores.append(bulk_score)
                    neg_trigs = [t for t in bulk_trigs if t in TRIGGER_DETAILS]
                    triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
                    triggers_list.append(triggers_str)
                df_bulk["Attrition Score"] = scores
                df_bulk["Negative Triggers"] = triggers_list
                df_bulk["Name"] = names
                st.session_state.bulk_result = df_bulk.copy()
                st.session_state.bulk_prediction_complete = True
                st.session_state.bulk_result["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_user_event(st.session_state.user["email"], "bulk_prediction", {"rows": len(df_bulk)})

elif st.session_state.nav == "Cluster Analysis":
    st.header("Cluster Analysis")
    st.subheader("Automatic Clusters with Visual Enhancements")
    if "bulk_result" not in st.session_state or st.session_state.bulk_result is None:
        st.info("Please run bulk prediction first (Test Mode) to see clusters.")
    else:
        clusters_df = generate_automatic_clusters(st.session_state.bulk_result)
        if clusters_df.empty:
            st.info("No significant clusters found. Try adjusting your settings or bulk data.")
        else:
            cols = st.columns(3)
            for i, (_, cluster) in enumerate(clusters_df.iterrows()):
                col = cols[i % 3]
                with col:
                    st.markdown(f"""
                    <div class="cluster-card">
                      <h4>{cluster['Gender']} | Age {cluster['Age Group']} | {cluster['Primary Trigger']}</h4>
                      <p><strong>Avg Attrition Score:</strong> {cluster['Attrition Score']:.2f}</p>
                      <p><strong>Employee Count:</strong> {cluster['Count']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Mini chart within the cluster card
                    cluster_data = st.session_state.bulk_result[
                        (st.session_state.bulk_result["Gender"] == cluster["Gender"]) &
                        (st.session_state.bulk_result["Employee Age"] >= int(str(cluster["Age Group"]).split("-")[0])) &
                        (st.session_state.bulk_result["Employee Age"] <= (int(str(cluster["Age Group"]).split("-")[-1]) if "-" in str(cluster["Age Group"]) else 100)) &
                        (st.session_state.bulk_result["Negative Triggers"].str.contains(cluster["Primary Trigger"], na=False))
                    ]
                    if not cluster_data.empty:
                        mini_chart = alt.Chart(cluster_data).mark_bar().encode(
                            x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=10), title="Score"),
                            y=alt.Y("count()", title="Count")
                        ).properties(width=150, height=100)
                        st.altair_chart(mini_chart, use_container_width=True)
    st.subheader("Custom Cluster Analysis")
    with st.form("custom_cluster_form"):
        custom_age = st.slider("Employee Age", 18, 100, (30, 50))
        custom_gender = st.multiselect("Gender", options=st.session_state.bulk_result["Gender"].unique().tolist(),
                                       default=st.session_state.bulk_result["Gender"].unique().tolist())
        custom_industry = st.multiselect("Industry", options=st.session_state.bulk_result["Industry"].unique().tolist(),
                                         default=st.session_state.bulk_result["Industry"].unique().tolist())
        custom_college = st.multiselect("College Tier", options=st.session_state.bulk_result["College Tier"].unique().tolist(),
                                        default=st.session_state.bulk_result["College Tier"].unique().tolist())
        custom_company = st.multiselect("Company Type", options=st.session_state.bulk_result["Company Type"].unique().tolist(),
                                        default=st.session_state.bulk_result["Company Type"].unique().tolist())
        submit_custom = st.form_submit_button("Generate Custom Cluster")
    if submit_custom:
        df_custom = st.session_state.bulk_result.copy()
        df_custom = df_custom[(df_custom["Employee Age"] >= custom_age[0]) & (df_custom["Employee Age"] <= custom_age[1])]
        df_custom = df_custom[df_custom["Gender"].isin(custom_gender)]
        df_custom = df_custom[df_custom["Industry"].isin(custom_industry)]
        df_custom = df_custom[df_custom["College Tier"].isin(custom_college)]
        df_custom = df_custom[df_custom["Company Type"].isin(custom_company)]
        if df_custom.empty:
            st.warning("No data matching the custom cluster criteria.")
        else:
            avg_score = df_custom["Attrition Score"].mean()
            count = df_custom.shape[0]
            triggers_series = compute_trigger_counts(df_custom, "Negative Triggers")
            common_trigger = triggers_series.idxmax() if not triggers_series.empty else "None"
            st.markdown(f"""
            <div class="custom-cluster-card">
              <h4>Custom Cluster Summary</h4>
              <p><strong>Employee Count:</strong> {count}</p>
              <p><strong>Average Attrition Score:</strong> {avg_score:.2f}</p>
              <p><strong>Common Negative Trigger:</strong> {common_trigger}</p>
            </div>
            """, unsafe_allow_html=True)
            chart_custom = alt.Chart(df_custom).mark_circle(size=60).encode(
                x=alt.X("Employee Age:Q", title="Employee Age"),
                y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                tooltip=["Name", "Attrition Score"]
            )
            st.altair_chart(chart_custom, use_container_width=True)

elif st.session_state.nav == "Additional Features":
    st.header("Additional Features")
    st.subheader("User-Driven Clustering Algorithms")
    num_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)
    features = st.multiselect("Select features for clustering", 
                              options=["Employee Age", "Tenure (Months)", "Last Performance Rating", "Compa Ratio"],
                              default=["Employee Age", "Compa Ratio"])
    if st.button("Run K-Means Clustering"):
        from sklearn.cluster import KMeans
        df_cluster = st.session_state.bulk_result.copy() if "bulk_result" in st.session_state else pd.DataFrame()
        if df_cluster.empty:
            st.error("No bulk data available for clustering.")
        else:
            df_features = df_cluster[features].dropna()
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df_features["Cluster"] = kmeans.fit_predict(df_features)
            cluster_summary = df_features.groupby("Cluster").agg({ 
                "Employee Age": "mean", 
                "Tenure (Months)": "mean", 
                "Last Performance Rating": "mean", 
                "Compa Ratio": "mean"
            }).reset_index()
            st.dataframe(cluster_summary)
            if len(features) >= 2:
                scatter_k = alt.Chart(df_features.reset_index()).mark_circle(size=60).encode(
                    x=alt.X(f"{features[0]}:Q", title=features[0]),
                    y=alt.Y(f"{features[1]}:Q", title=features[1]),
                    color="Cluster:N",
                    tooltip=features + ["Cluster"]
                )
                st.altair_chart(scatter_k, use_container_width=True)
    st.subheader("Advanced Data Visualization")
    st.info("This section is a placeholder for integrating additional interactive visualizations or real-time data updates.")
    if st.button("Simulate Data Refresh"):
        st.success("Data refreshed! (This is a simulation of real-time integration.)")

elif st.session_state.nav == "My Account":
    st.markdown("<div style='text-align: center;'><h2>My Account</h2></div>", unsafe_allow_html=True)
    user = st.session_state.user
    st.write("### Account Information")
    st.write(f"**Name:** {user.get('name', '')}")
    st.write(f"**Designation:** {user.get('designation', '')}")
    st.write(f"**Company:** {user.get('company', '')}")
    st.write(f"**Email:** {user.get('email', '')}")
    st.write("### Saved Global Settings")
    user_settings = user.get("settings") or {}
    if user_settings:
        st.json(user_settings)
    else:
        st.info("No global settings saved. Please train your model to save settings.")
    st.write("### Analysis History")
    history = load_user_history(user["email"])
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.info("No history available yet.")
    if st.button("Back to Main"):
        st.session_state.nav = "Dashboard"
