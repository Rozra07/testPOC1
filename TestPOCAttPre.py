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

# -----------------------------
# Define and enable a dark Altair theme
# -----------------------------
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

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Custom CSS for an engaging design
# -----------------------------
st.markdown("""
    <style>
      .cluster-card, .custom-cluster-card {
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 8px;
          background-color: #222;
          color: white;
          margin: 5px;
      }
      .cluster-card {
          border: 1px solid #4c78a8;
      }
      .header-title {
          text-align: center;
          color: white;
      }
      .nav-button {
          margin-right: 10px;
          padding: 6px 12px;
          border: 1px solid #4c78a8;
          border-radius: 4px;
          background-color: #333;
          color: white;
          cursor: pointer;
      }
      .nav-button.active {
          background-color: #4c78a8;
      }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Helper function for safe rerun
# -----------------------------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Refresh functionality is not available. Please update Streamlit (>=0.65.0).")

# -----------------------------
# Global helper: colored_metric for trustworthiness table
# -----------------------------
def colored_metric(value, threshold, higher_better=True, is_lower=False):
    if is_lower:
        color = "green" if value <= threshold else "red"
    else:
        color = "green" if value >= threshold else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.2f}</span>'

# -----------------------------
# Initialize st.session_state keys if not already set
# -----------------------------
for key in ["logged_in", "nav", "user", "bulk_prediction_complete", "bulk_result", 
            "enable_what_if", "custom_charts", "custom_filters", "training_data"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["custom_charts", "custom_filters"] else (False if key=="logged_in" else ("Home" if key=="nav" else None))
if "enable_cluster_analysis" not in st.session_state:
    st.session_state.enable_cluster_analysis = False

# -----------------------------
# User storage helper functions (load/save users, history)
# -----------------------------
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

# -----------------------------
# Global: Expanded Industry Options
# -----------------------------
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]

# -----------------------------
# Function to generate automatic clusters based on bulk prediction results
# (Now with an embedded mini chart)
# -----------------------------
def generate_automatic_clusters(df):
    if df.empty:
        return pd.DataFrame()
    df_auto = df.copy()
    # Create Age Group using bins. Adjust bins/labels as needed.
    bins = [18, 30, 40, 50, 60, 100]
    labels = ["18-30", "31-40", "41-50", "51-60", "61+"]
    df_auto["Age Group"] = pd.cut(df_auto["Employee Age"], bins=bins, labels=labels, right=False)
    # Extract primary negative trigger; if none exists, return NaN.
    def get_primary(trigger_str):
        if pd.isna(trigger_str) or trigger_str.strip() == "" or trigger_str.strip().lower() == "none":
            return np.nan
        else:
            return trigger_str.split(",")[0].strip()
    df_auto["Primary Trigger"] = df_auto["Negative Triggers"].apply(get_primary)
    df_auto = df_auto.dropna(subset=["Primary Trigger"])
    # Group by Gender, Age Group, and Primary Trigger to compute average attrition score and employee count
    clusters = df_auto.groupby(["Gender", "Age Group", "Primary Trigger"]).agg({"Attrition Score": "mean", "Name": "count"}).reset_index()
    clusters.rename(columns={"Name": "Count"}, inplace=True)
    # Filter for clusters with a minimum count and high attrition score
    clusters = clusters[(clusters["Count"] >= 3) & (clusters["Attrition Score"] >= 60)]
    clusters = clusters.sort_values(by="Attrition Score", ascending=False)
    # Limit to maximum 9 clusters
    clusters = clusters.head(9)
    return clusters

# -----------------------------
# Functions for model training/prediction and industry record updates
# -----------------------------
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
    
    # Model Evaluation Metrics
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, log_loss, average_precision_score
    preds = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, preds)
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
    
    # Compute Trustworthiness Metrics
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
    
    # Save global settings to user record (example settings update)
    user = st.session_state.user
    user_settings = user.get("settings") or {}
    user_settings["global_avg_age"] = st.session_state.get("global_avg_age", 35)
    user_settings["global_female_ratio"] = st.session_state.get("global_female_ratio", 40)
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

# -----------------------------
# Rule-based attrition computation and prediction functions
# -----------------------------
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
    if employee.get("College Tier Retention", 0) < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low college tier retention")
    if employee.get("Industry Retention", 0) < 15:
        score += 15; extreme_factors += 0.5; triggers.append("Low industry retention")
    if employee.get("Company Type Retention", 0) < 15:
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

# -----------------------------
# Navigation: Top header with separate pages
# -----------------------------
def render_navigation():
    nav_options = ["Home", "Bulk Analysis", "Cluster Analysis", "Additional Features", "My Account"]
    nav_buttons = st.columns(len(nav_options))
    for i, option in enumerate(nav_options):
        if nav_buttons[i].button(option, key=f"nav_{option}"):
            st.session_state.nav = option
            safe_rerun()
    st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Login/Sign Up System
# -----------------------------
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

# -----------------------------
# Header: Navigation and Logout / My Account
# -----------------------------
with st.container():
    render_navigation()
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

# -----------------------------
# Page Rendering based on Navigation
# -----------------------------
if st.session_state.nav == "My Account":
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

elif st.session_state.nav == "Bulk Analysis":
    st.header("Bulk Employee Attrition Prediction")
    st.info("Bulk Analysis page. [Insert your bulk analysis logic here]")
    # Insert your bulk prediction, filtering, and charting logic here.

elif st.session_state.nav == "Cluster Analysis":
    st.header("Cluster Analysis")
    st.subheader("Automatic Clusters")
    if "bulk_result" not in st.session_state or st.session_state.bulk_result is None:
        st.info("No bulk prediction data available. Please run bulk prediction in the Bulk Analysis page.")
    else:
        clusters_df = generate_automatic_clusters(st.session_state.bulk_result)
        if clusters_df.empty:
            st.info("No significant clusters found. Try adjusting your global settings or bulk data.")
        else:
            cols = st.columns(3)
            for i, (_, cluster) in enumerate(clusters_df.iterrows()):
                col = cols[i % 3]
                with col:
                    # Create a mini chart for this cluster (a simple bar chart showing count)
                    mini_chart = alt.Chart(pd.DataFrame({
                        "Metric": ["Count"],
                        "Value": [cluster["Count"]]
                    })).mark_bar().encode(
                        x=alt.X("Metric:N", title=""),
                        y=alt.Y("Value:Q", title="")
                    ).properties(width=120, height=80)
                    st.markdown(f"""
                    <div class="cluster-card">
                      <h4>{cluster['Gender']} | Age {cluster['Age Group']} | {cluster['Primary Trigger']}</h4>
                      <p><strong>Avg Attrition Score:</strong> {cluster['Attrition Score']:.2f}</p>
                      <p><strong>Employee Count:</strong> {cluster['Count']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.altair_chart(mini_chart, use_container_width=True)
    st.subheader("Custom Cluster Analysis")
    with st.form("custom_cluster_form"):
        st.markdown("Customize your cluster filters:")
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
            # Create a custom chart (histogram of attrition scores)
            custom_chart = alt.Chart(df_custom).mark_bar(color="#4c78a8").encode(
                x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=10), title="Attrition Score"),
                y=alt.Y("count()", title="Frequency")
            ).properties(width=250, height=150)
            st.markdown(f"""
            <div class="custom-cluster-card">
              <h4>Custom Cluster Summary</h4>
              <p><strong>Employee Count:</strong> {count}</p>
              <p><strong>Average Attrition Score:</strong> {avg_score:.2f}</p>
              <p><strong>Common Negative Trigger:</strong> {common_trigger}</p>
            </div>
            """, unsafe_allow_html=True)
            st.altair_chart(custom_chart, use_container_width=False)

elif st.session_state.nav == "Additional Features":
    st.header("Additional Features")
    st.markdown("Choose from the dropdown below to explore advanced features:")
    advanced_feature = st.selectbox("Select Advanced Feature", ["User-Driven Clustering", "Advanced Data Visualization", "Real-Time Data Integration"])
    if advanced_feature == "User-Driven Clustering":
        st.subheader("User-Driven Clustering")
        num_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)
        features = st.multiselect("Select features for clustering", 
                                  options=["Employee Age", "Tenure (Months)", "Last Performance Rating", "Compa Ratio"],
                                  default=["Employee Age", "Compa Ratio"])
        if st.button("Run K-Means Clustering"):
            from sklearn.cluster import KMeans
            df_cluster = st.session_state.bulk_result.copy()
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
    elif advanced_feature == "Advanced Data Visualization":
        st.subheader("Advanced Data Visualization")
        st.info("This section is a placeholder for advanced interactive visualizations (e.g., using Plotly).")
    elif advanced_feature == "Real-Time Data Integration":
        st.subheader("Real-Time Data Integration")
        st.markdown("Simulate real-time data integration by refreshing the data dashboard.")
        if st.button("Refresh Data Integration"):
            st.success("Data refreshed! (Simulation only.)")
else:
    st.header("Welcome to the Employee Attrition Prediction Tool")
    st.info("Please select a page from the navigation above.")
