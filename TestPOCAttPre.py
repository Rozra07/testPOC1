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
from sklearn.cluster import KMeans
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

# ---------------------------------------
# Helper function for safe rerun
# ---------------------------------------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Refresh functionality is not available. Please update Streamlit.")

# ---------------------------------------
# Global helper: colored_metric for trustworthiness table
# ---------------------------------------
def colored_metric(value, threshold, higher_better=True, is_lower=False):
    if is_lower:
        color = "green" if value <= threshold else "red"
    else:
        color = "green" if value >= threshold else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.2f}</span>'

# ----------------------------------------------------
# Initialize st.session_state keys if not already set
# ----------------------------------------------------
for key in ["logged_in", "nav", "user", "bulk_prediction_complete", "bulk_result", "enable_what_if", "custom_charts", "custom_filters", "training_data"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["custom_charts", "custom_filters"] else (False if key=="logged_in" else ("Tabs" if key=="nav" else None))

# Initialize separate custom filters for bulk and cohort analysis
if "bulk_custom_filters" not in st.session_state:
    st.session_state.bulk_custom_filters = []
if "cohort_custom_filters" not in st.session_state:
    st.session_state.cohort_custom_filters = []

# ----------------------------------------------------
# Helper functions for user storage
# ----------------------------------------------------
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

# ----------------------------------------------------
# Global: Expanded Industry Options
# ----------------------------------------------------
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]

# ----------------------------------------------------
# Dummy Training File Generator (Missing function now provided)
# ----------------------------------------------------
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

# ----------------------------------------------------
# Functions for model training/prediction
# ----------------------------------------------------
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

# ----------------------------------------------------
# Automatic Cohort Analysis – Boxed Layout
# ----------------------------------------------------
def run_auto_cohort_analysis_boxed(df):
    # Required columns for automatic clustering
    required_cols = ["Attrition Score", "Employee Age", "Tenure (Months)", "Compa Ratio", "Negative Triggers"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column '{col}' for automatic cohort analysis.")
            return
    df = df.copy()
    # Compute Negative Trigger Count from the Negative Triggers column
    df["NegTriggerCount"] = df["Negative Triggers"].apply(
        lambda x: 0 if pd.isna(x) or x=="None" or x.strip()=="" 
        else len([item for item in x.split(",") if item.strip() != ""])
    )
    features = ["Attrition Score", "Employee Age", "Tenure (Months)", "Compa Ratio", "NegTriggerCount"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Cluster into 9 groups
    kmeans = KMeans(n_clusters=9, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # Build cluster summary information
    cluster_summaries = []
    for cluster in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == cluster]
        count = cluster_df.shape[0]
        avg_attrition = cluster_df["Attrition Score"].mean()
        avg_age = cluster_df["Employee Age"].mean()
        avg_tenure = cluster_df["Tenure (Months)"].mean()
        avg_compa = cluster_df["Compa Ratio"].mean()
        avg_neg = cluster_df["NegTriggerCount"].mean()
        # Aggregate all negative triggers
        all_triggers = cluster_df["Negative Triggers"].dropna().apply(lambda x: [item.strip() for item in x.split(",") if item.strip()]).sum()
        top_trigger = pd.Series(all_triggers).value_counts().idxmax() if len(all_triggers) > 0 else "None"
        cluster_summaries.append({
            "Cluster": cluster,
            "Count": count,
            "Avg Attrition": avg_attrition,
            "Avg Age": avg_age,
            "Avg Tenure": avg_tenure,
            "Avg Compa": avg_compa,
            "Avg Neg Triggers": avg_neg,
            "Top Issue": top_trigger
        })
    summary_df = pd.DataFrame(cluster_summaries)
    
    st.subheader("Automatic Cohort Analysis")
    st.markdown("The following 9 clusters are automatically computed from key metrics. Each box shows the key insights for that cohort.")
    
    # Display in a 3-column grid (3 boxes per row)
    num_clusters = summary_df.shape[0]
    num_cols = 3
    for i in range(0, num_clusters, num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < num_clusters:
                row = summary_df.iloc[idx]
                box_html = f"""
                <div style="border:1px solid white; border-radius:5px; padding:10px; margin:5px; background-color:#333;">
                    <h3 style="color:white;">Cluster {int(row['Cluster'])}</h3>
                    <p style="color:white;"><b>Count:</b> {row['Count']}</p>
                    <p style="color:white;"><b>Avg Attrition Score:</b> {row['Avg Attrition']:.2f}</p>
                    <p style="color:white;"><b>Avg Age:</b> {row['Avg Age']:.2f}</p>
                    <p style="color:white;"><b>Avg Tenure:</b> {row['Avg Tenure']:.2f}</p>
                    <p style="color:white;"><b>Avg Compa Ratio:</b> {row['Avg Compa']:.2f}</p>
                    <p style="color:white;"><b>Avg Neg Trigger Count:</b> {row['Avg Neg Triggers']:.2f}</p>
                    <p style="color:white;"><b>Top Issue:</b> {row['Top Issue']}</p>
                </div>
                """
                col.markdown(box_html, unsafe_allow_html=True)

# ----------------------------------------------------
# Function to compute trigger counts from a column with comma-separated triggers (repeated)
# ----------------------------------------------------
def compute_trigger_counts(df, column_name):
    triggers_list = []
    for val in df[column_name].dropna():
        if val.strip() != "" and val != "None":
            triggers_list.extend([x.strip() for x in val.split(",") if x.strip()])
    if triggers_list:
        return pd.Series(triggers_list).value_counts()
    else:
        return pd.Series(dtype=int)

# ---------------------------------------
# Helper: generate a custom chart from a saved configuration and current filtered data
# ---------------------------------------
def generate_custom_chart(config, data):
    x_axis = config.get("x_axis")
    y_axis = config.get("y_axis")
    missing_cols = [col for col in [x_axis, y_axis] if col not in data.columns]
    if missing_cols:
        error_msg = f"Column(s) not found: {', '.join(missing_cols)}"
        return alt.Chart(pd.DataFrame({'Error': [error_msg]})).mark_text(align='center', baseline='middle', color='red').encode(text='Error:N')
    if x_axis == "Negative Triggers" or y_axis == "Negative Triggers":
        ct = compute_trigger_counts(data, "Negative Triggers").reset_index()
        ct.columns = ["Trigger", "Count"]
        chart = alt.Chart(ct).mark_bar(color="#e45756").encode(x=alt.X("Trigger:N", title="Negative Triggers"), y=alt.Y("Count:Q", title="Count"), tooltip=["Trigger", "Count"])
    else:
        x_is_numeric = pd.api.types.is_numeric_dtype(data[x_axis])
        y_is_numeric = pd.api.types.is_numeric_dtype(data[y_axis])
        if x_is_numeric and y_is_numeric:
            chart = alt.Chart(data).mark_circle(size=60, color="#4c78a8").encode(x=alt.X(f"{x_axis}:Q", title=x_axis), y=alt.Y(f"{y_axis}:Q", title=y_axis), tooltip=[x_axis, y_axis])
        elif not x_is_numeric and y_is_numeric:
            chart = alt.Chart(data).mark_boxplot(color="#e45756").encode(x=alt.X(f"{x_axis}:N", title=x_axis), y=alt.Y(f"{y_axis}:Q", title=y_axis), tooltip=[x_axis, y_axis])
        elif x_is_numeric and not y_is_numeric:
            chart = alt.Chart(data).mark_boxplot(color="#e45756").encode(x=alt.X(f"{y_axis}:N", title=y_axis), y=alt.Y(f"{x_axis}:Q", title=x_axis), tooltip=[x_axis, y_axis])
        else:
            chart = alt.Chart(data).mark_bar(color="#4c78a8").encode(x=alt.X(f"{x_axis}:N", title=x_axis), y=alt.Y("count()", title="Count"), tooltip=[x_axis])
    return chart

# ---------------------------------------
# Horizontal Filter Functions for Bulk/Cohort Analysis with Unique Keys
# ---------------------------------------
def horizontal_filters(df, prefix="", custom_filters_state=None):
    if custom_filters_state is None:
        custom_filters_state = st.session_state.custom_filters
    filter_values = {}
    st.markdown("""
        <style>
        div[data-testid="column"] { min-width: 200px; }
        </style>
        """, unsafe_allow_html=True)
    num_custom = len(custom_filters_state)
    if "Attrition Score" in df.columns:
        total_filters = 1 + num_custom
    else:
        total_filters = num_custom if num_custom > 0 else 1
    cols = st.columns(total_filters)
    col_index = 0
    if "Attrition Score" in df.columns:
        with cols[0]:
            score_range = st.slider("Attrition Score", 0, 100, (0, 100), key=f"{prefix}filter_attrition_score")
            filter_values["Attrition Score"] = score_range
        col_index = 1
    for i, filter_id in enumerate(custom_filters_state):
        with cols[i + col_index]:
            possible_cols = [col for col in df.columns if col not in ["Name", "Attrition Score", "What-If Attrition Score", "What-If Negative Triggers", "Prediction Time"]]
            col_selected = st.selectbox("Select column", options=possible_cols, key=f"{prefix}custom_filter_col_{filter_id}")
            filter_values[f"custom_filter_col_{filter_id}"] = col_selected
            if st.button("Remove", key=f"{prefix}remove_{filter_id}"):
                custom_filters_state.remove(filter_id)
                safe_rerun()
            if pd.api.types.is_numeric_dtype(df[col_selected]):
                min_val = float(df[col_selected].min())
                max_val = float(df[col_selected].max())
                selected_range = st.slider("Range", min_val, max_val, (min_val, max_val), key=f"{prefix}custom_filter_range_{filter_id}")
                filter_values[f"custom_filter_range_{filter_id}"] = selected_range
            else:
                unique_vals = list(df[col_selected].dropna().unique())
                selected_vals = st.multiselect("Values", options=unique_vals, default=unique_vals, key=f"{prefix}custom_filter_vals_{filter_id}")
                filter_values[f"custom_filter_vals_{filter_id}"] = selected_vals
    if st.button("Add Custom Filter", key=f"{prefix}add_custom_filter"):
        new_id = str(datetime.now().timestamp())
        custom_filters_state.append(new_id)
        safe_rerun()
    return filter_values

def apply_filters(df, filter_values):
    filtered_df = df.copy()
    if "Attrition Score" in filter_values and "Attrition Score" in df.columns:
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

# ---------------------------------------
# Login/Sign Up System
# ---------------------------------------
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
                        user = {"name": name, "designation": designation, "company": company, "email": email, "password": password, "settings": {}}
                        users[email] = user
                        save_users(users)
                        st.success("Account created and logged in!")
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        safe_rerun()
    if not st.session_state.logged_in:
        st.stop()

# ---------------------------------------
# Top Header with Title, My Account Icon, and Logout
# ---------------------------------------
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    safe_rerun()

header_container = st.container()
with header_container:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>Employee Attrition Prediction Tool</h1>", unsafe_allow_html=True)
    with col2:
        if st.button("👤 My Account", key="account_button"):
            st.session_state.nav = "My Account"
        if st.button("Logout", key="logout_button"):
            logout()

# ---------------------------------------
# Sidebar: Global Settings and Mode Selection
# ---------------------------------------
if st.session_state.nav != "My Account":
    with st.sidebar:
        mode = st.radio("Select Mode", ["Train Mode", "Test Mode"], index=0, key="main_mode")
        disabled_flag = (mode == "Test Mode")
        st.markdown("### Global Settings for Bulk Analysis\n*These settings MUST be filled for bulk analysis*")
        global_avg_age = st.slider("Average Employee Age in Company", 18, 100,
                                   st.session_state.user.get("settings", {}).get("global_avg_age", 35),
                                   key="global_avg_age", disabled=disabled_flag)
        global_female_ratio = st.slider("Women % in Organization", 0, 100,
                                        st.session_state.user.get("settings", {}).get("global_female_ratio", 40),
                                        key="global_female_ratio", disabled=disabled_flag)
        with st.expander("College Tier Retention Settings", expanded=False):
            bulk_tier1 = st.slider("Tier 1 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier1", 60),
                                   key="bulk_tier1", disabled=disabled_flag)
            bulk_tier2 = st.slider("Tier 2 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier2", 50),
                                   key="bulk_tier2", disabled=disabled_flag)
            bulk_tier3 = st.slider("Tier 3 Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_tier3", 40),
                                   key="bulk_tier3", disabled=disabled_flag)
        with st.expander("Industry Retention Settings", expanded=False):
            bulk_industry_retention = {}
            for ind in industry_options:
                default_val = st.session_state.user.get("settings", {}).get("bulk_industry_retention", {}).get(ind, 60 if ind=="Tech" else 50)
                bulk_industry_retention[ind] = st.slider(f"{ind} Retention (%)", 10, 100, default_val,
                                                         key=f"bulk_ind_{ind}", disabled=disabled_flag)
        with st.expander("Company Type Retention Settings", expanded=False):
            bulk_startup = st.slider("Startup Retention (%)", 10, 100,
                                     st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Startup", 60),
                                     key="bulk_startup", disabled=disabled_flag)
            bulk_small = st.slider("Small Size Retention (%)", 10, 100,
                                     st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Small Size", 55),
                                     key="bulk_small", disabled=disabled_flag)
            bulk_mid = st.slider("Mid Size Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("Mid Size", 50),
                                   key="bulk_mid", disabled=disabled_flag)
            bulk_mnc = st.slider("MNC/Giant Company Retention (%)", 10, 100,
                                   st.session_state.user.get("settings", {}).get("bulk_company_retention", {}).get("MNC/Giant Company", 45),
                                   key="bulk_mnc", disabled=disabled_flag)

# ---------------------------------------
# Main Navigation
# ---------------------------------------
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
    if st.button("Back to Main"):
        st.session_state.nav = "Tabs"
else:
    if st.session_state.main_mode == "Test Mode":
        selected_test_industry = st.selectbox("Select Your Industry", industry_options, index=0, key="test_industry")
        st.markdown("""
        <div class="tooltip">Read Instructions
          <span class="tooltiptext">
            Ensure you have trained a model in Train Mode.
            <br><br>
            Upload a CSV/Excel file with columns: Name, Employee Age, Gender, Tenure (Months),
            Pulse(Chance of leaving), Hasn't been promoted, Minimum Promotion Cycle,
            College Tier, Industry, Company Type, Last Performance Rating, Compa Ratio.
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
        if "trust_metrics" in st.session_state:
            tm = st.session_state.trust_metrics
            trust_line = (f"Trustworthiness Summary: Accuracy: {tm['accuracy']:.2f}, Precision: {tm['precision']:.2f}, "
                          f"Recall: {tm['recall']:.2f}, Specificity: {tm['specificity']:.2f}, F1: {tm['f1']:.2f}, "
                          f"ROC AUC: {tm['roc_auc']:.2f}, PR AUC: {tm['pr_auc']:.2f}, Log Loss: {tm['logloss']:.2f}")
            st.markdown(f"<div style='text-align:center; font-size:16px; color: white;'>{trust_line}</div>", unsafe_allow_html=True)
    else:
        selected_test_industry = None

    if st.session_state.main_mode == "Train Mode":
        st.header("Train Mode")
        selected_train_industry = st.selectbox("Select Your Industry", industry_options, key="train_industry")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_train = st.file_uploader("Upload Training Data (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
        with col2:
            st.markdown("### Training File Guide")
            st.markdown("""
            Your training file must include:
            - A **target column** (e.g., Attrition; use 0 for active, 1 for non‑active).
            - **Feature columns:** Employee Age, Gender, Tenure (Months), Pulse(Chance of leaving), Hasn't been promoted,
              Minimum Promotion Cycle, College Tier, Industry, Company Type, Last Performance Rating, Compa Ratio.
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
    else:
        st.header("Bulk Employee Attrition Prediction")
        uploaded_file = st.file_uploader("Upload Bulk Data (CSV or Excel)", type=["csv", "xlsx"], key="bulk_file")
        if uploaded_file is not None:
            try:
                df_bulk = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
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
                st.error(f"❌ Missing columns: {missing}")
            else:
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("🚀 Run Bulk Prediction"):
                        scores, triggers_list, names = [], [], []
                        for idx, row in df_bulk.iterrows():
                            row_dict = row.to_dict()
                            names.append(row_dict.get("Name"))
                            row_dict["Average Employee Age"] = global_avg_age
                            row_dict["Female Employee Ratio"] = global_female_ratio
                            college_tier = row_dict.get("College Tier")
                            if college_tier == "Tier 1":
                                row_dict["College Tier Retention"] = bulk_tier1
                            elif college_tier == "Tier 2":
                                row_dict["College Tier Retention"] = bulk_tier2
                            elif college_tier == "Tier 3":
                                row_dict["College Tier Retention"] = bulk_tier3
                            else:
                                st.warning(f"Row {idx}: Unknown College Tier '{college_tier}'. Using default 40%.")
                                row_dict["College Tier Retention"] = 40
                            ind_val = row_dict.get("Industry")
                            row_dict["Industry Retention"] = bulk_industry_retention.get(ind_val, 50)
                            ctype_val = row_dict.get("Company Type", "Startup")
                            if ctype_val.lower() == "startup":
                                row_dict["Company Type Retention"] = bulk_startup
                            elif "small" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_small
                            elif "mid" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_mid
                            elif "mnc" in ctype_val.lower() or "giant" in ctype_val.lower():
                                row_dict["Company Type Retention"] = bulk_mnc
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

                with btn_cols[1]:
                    if st.session_state.bulk_prediction_complete:
                        st.session_state.enable_what_if = st.checkbox("Enable What-If Analysis", key="whatif_toggle")
                
                # -------------------------------
                # Automatic Cohort Analysis – Boxed Layout
                # -------------------------------
                if st.session_state.bulk_prediction_complete:
                    st.markdown("### Automatic Cohort Analysis")
                    if st.button("Run Automatic Cohort Analysis"):
                        run_auto_cohort_analysis_boxed(st.session_state.bulk_result)
                
                # -------------------------------
                # TOP-LEVEL Custom Cohort Analysis (Custom Builder)
                # -------------------------------
                if st.session_state.bulk_prediction_complete:
                    cohort_toggle = st.checkbox("Enable Custom Cohort Analysis", key="cohort_toggle_top")
                    if cohort_toggle:
                        st.markdown("### Enhanced Cohort Analysis")
                        cohort_df = st.session_state.bulk_result.copy()
                        cohort_filter_values = horizontal_filters(cohort_df, prefix="cohort_", custom_filters_state=st.session_state.cohort_custom_filters)
                        filtered_cohort_df = cohort_df if not cohort_filter_values else apply_filters(cohort_df, cohort_filter_values)
                        st.markdown("#### Filtered Bulk Data for Cohort Analysis")
                        st.dataframe(filtered_cohort_df)
                        st.markdown("#### Define Cohorts")
                        possible_cohort_columns = [col for col in filtered_cohort_df.columns if col not in ["Name", "Attrition Score", "What-If Attrition Score", "What-If Negative Triggers", "Prediction Time"]]
                        primary_cohort_col = st.selectbox("Select Primary Cohort Column", options=possible_cohort_columns, key="primary_cohort")
                        secondary_cohort_option = st.checkbox("Enable Secondary Cohort Dimension", key="enable_secondary_cohort")
                        if secondary_cohort_option:
                            secondary_possible = [col for col in possible_cohort_columns if col != primary_cohort_col]
                            secondary_cohort_col = st.selectbox("Select Secondary Cohort Column", options=secondary_possible, key="secondary_cohort")
                        else:
                            secondary_cohort_col = None
                        if pd.api.types.is_numeric_dtype(filtered_cohort_df[primary_cohort_col]):
                            bin_option = st.checkbox("Bin numeric column?", key="cohort_bin_option")
                            if bin_option:
                                bin_size = st.number_input("Bin size", min_value=1, value=5, key="cohort_bin_size")
                                min_val = int(filtered_cohort_df[primary_cohort_col].min())
                                max_val = int(filtered_cohort_df[primary_cohort_col].max())
                                bins = list(range(min_val, max_val + bin_size, bin_size))
                                filtered_cohort_df["Cohort_Primary"] = pd.cut(filtered_cohort_df[primary_cohort_col], bins=bins)
                            else:
                                filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col]
                        elif pd.api.types.is_datetime64_any_dtype(filtered_cohort_df[primary_cohort_col]):
                            time_window = st.selectbox("Select Time Window", ["Daily", "Weekly", "Monthly", "Quarterly"], key="time_window")
                            if time_window == "Daily":
                                filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col].dt.date
                            elif time_window == "Weekly":
                                filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col].dt.to_period("W").astype(str)
                            elif time_window == "Monthly":
                                filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col].dt.to_period("M").astype(str)
                            elif time_window == "Quarterly":
                                filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col].dt.to_period("Q").astype(str)
                        else:
                            filtered_cohort_df["Cohort_Primary"] = filtered_cohort_df[primary_cohort_col]
                        if secondary_cohort_col is not None:
                            filtered_cohort_df["Cohort_Secondary"] = filtered_cohort_df[secondary_cohort_col]
                            filtered_cohort_df["Cohort_Combined"] = filtered_cohort_df["Cohort_Primary"].astype(str) + " | " + filtered_cohort_df["Cohort_Secondary"].astype(str)
                            cohort_group_col = "Cohort_Combined"
                        else:
                            cohort_group_col = "Cohort_Primary"
                        st.markdown(f"#### Cohort Grouping Based on: {cohort_group_col}")
                        st.dataframe(filtered_cohort_df[[cohort_group_col]].drop_duplicates())
                        st.markdown("#### Cohort Metrics and KPIs")
                        metric_options = ["Count", "Average Employee Age", "Average Tenure (Months)", "Average Compa Ratio", "Average Last Performance Rating"]
                        if "Attrition" in filtered_cohort_df.columns:
                            metric_options.append("Attrition Rate")
                        selected_metric = st.selectbox("Select Primary Metric for Cohort Analysis", options=metric_options, key="cohort_metric")
                        sort_option = st.selectbox("Sort Cohorts By", options=["Cohort", selected_metric], key="cohort_sort")
                        if selected_metric == "Count":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col).size().reset_index(name="Count")
                        elif selected_metric == "Average Employee Age":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col)["Employee Age"].mean().reset_index(name="Average Employee Age")
                        elif selected_metric == "Average Tenure (Months)":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col)["Tenure (Months)"].mean().reset_index(name="Average Tenure (Months)")
                        elif selected_metric == "Average Compa Ratio":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col)["Compa Ratio"].mean().reset_index(name="Average Compa Ratio")
                        elif selected_metric == "Average Last Performance Rating":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col)["Last Performance Rating"].mean().reset_index(name="Average Last Performance Rating")
                        elif selected_metric == "Attrition Rate":
                            cohort_data = filtered_cohort_df.groupby(cohort_group_col)["Attrition"].mean().reset_index(name="Attrition Rate")
                        if sort_option == selected_metric:
                            cohort_data = cohort_data.sort_values(by=selected_metric, ascending=False)
                        else:
                            cohort_data = cohort_data.sort_values(by=cohort_group_col)
                        st.markdown("##### Cohort KPI Table")
                        st.dataframe(cohort_data)
                        st.markdown("#### Visualization Options for Cohort KPIs")
                        viz_options = ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]
                        selected_viz = st.selectbox("Select Visualization Type", options=viz_options, key="cohort_viz")
                        if selected_viz == "Bar Chart":
                            cohort_chart = alt.Chart(cohort_data).mark_bar().encode(
                                x=alt.X(f"{cohort_group_col}:N", title="Cohort"),
                                y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
                                tooltip=[cohort_group_col, selected_metric]
                            )
                        elif selected_viz == "Line Chart":
                            cohort_chart = alt.Chart(cohort_data).mark_line(point=True).encode(
                                x=alt.X(f"{cohort_group_col}:N", title="Cohort"),
                                y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
                                tooltip=[cohort_group_col, selected_metric]
                            )
                        elif selected_viz == "Pie Chart":
                            cohort_chart = alt.Chart(cohort_data).mark_arc().encode(
                                theta=alt.Theta(field=selected_metric, type="quantitative"),
                                color=alt.Color(field=cohort_group_col, type="nominal"),
                                tooltip=[cohort_group_col, selected_metric]
                            )
                        elif selected_viz == "Area Chart":
                            cohort_chart = alt.Chart(cohort_data).mark_area(opacity=0.5).encode(
                                x=alt.X(f"{cohort_group_col}:N", title="Cohort"),
                                y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
                                tooltip=[cohort_group_col, selected_metric]
                            )
                        st.altair_chart(cohort_chart, use_container_width=True)
                # End TOP-LEVEL Custom Cohort Analysis
                
                # -------------------------------
                # Bulk Analysis Charts and What-If Analysis (Custom)
                # -------------------------------
                st.markdown("### Bulk Analysis")
                st.markdown("<hr>", unsafe_allow_html=True)
                filter_values = horizontal_filters(st.session_state.bulk_result, prefix="bulk_", custom_filters_state=st.session_state.bulk_custom_filters)
                filtered_df = apply_filters(st.session_state.bulk_result, filter_values)
                col_table, col_charts = st.columns([1, 4])
                with col_table:
                    st.markdown("#### Data Filters")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("#### Filtered Data Table")
                    st.dataframe(filtered_df)
                    if "Attrition Score" in filtered_df.columns:
                        high_risk = (filtered_df["Attrition Score"] >= 75).sum()
                        mod_high = ((filtered_df["Attrition Score"] >= 60) & (filtered_df["Attrition Score"] < 75)).sum()
                        moderate = ((filtered_df["Attrition Score"] >= 35) & (filtered_df["Attrition Score"] < 60)).sum()
                        low = (filtered_df["Attrition Score"] < 35).sum()
                        risk_df = pd.DataFrame({
                            "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                            "Count": [high_risk, mod_high, moderate, low]
                        })
                        st.markdown("##### Risk Distribution")
                        st.bar_chart(risk_df.set_index("Risk Category"))
                with col_charts:
                    if st.session_state.enable_what_if:
                        st.markdown("#### What-If Analysis")
                        filtered_whatif_df = filtered_df.copy()
                        trig_series = compute_trigger_counts(filtered_whatif_df, "Negative Triggers")
                        
                        trigger_widget_config = {
                            "Low gender diversity": {"widget": "slider", "label": "Women % in Organization", "min": 0, "max": 100, "default": global_female_ratio, "param": "female_ratio"},
                            "Stagnant promotions": {"widget": "slider_pair", "labels": ["Months Since Last Promotion", "Minimum Promotion Cycle"], "min": [0, 12], "max": [60, 60],
                                                      "default": [int(filtered_whatif_df["Hasn't been promoted"].mean()) if not filtered_whatif_df.empty else 0,
                                                                  int(filtered_whatif_df["Minimum Promotion Cycle"].mean()) if not filtered_whatif_df.empty else 24],
                                                      "params": ["not_promoted", "min_cycle"]},
                            "Very low performance rating": {"widget": "selectbox", "label": "Last Performance Rating", "options": [1,2,3,4,5], "default": 3, "param": "rating"},
                            "Low performance rating": {"widget": "selectbox", "label": "Last Performance Rating", "options": [1,2,3,4,5], "default": 3, "param": "rating"},
                            "Low compensation competitiveness": {"widget": "slider", "label": "Compa Ratio (%)", "min": 50, "max": 150, "default": 90, "param": "compa_ratio"},
                            "High compensation ratio": {"widget": "slider", "label": "Compa Ratio (%)", "min": 50, "max": 150, "default": 90, "param": "compa_ratio"},
                            "Low college tier retention": {"widget": "slider_group", "labels": ["Tier 1 Retention (%)", "Tier 2 Retention (%)", "Tier 3 Retention (%)"],
                                                           "min": [10, 10, 10], "max": [100, 100, 100], "default": [bulk_tier1, bulk_tier2, bulk_tier3], "params": ["tier1", "tier2", "tier3"]},
                            "Low industry retention": {"widget": "slider", "label": "Industry Retention (%)", "min": 10, "max": 100, "default": 50, "param": "industry_retention"},
                            "Low company type retention": {"widget": "slider", "label": "Company Type Retention (%)", "min": 10, "max": 100, "default": 60, "param": "company_retention"},
                            "High dissatisfaction (Pulse)": {"widget": "selectbox", "label": "Pulse", "options": ["High", "Medium", "Low"], "default": "High", "param": "pulse"}
                        }
                        
                        whatif_params = {}
                        displayed_params = set()
                        for trigger, config in trigger_widget_config.items():
                            if trigger in trig_series.index:
                                if config["widget"] == "slider":
                                    if config["param"] not in displayed_params:
                                        param_name = config["param"]
                                        whatif_params[param_name] = st.slider(config["label"], config["min"], config["max"], config["default"], key=f"whatif_{param_name}")
                                        displayed_params.add(param_name)
                                elif config["widget"] == "selectbox":
                                    if config["param"] not in displayed_params:
                                        param_name = config["param"]
                                        try:
                                            default_index = config["options"].index(config["default"])
                                        except ValueError:
                                            default_index = 0
                                        whatif_params[param_name] = st.selectbox(config["label"], config["options"], index=default_index, key=f"whatif_{param_name}")
                                        displayed_params.add(param_name)
                                elif config["widget"] == "slider_pair":
                                    param_names = config["params"]
                                    values = []
                                    for i, p in enumerate(param_names):
                                        values.append(st.slider(config["labels"][i], config["min"][i], config["max"][i], config["default"][i], key=f"whatif_{p}"))
                                    for i, p in enumerate(param_names):
                                        whatif_params[p] = values[i]
                                    displayed_params.update(param_names)
                                elif config["widget"] == "slider_group":
                                    param_names = config["params"]
                                    values = []
                                    for i, p in enumerate(param_names):
                                        values.append(st.slider(config["labels"][i], config["min"][i], config["max"][i], config["default"][i], key=f"whatif_{p}"))
                                    for i, p in enumerate(param_names):
                                        whatif_params[p] = values[i]
                                    displayed_params.update(param_names)
                        
                        st.markdown("##### Recalculated Predictions (What-If)")
                        new_scores = []
                        new_triggers_list = []
                        df_bulk_whatif = filtered_whatif_df.copy()
                        for idx, row in df_bulk_whatif.iterrows():
                            new_row = dict(row)
                            new_row["Average Employee Age"] = global_avg_age
                            new_row["Female Employee Ratio"] = whatif_params.get("female_ratio", row.get("Female Employee Ratio", global_female_ratio))
                            new_row["Hasn't been promoted"] = whatif_params.get("not_promoted", row.get("Hasn't been promoted"))
                            new_row["Minimum Promotion Cycle"] = whatif_params.get("min_cycle", row.get("Minimum Promotion Cycle"))
                            new_row["Last Performance Rating"] = whatif_params.get("rating", row.get("Last Performance Rating"))
                            new_row["Compa Ratio"] = whatif_params.get("compa_ratio", row.get("Compa Ratio"))
                            college_tier = row.get("College Tier")
                            if college_tier == "Tier 1":
                                new_row["College Tier Retention"] = whatif_params.get("tier1", row.get("College Tier Retention", bulk_tier1))
                            elif college_tier == "Tier 2":
                                new_row["College Tier Retention"] = whatif_params.get("tier2", row.get("College Tier Retention", bulk_tier2))
                            elif college_tier == "Tier 3":
                                new_row["College Tier Retention"] = whatif_params.get("tier3", row.get("College Tier Retention", bulk_tier3))
                            else:
                                new_row["College Tier Retention"] = row.get("College Tier Retention", 40)
                            industry_val = row.get("Industry")
                            new_row["Industry Retention"] = whatif_params.get("industry_retention", row.get("Industry Retention", bulk_industry_retention.get(industry_val, 50)))
                            ctype_val = row.get("Company Type", "Startup")
                            if ctype_val.lower() == "startup":
                                default_company_retention = bulk_startup
                            elif "small" in ctype_val.lower():
                                default_company_retention = bulk_small
                            elif "mid" in ctype_val.lower():
                                default_company_retention = bulk_mid
                            elif "mnc" in ctype_val.lower() or "giant" in ctype_val.lower():
                                default_company_retention = bulk_mnc
                            else:
                                default_company_retention = 50
                            new_row["Company Type Retention"] = whatif_params.get("company_retention", row.get("Company Type Retention", default_company_retention))
                            new_row["Pulse"] = whatif_params.get("pulse", row.get("Pulse"))
                            try:
                                new_score, new_trigs, _ = predict_attrition(new_row, selected_test_industry)
                            except Exception as e:
                                new_score = None
                                new_trigs = ["Prediction Failed"]
                            new_scores.append(new_score)
                            neg_trigs = [t for t in new_trigs if t in TRIGGER_DETAILS]
                            triggers_str = ", ".join(neg_trigs) if neg_trigs else "None"
                            new_triggers_list.append(triggers_str)
                        df_bulk_whatif["What-If Attrition Score"] = new_scores
                        df_bulk_whatif["What-If Negative Triggers"] = new_triggers_list
                        st.dataframe(df_bulk_whatif)
                        high_risk_w = (df_bulk_whatif["What-If Attrition Score"] >= 75).sum()
                        mod_high_w = ((df_bulk_whatif["What-If Attrition Score"] >= 60) & (df_bulk_whatif["What-If Attrition Score"] < 75)).sum()
                        moderate_w = ((df_bulk_whatif["What-If Attrition Score"] >= 35) & (df_bulk_whatif["What-If Attrition Score"] < 60)).sum()
                        low_w = (df_bulk_whatif["What-If Attrition Score"] < 35).sum()
                        risk_df_w = pd.DataFrame({
                            "Risk Category": ["High (>=75)", "Mod-High (60-74)", "Moderate (35-59)", "Low (<35)"],
                            "Count": [high_risk_w, mod_high_w, moderate_w, low_w]
                        })
                        st.markdown("##### What-If Risk Distribution")
                        st.bar_chart(risk_df_w.set_index("Risk Category"))
                    else:
                        st.markdown("#### Quick Charts & Custom Graph Builder")
                        custom_col, quick_col = st.columns([1, 1])
                        with custom_col:
                            st.markdown("##### Custom Graph Builder")
                            with st.form("custom_graph_form"):
                                x_axis = st.selectbox("Select X Axis", options=filtered_df.columns, key="custom_x")
                                y_axis = st.selectbox("Select Y Axis", options=filtered_df.columns, key="custom_y")
                                data_label = st.selectbox("Select Data Label (Optional)", options=["None"] + list(filtered_df.columns), key="custom_label")
                                submitted_custom = st.form_submit_button("Generate Custom Chart")
                            if submitted_custom:
                                header_text = f"Custom Chart: {x_axis} vs {y_axis}"
                                if data_label != "None":
                                    header_text += f" with {data_label}"
                                config = {"header": header_text, "x_axis": x_axis, "y_axis": y_axis, "data_label": data_label}
                                st.session_state.custom_charts.insert(0, config)
                            st.session_state.custom_charts = [cfg for cfg in st.session_state.custom_charts if isinstance(cfg, dict)]
                            if st.session_state.custom_charts:
                                st.markdown("### Custom Charts")
                                for config in st.session_state.custom_charts:
                                    header_text = config.get("header", f"Custom Chart: {config.get('x_axis', 'X')} vs {config.get('y_axis', 'Y')}")
                                    st.markdown(f"#### {header_text}")
                                    chart = generate_custom_chart(config, filtered_df)
                                    st.altair_chart(chart, use_container_width=True)
                        with quick_col:
                            st.markdown("##### Quick Charts")
                            with st.expander("Distribution Analysis", expanded=False):
                                st.markdown("""
                                **Distribution Analysis:**
                                - Histogram of Attrition Score
                                - Histogram of Employee Age
                                - Histogram of Tenure (Months)
                                - Histogram of Compa Ratio
                                - Histogram of Last Performance Rating
                                - Box Plots for continuous variables (optional)
                                """, unsafe_allow_html=True)
                                if "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    chart1 = alt.Chart(filtered_df).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Attrition Score:Q", bin=alt.Bin(maxbins=20), title="Attrition Score"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart1, use_container_width=True)
                                else:
                                    st.write("No data for Attrition Score Distribution.")
                                if "Employee Age" in filtered_df.columns and not filtered_df.empty:
                                    chart2 = alt.Chart(filtered_df).mark_bar(color="#e45756").encode(
                                        x=alt.X("Employee Age:Q", bin=alt.Bin(maxbins=20), title="Employee Age"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart2, use_container_width=True)
                                else:
                                    st.write("No data for Employee Age Distribution.")
                                if "Tenure (Months)" in filtered_df.columns and not filtered_df.empty:
                                    chart3 = alt.Chart(filtered_df).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Tenure (Months):Q", bin=alt.Bin(maxbins=20), title="Tenure (Months)"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart3, use_container_width=True)
                                else:
                                    st.write("No data for Tenure Distribution.")
                                if "Compa Ratio" in filtered_df.columns and not filtered_df.empty:
                                    chart4 = alt.Chart(filtered_df).mark_bar(color="#e45756").encode(
                                        x=alt.X("Compa Ratio:Q", bin=alt.Bin(maxbins=20), title="Compa Ratio"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart4, use_container_width=True)
                                else:
                                    st.write("No data for Compa Ratio Distribution.")
                                if "Last Performance Rating" in filtered_df.columns and not filtered_df.empty:
                                    chart5 = alt.Chart(filtered_df).mark_bar(color="#4c78a8").encode(
                                        x=alt.X("Last Performance Rating:O", title="Last Performance Rating"),
                                        y=alt.Y("count()", title="Frequency")
                                    )
                                    st.altair_chart(chart5, use_container_width=True)
                                else:
                                    st.write("No data for Performance Rating Distribution.")
                            with st.expander("Comparative Analysis", expanded=False):
                                st.markdown("""
                                **Comparative Analysis:**
                                - Scatter Plot: Employee Age vs Attrition Score
                                - Scatter Plot: Tenure (Months) vs Attrition Score
                                - Scatter Plot: Compa Ratio vs Attrition Score
                                - Box Plot: Attrition Score by Gender
                                - Box Plot: Attrition Score by College Tier
                                - Box Plot: Attrition Score by Industry
                                - Bar Chart: Average Attrition Score by Company Type
                                """, unsafe_allow_html=True)
                                if "Employee Age" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    scatter1 = alt.Chart(filtered_df).mark_circle(size=60, color="#4c78a8").encode(
                                        x=alt.X("Employee Age:Q", title="Employee Age"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Employee Age", "Attrition Score"]
                                    )
                                    st.altair_chart(scatter1, use_container_width=True)
                                if "Tenure (Months)" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    scatter2 = alt.Chart(filtered_df).mark_circle(size=60, color="#e45756").encode(
                                        x=alt.X("Tenure (Months):Q", title="Tenure (Months)"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Tenure (Months)", "Attrition Score"]
                                    )
                                    st.altair_chart(scatter2, use_container_width=True)
                                if "Compa Ratio" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    scatter3 = alt.Chart(filtered_df).mark_circle(size=60, color="#4c78a8").encode(
                                        x=alt.X("Compa Ratio:Q", title="Compa Ratio"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score"),
                                        tooltip=["Compa Ratio", "Attrition Score"]
                                    )
                                    st.altair_chart(scatter3, use_container_width=True)
                                if "Gender" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    box1 = alt.Chart(filtered_df).mark_boxplot(color="#e45756").encode(
                                        x=alt.X("Gender:N", title="Gender"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score")
                                    )
                                    st.altair_chart(box1, use_container_width=True)
                                if "College Tier" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    box2 = alt.Chart(filtered_df).mark_boxplot(color="#4c78a8").encode(
                                        x=alt.X("College Tier:N", title="College Tier"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score")
                                    )
                                    st.altair_chart(box2, use_container_width=True)
                                if "Industry" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    box3 = alt.Chart(filtered_df).mark_boxplot(color="#e45756").encode(
                                        x=alt.X("Industry:N", title="Industry"),
                                        y=alt.Y("Attrition Score:Q", title="Attrition Score")
                                    )
                                    st.altair_chart(box3, use_container_width=True)
                                if "Company Type" in filtered_df.columns and "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                                    avg_df = filtered_df.groupby("Company Type")["Attrition Score"].mean().reset_index()
                                    bar1 = alt.Chart(avg_df).mark_bar().encode(
                                        x=alt.X("Company Type:N", title="Company Type"),
                                        y=alt.Y("Attrition Score:Q", title="Average Attrition Score"),
                                        tooltip=["Company Type", "Attrition Score"]
                                    )
                                    st.altair_chart(bar1, use_container_width=True)
                            with st.expander("Correlation Analysis", expanded=False):
                                st.markdown("""
                                **Correlation Analysis:**
                                - Correlation Heatmap for numeric variables
                                - Scatter Plot: Employee Age vs Tenure (Months)
                                - Scatter Plot: Employee Age vs Compa Ratio
                                - Scatter Plot: Tenure (Months) vs Compa Ratio
                                """, unsafe_allow_html=True)
                                try:
                                    numeric_df = filtered_df.select_dtypes(include=[np.number])
                                    if not numeric_df.empty:
                                        corr = numeric_df.corr().reset_index().melt(id_vars="index")
                                        heatmap = alt.Chart(corr).mark_rect().encode(
                                            x=alt.X("index:N", title=""),
                                            y=alt.Y("variable:N", title=""),
                                            color=alt.Color("value:Q", scale=alt.Scale(scheme='redblue')),
                                            tooltip=["index", "variable", "value"]
                                        )
                                        st.altair_chart(heatmap, use_container_width=True)
                                    else:
                                        st.write("No numeric data for correlation heatmap.")
                                except Exception as e:
                                    st.write("Correlation Heatmap not available.")
                                if "Employee Age" in numeric_df.columns and "Tenure (Months)" in numeric_df.columns:
                                    scatter_a = alt.Chart(filtered_df).mark_circle(size=40).encode(
                                        x=alt.X("Employee Age:Q", title="Employee Age"),
                                        y=alt.Y("Tenure (Months):Q", title="Tenure (Months)"),
                                        tooltip=["Employee Age", "Tenure (Months)"]
                                    )
                                    st.altair_chart(scatter_a, use_container_width=True)
                                if "Employee Age" in numeric_df.columns and "Compa Ratio" in numeric_df.columns:
                                    scatter_b = alt.Chart(filtered_df).mark_circle(size=40).encode(
                                        x=alt.X("Employee Age:Q", title="Employee Age"),
                                        y=alt.Y("Compa Ratio:Q", title="Compa Ratio"),
                                        tooltip=["Employee Age", "Compa Ratio"]
                                    )
                                    st.altair_chart(scatter_b, use_container_width=True)
                                if "Tenure (Months)" in numeric_df.columns and "Compa Ratio" in numeric_df.columns:
                                    scatter_c = alt.Chart(filtered_df).mark_circle(size=40).encode(
                                        x=alt.X("Tenure (Months):Q", title="Tenure (Months)"),
                                        y=alt.Y("Compa Ratio:Q", title="Compa Ratio"),
                                        tooltip=["Tenure (Months)", "Compa Ratio"]
                                    )
                                    st.altair_chart(scatter_c, use_container_width=True)
