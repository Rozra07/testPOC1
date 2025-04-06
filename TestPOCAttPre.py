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

#####################################################################
# 1) ALTair THEME & PAGE CONFIG
#####################################################################
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

st.set_page_config(layout="wide")


#####################################################################
# 2) SESSION STATE INITIALIZATION
#####################################################################
def safe_rerun():
    """Safely rerun if Streamlit version supports it; else instruct user."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Refresh functionality is not available. Update Streamlit >=0.65.0.")

for key in [
    "logged_in", "nav", "user", "bulk_prediction_complete", "bulk_result", 
    "enable_what_if", "custom_charts", "custom_filters", "training_data", "trust_metrics"
]:
    if key not in st.session_state:
        if key in ["custom_charts", "custom_filters"]:
            st.session_state[key] = []
        elif key == "logged_in":
            st.session_state[key] = False
        elif key == "nav":
            st.session_state[key] = "Tabs"
        else:
            st.session_state[key] = None


#####################################################################
# 3) USER MANAGEMENT
#####################################################################
import json

USERS_FILE = "users.json"
USER_DATA_DIR = "user_data"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
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
    return []

def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    safe_rerun()


#####################################################################
# 4) HELPER: COLORED METRIC
#####################################################################
def colored_metric(value, threshold, higher_better=True, is_lower=False):
    """
    Renders a numeric value as colored text. 
      If is_lower=True, we want the value <= threshold to be green, else red.
      Otherwise, value >= threshold is green, else red.
    """
    if is_lower:
        color = "green" if value <= threshold else "red"
    else:
        color = "green" if value >= threshold else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.2f}</span>'


#####################################################################
# 5) GLOBAL INDUSTRY OPTIONS
#####################################################################
industry_options = [
    "Tech", "Finance", "Healthcare", "Education", "Manufacturing", 
    "Retail", "Energy", "Telecommunications", "Government", "Nonprofit", "Other"
]


#####################################################################
# 6) MODEL TRAIN/LOAD UTILS
#####################################################################
def update_industry_record(industry, model_file, scaler_file, feature_file):
    csv_filename = "industry_models.csv"
    record = {
        "Industry": industry,
        "Model_File": model_file,
        "Scaler_File": scaler_file,
        "Feature_File": feature_file,
        "Training_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if industry in df["Industry"].values:
            df.loc[df["Industry"]==industry, ["Model_File","Scaler_File","Feature_File","Training_Date"]] = \
                [model_file, scaler_file, feature_file, record["Training_Date"]]
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(csv_filename, index=False)

def load_model(industry):
    mf = f"{industry}_model.pkl"
    sf = f"{industry}_scaler.pkl"
    ff = f"{industry}_feature_columns.pkl"
    if os.path.exists(mf) and os.path.exists(sf) and os.path.exists(ff):
        with open(mf,"rb") as f:
            model = pickle.load(f)
        with open(sf,"rb") as f:
            scaler = pickle.load(f)
        with open(ff,"rb") as f:
            feats = pickle.load(f)
        return model, scaler, feats
    else:
        st.error("No trained model found for the selected industry. Train in Train Mode first.")
        return None, None, None


#####################################################################
# 7) RULES & TRIGGERS
#####################################################################
TRIGGER_DETAILS = {
    "Low gender diversity": {...},  # truncated for brevity
    "Stagnant promotions": {...},
    "Very low performance rating": {...},
    "Low performance rating": {...},
    "Low compensation competitiveness": {...}
}

def compute_weighted_attrition(employee, return_triggers=False):
    """
    A typical rule-based approach, returning a score 0-100 + triggers.
    Combine with ML in final predictions.
    """
    score = 0
    extreme_factors = 0
    triggers = []

    # 1) Low gender diversity
    if employee["Gender"]=="Female" and employee["Female Employee Ratio"]<=15:
        score+=30; extreme_factors+=1; triggers.append("Low gender diversity")

    # 2) Stagnant promotions
    if employee["Hasn't been promoted"] >= 2*employee["Minimum Promotion Cycle"]:
        score+=30; extreme_factors+=1; triggers.append("Stagnant promotions")

    # 3) Performance rating
    rating = employee["Last Performance Rating"]
    if rating==1:
        score+=25; extreme_factors+=1; triggers.append("Very low performance rating")
    elif rating==2:
        score+=15; extreme_factors+=0.5; triggers.append("Low performance rating")
    elif rating==5:
        score-=15; extreme_factors-=0.5; triggers.append("Excellent performance rating")

    # 4) Compa ratio
    c_ratio = employee["Compa Ratio"]
    if c_ratio<70:
        score+=25; extreme_factors+=1; triggers.append("Low compensation competitiveness")
    elif c_ratio<80:
        score+=20; extreme_factors+=0.8; triggers.append("Low compensation competitiveness")
    elif c_ratio>110:
        score-=15; extreme_factors-=0.5; triggers.append("High compensation ratio")

    # 5) Retention
    if employee["College Tier Retention"]<15:
        score+=15; extreme_factors+=0.5; triggers.append("Low college tier retention")
    if employee["Industry Retention"]<15:
        score+=15; extreme_factors+=0.5; triggers.append("Low industry retention")
    if employee["Company Type Retention"]<15:
        score+=15; extreme_factors+=0.5; triggers.append("Low company type retention")

    # 6) Pulse
    if employee["Pulse"]=="High":
        score+=20; extreme_factors+=0.5; triggers.append("High dissatisfaction (Pulse)")
    elif employee["Pulse"]=="Low":
        score-=20; extreme_factors-=0.5; triggers.append("Low dissatisfaction (Pulse)")

    # Multiply for "extreme_factors"
    if extreme_factors==2:
        score = min(100, score*1.3)
    elif extreme_factors==3:
        score = min(100, score*1.6)
    elif extreme_factors>=4:
        score = min(100, score*2)

    final_score = min(100, max(0, score))
    if return_triggers:
        return final_score, triggers
    return final_score


#####################################################################
# 8) TRAIN MODEL
#####################################################################
def train_model(training_df, target_column, industry):
    from sklearn.metrics import (
        roc_curve, auc, confusion_matrix, classification_report, accuracy_score,
        precision_score, recall_score, f1_score, log_loss, average_precision_score
    )

    st.write("Training on data shape:", training_df.shape)
    X = training_df.drop(columns=[target_column])
    y = training_df[target_column]

    X_enc = pd.get_dummies(X)
    feat_cols = list(X_enc.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_scaled, y)

    st.write("Model coefficients:", model.coef_)

    preds = model.predict_proba(X_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr,tpr)
    cm = confusion_matrix(y, model.predict(X_scaled))
    report = classification_report(y, model.predict(X_scaled), output_dict=True)

    st.subheader("Model Evaluation Metrics")
    st.write(f"**ROC AUC:** {roc_auc:.2f}")

    fig, ax = plt.subplots(facecolor='black')
    ax.set_facecolor('black')
    ax.plot(fpr, tpr, color='cyan', label=f"ROC (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel("FPR", color='white')
    ax.set_ylabel("TPR", color='white')
    ax.set_title("ROC Curve", color='white')
    ax.legend(loc="best", facecolor='black', edgecolor='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"]))
    st.write("**Classification Report:**")
    st.json(report)

    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0
    pr_auc = average_precision_score(y, preds)
    logloss = log_loss(y, model.predict_proba(X_scaled))

    st.session_state.trust_metrics = dict(
        accuracy=accuracy, precision=precision, recall=recall, specificity=specificity,
        f1=f1, roc_auc=roc_auc, pr_auc=pr_auc, logloss=logloss
    )

    # Display trust table
    table_html = f"""
    <table style="width:100%; border:1px solid white; border-collapse:collapse;">
      <tr><th>Metric</th><th>Value</th><th>Ideal</th></tr>
      <tr><td>Accuracy</td><td>{colored_metric(accuracy,0.8)}</td><td>>=0.8</td></tr>
      <tr><td>Precision</td><td>{colored_metric(precision,0.7)}</td><td>>=0.7</td></tr>
      <tr><td>Recall</td><td>{colored_metric(recall,0.7)}</td><td>>=0.7</td></tr>
      <tr><td>Specificity</td><td>{colored_metric(specificity,0.7)}</td><td>>=0.7</td></tr>
      <tr><td>F1 Score</td><td>{colored_metric(f1,0.7)}</td><td>>=0.7</td></tr>
      <tr><td>ROC AUC</td><td>{colored_metric(roc_auc,0.8)}</td><td>>=0.8</td></tr>
      <tr><td>PR AUC</td><td>{colored_metric(pr_auc,0.7)}</td><td>>=0.7</td></tr>
      <tr><td>Log Loss</td><td>{colored_metric(logloss,0.5,is_lower=True)}</td><td><=0.5</td></tr>
    </table>
    """
    st.markdown("### Trustworthiness of Model")
    st.markdown(table_html, unsafe_allow_html=True)

    # Save model
    mfn = f"{industry}_model.pkl"
    sfn = f"{industry}_scaler.pkl"
    ffn = f"{industry}_feature_columns.pkl"
    with open(mfn,"wb") as f:
        pickle.dump(model,f)
    with open(sfn,"wb") as f:
        pickle.dump(scaler,f)
    with open(ffn,"wb") as f:
        pickle.dump(feat_cols,f)

    st.success("Model trained & saved.")
    tr_acc = model.score(X_scaled,y)*100
    st.info(f"Training Accuracy: {tr_acc:.2f}%")

    update_industry_record(industry, mfn, sfn, ffn)

    # Save user settings
    user = st.session_state.user
    user_settings = user.get("settings",{})
    user_settings["global_avg_age"] = st.session_state.global_avg_age
    user_settings["global_female_ratio"] = st.session_state.global_female_ratio
    user_settings["bulk_tier1"] = st.session_state.bulk_tier1
    user_settings["bulk_tier2"] = st.session_state.bulk_tier2
    user_settings["bulk_tier3"] = st.session_state.bulk_tier3

    # store all industry retention
    dct = {}
    for ind in industry_options:
        dct[ind] = st.session_state.get(f"bulk_ind_{ind}",50)
    user_settings["bulk_industry_retention"] = dct
    user_settings["bulk_company_retention"] = {
        "Startup": st.session_state.bulk_startup,
        "Small Size": st.session_state.bulk_small,
        "Mid Size": st.session_state.bulk_mid,
        "MNC/Giant Company": st.session_state.bulk_mnc
    }
    user["settings"] = user_settings
    us = load_users()
    us[user["email"]] = user
    save_users(us)
    save_user_event(user["email"], "training", {"action":"Model retrained","industry":industry})


#####################################################################
# 9) PREDICT (COMBINE ML + RULES)
#####################################################################
def predict_attrition(employee_data, industry):
    """
    50% from logistic regression, 50% from rule-based
    """
    model, scaler, feat_cols = load_model(industry)
    if model is None:
        return None,None,None

    df_input = pd.DataFrame([employee_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feat_cols, fill_value=0)
    X_scaled = scaler.transform(df_input)

    ml_prob = model.predict_proba(X_scaled)[:,1][0]*100
    rule_prob, triggers = compute_weighted_attrition(employee_data, True)
    combined = 0.5*ml_prob + 0.5*rule_prob
    return combined, triggers, ml_prob


#####################################################################
# 10) CSV GENERATION
#####################################################################
def generate_sample_csv():
    data = {
        "Employee Age":[30,45],
        "Gender":["Male","Female"],
        "Tenure (Months)":[36,48],
        "Pulse":["Medium","High"],
        "Hasn't been promoted":[12,36],
        "Minimum Promotion Cycle":[24,24],
        "College Tier":["Tier 1","Tier 2"],
        "Industry":["Tech","Finance"],
        "Company Type":["Startup","Enterprise"],
        "Last Performance Rating":[3,1],
        "Compa Ratio":[90,65]
    }
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf,index=False)
    return buf.getvalue()

def generate_dummy_training_file():
    df = pd.DataFrame({
        "Name":["Example1","Example2","Example3"],
        "Employee Age":[30,40,35],
        "Gender":["Male","Female","Male"],
        "Tenure (Months)":[36,48,24],
        "Pulse":["Medium","High","Low"],
        "Hasn't been promoted":[12,30,15],
        "Minimum Promotion Cycle":[24,24,24],
        "College Tier":["Tier 1","Tier 2","Tier 3"],
        "Industry":["Tech","Finance","Healthcare"],
        "Company Type":["Startup","Enterprise","SME"],
        "Last Performance Rating":[3,1,4],
        "Compa Ratio":[90,65,100],
        "Attrition":[0,1,0]
    })
    buf = io.StringIO()
    df.to_csv(buf,index=False)
    return buf.getvalue()

#####################################################################
# 11) TRIGGER COUNTS
#####################################################################
def compute_trigger_counts(df, column_name):
    """
    Column has comma-separated triggers. Let's parse & count freq.
    """
    triggers_list = []
    for val in df[column_name].dropna():
        txt = val.strip()
        if txt and txt!="None":
            triggers_list.extend([x.strip() for x in txt.split(",") if x.strip()])
    if triggers_list:
        return pd.Series(triggers_list).value_counts()
    else:
        return pd.Series(dtype=int)


#####################################################################
# 12) CUSTOM CHARTS (HANDLES TEXT vs. TEXT, ETC.)
#####################################################################
def generate_custom_chart(config, data):
    x_axis = config.get("x_axis")
    y_axis = config.get("y_axis")
    data_label = config.get("data_label", "None")

    st.write("DEBUG: generate_custom_chart called with:")
    st.write(" - x_axis:", x_axis, " - y_axis:", y_axis, " - data_label:", data_label)
    st.write(" - data shape:", data.shape)
    st.write(" - columns:", data.columns.tolist())

    if data.empty:
        st.warning("Data is empty; no chart.")
        return alt.Chart(pd.DataFrame({"Info":[]}))\
            .mark_text().encode(text="Info:N")

    # Check if x_axis or y_axis is missing
    missing = [c for c in [x_axis,y_axis] if c not in data.columns]
    if missing:
        err = f"Missing column(s): {missing}"
        return alt.Chart(pd.DataFrame({"Error":[err]})).mark_text(color="red").encode(text="Error:N")

    # Special case: if x_axis=="Negative Triggers" or y_axis=="Negative Triggers"
    if x_axis=="Negative Triggers" or y_axis=="Negative Triggers":
        ct = compute_trigger_counts(data, "Negative Triggers").reset_index()
        ct.columns = ["Trigger","Count"]
        if ct.empty:
            return alt.Chart(pd.DataFrame({"NoTriggers":[]}))\
                .mark_text().encode(text="NoTriggers:N")
        chart = alt.Chart(ct).mark_bar().encode(
            x=alt.X("Trigger:N", sort='-y', title="Negative Triggers"),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=["Trigger","Count"]
        )
        return chart

    # handle color dimension
    use_color = (data_label!="None") and (data_label in data.columns)

    # numeric vs. cat check
    x_is_num = pd.api.types.is_numeric_dtype(data[x_axis])
    y_is_num = pd.api.types.is_numeric_dtype(data[y_axis])

    base = alt.Chart(data)
    if x_is_num and y_is_num:
        # scatter
        mark = base.mark_circle(size=60)
        enc = {
            "x": alt.X(f"{x_axis}:Q", title=x_axis),
            "y": alt.Y(f"{y_axis}:Q", title=y_axis),
            "tooltip":[x_axis,y_axis]
        }
        if use_color:
            is_num_label = pd.api.types.is_numeric_dtype(data[data_label])
            enc["color"] = alt.Color(f"{data_label}:{'Q' if is_num_label else 'N'}", title=data_label)
            enc["tooltip"].append(data_label)
        return mark.encode(**enc)

    elif (not x_is_num) and y_is_num:
        # box: x=cat, y=num
        mark = base.mark_boxplot()
        enc = {
            "x": alt.X(f"{x_axis}:N", title=x_axis),
            "y": alt.Y(f"{y_axis}:Q", title=y_axis)
        }
        if use_color:
            is_num_label = pd.api.types.is_numeric_dtype(data[data_label])
            enc["color"] = alt.Color(f"{data_label}:{'Q' if is_num_label else 'N'}", title=data_label)
        return mark.encode(**enc)

    elif x_is_num and (not y_is_num):
        # box: x=num, y=cat
        mark = base.mark_boxplot()
        enc = {
            "x": alt.X(f"{x_axis}:Q", title=x_axis),
            "y": alt.Y(f"{y_axis}:N", title=y_axis)
        }
        if use_color:
            is_num_label = pd.api.types.is_numeric_dtype(data[data_label])
            enc["color"] = alt.Color(f"{data_label}:{'Q' if is_num_label else 'N'}", title=data_label)
        return mark.encode(**enc)
    else:
        # x,y both text => bar
        if x_axis!=y_axis:
            # stacked bar
            mark = base.mark_bar()
            enc = {
                "x":alt.X(f"{x_axis}:N", title=x_axis),
                "y":alt.Y("count()", title="Count"),
                "color":alt.Color(f"{y_axis}:N", title=y_axis),
                "tooltip":[x_axis,y_axis,alt.Tooltip("count()", title="Count")]
            }
            return mark.encode(**enc)
        else:
            # x==y => single cat => simple bar
            mark = base.mark_bar()
            enc = {
                "x":alt.X(f"{x_axis}:N", title=x_axis),
                "y":alt.Y("count()", title="Count"),
                "tooltip":[x_axis, alt.Tooltip("count()", title="Count")]
            }
            if use_color:
                is_num_label = pd.api.types.is_numeric_dtype(data[data_label])
                enc["color"] = alt.Color(f"{data_label}:{'Q' if is_num_label else 'N'}", title=data_label)
            return mark.encode(**enc)


#####################################################################
# 13) HORIZONTAL FILTERS (NO NESTED COLUMNS)
#####################################################################
def horizontal_filters(df):
    filter_vals = {}
    with st.expander("Filters", expanded=True):
        st.write("Use these filters to limit rows in your data.")
        # Basic filter for "Attrition Score"
        if "Attrition Score" in df.columns:
            rng = st.slider("Attrition Score Range", 0,100, (0,100))
            filter_vals["AttritionScoreRange"] = rng

        # handle custom filters from session state
        for fid in st.session_state.custom_filters:
            st.markdown("---")
            possible_cols = [
                c for c in df.columns
                if c not in ["Name","Attrition Score","What-If Attrition Score","What-If Negative Triggers","Prediction Time"]
            ]
            col_selected = st.selectbox("Select Column", possible_cols, key=f"col_{fid}")
            if pd.api.types.is_numeric_dtype(df[col_selected]):
                minv = float(df[col_selected].min())
                maxv = float(df[col_selected].max())
                r = st.slider(f"{col_selected} range", minv, maxv, (minv,maxv), key=f"rng_{fid}")
                filter_vals[f"rng_{fid}"] = (col_selected, r[0], r[1])
            else:
                uniq = df[col_selected].dropna().unique().tolist()
                sel = st.multiselect(f"{col_selected} values", uniq, default=uniq, key=f"val_{fid}")
                filter_vals[f"val_{fid}"] = (col_selected, sel)
            if st.button("Remove Filter", key=f"rm_{fid}"):
                st.session_state.custom_filters.remove(fid)
                safe_rerun()

        if st.button("Add Another Filter"):
            st.session_state.custom_filters.append(str(datetime.now().timestamp()))
            safe_rerun()
    return filter_vals

def apply_filters(df, filter_vals):
    df2 = df.copy()
    # built-in "AttritionScoreRange"
    if "AttritionScoreRange" in filter_vals and "Attrition Score" in df2.columns:
        lo,hi = filter_vals["AttritionScoreRange"]
        df2 = df2[(df2["Attrition Score"]>=lo)&(df2["Attrition Score"]<=hi)]

    # custom
    for k,v in filter_vals.items():
        if k.startswith("rng_"):
            col, mn, mx = v
            df2 = df2[(df2[col]>=mn)&(df2[col]<=mx)]
        elif k.startswith("val_"):
            col, arr = v
            df2 = df2[df2[col].isin(arr)]
    return df2


#####################################################################
# 14) LOGIN / SIGNUP
#####################################################################
if not st.session_state.logged_in:
    st.title("Employee Attrition Prediction Tool - Login / Sign Up")
    auth_mode = st.radio("Select Mode", ["Login","Sign Up"], index=0)
    if auth_mode=="Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                us = load_users()
                if email in us and us[email]["password"]==password:
                    st.session_state.logged_in = True
                    st.session_state.user = us[email]
                    st.success(f"Welcome back, {us[email]['name']}!")
                    safe_rerun()
                else:
                    st.error("Invalid email or password.")
    else:
        with st.form("signup_form"):
            name = st.text_input("Name")
            designation = st.text_input("Designation")
            company = st.text_input("Company Name")
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")
            cpwd=st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if pwd!=cpwd:
                    st.error("Passwords do not match.")
                else:
                    us = load_users()
                    if email in us:
                        st.error("Email already exists. Please log in.")
                    else:
                        userobj = dict(
                            name=name, designation=designation, company=company,
                            email=email, password=pwd, settings={}
                        )
                        us[email] = userobj
                        save_users(us)
                        st.session_state.logged_in = True
                        st.session_state.user = userobj
                        st.success("Account created and logged in!")
                        safe_rerun()
    st.stop()


#####################################################################
# 15) TOP HEADER (IF LOGGED IN)
#####################################################################
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1>Employee Attrition Prediction Tool</h1>", unsafe_allow_html=True)
with col2:
    if st.button("My Account"):
        st.session_state.nav = "My Account"
    if st.button("Logout"):
        logout()


#####################################################################
# 16) SIDEBAR: Global Settings (IF NOT My Account)
#####################################################################
if st.session_state.nav!="My Account":
    mode = st.sidebar.radio("Select Mode", ["Train Mode","Test Mode"], index=0, key="main_mode")
    disabled_flag = (mode=="Test Mode")
    st.sidebar.markdown("### Global Settings for Bulk Analysis\n*(Must be filled for predictions.)*")

    st.session_state.global_avg_age = st.sidebar.slider(
        "Average Employee Age", 18, 100,
        st.session_state.user.get("settings",{}).get("global_avg_age",35),
        key="global_avg_age", disabled=disabled_flag
    )
    st.session_state.global_female_ratio = st.sidebar.slider(
        "Women % in Org", 0,100,
        st.session_state.user.get("settings",{}).get("global_female_ratio",40),
        key="global_female_ratio", disabled=disabled_flag
    )

    with st.sidebar.expander("College Tier Retention Settings", expanded=False):
        st.session_state.bulk_tier1 = st.slider(
            "Tier1 Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_tier1",60),
            key="bulk_tier1", disabled=disabled_flag
        )
        st.session_state.bulk_tier2 = st.slider(
            "Tier2 Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_tier2",50),
            key="bulk_tier2", disabled=disabled_flag
        )
        st.session_state.bulk_tier3 = st.slider(
            "Tier3 Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_tier3",40),
            key="bulk_tier3", disabled=disabled_flag
        )

    with st.sidebar.expander("Industry Retention Settings", expanded=False):
        for ind in industry_options:
            default_val = st.session_state.user.get("settings",{}).get("bulk_industry_retention",{}).get(ind, 60 if ind=="Tech" else 50)
            st.session_state[f"bulk_ind_{ind}"] = st.slider(
                f"{ind} Retention (%)", 10,100, default_val,
                key=f"bulk_ind_{ind}", disabled=disabled_flag
            )

    with st.sidebar.expander("Company Type Retention Settings", expanded=False):
        st.session_state.bulk_startup = st.slider(
            "Startup Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_company_retention",{}).get("Startup",60),
            key="bulk_startup", disabled=disabled_flag
        )
        st.session_state.bulk_small = st.slider(
            "Small Size Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_company_retention",{}).get("Small Size",55),
            key="bulk_small", disabled=disabled_flag
        )
        st.session_state.bulk_mid = st.slider(
            "Mid Size Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_company_retention",{}).get("Mid Size",50),
            key="bulk_mid", disabled=disabled_flag
        )
        st.session_state.bulk_mnc = st.slider(
            "MNC/Giant Company Retention (%)", 10,100,
            st.session_state.user.get("settings",{}).get("bulk_company_retention",{}).get("MNC/Giant Company",45),
            key="bulk_mnc", disabled=disabled_flag
        )


#####################################################################
# 17) MY ACCOUNT PAGE
#####################################################################
if st.session_state.nav=="My Account":
    st.markdown("<div style='text-align:center'><h2>My Account</h2></div>", unsafe_allow_html=True)
    usr = st.session_state.user
    st.write("### Account Information")
    st.write(f"**Name:** {usr.get('name','')}")
    st.write(f"**Designation:** {usr.get('designation','')}")
    st.write(f"**Company:** {usr.get('company','')}")
    st.write(f"**Email:** {usr.get('email','')}")
    st.write("### Saved Global Settings")
    if usr.get("settings"):
        st.json(usr["settings"])
    else:
        st.info("No global settings saved yet. Please train your model to save settings.")

    st.write("### Analysis History")
    hx = load_user_history(usr["email"])
    if hx:
        st.dataframe(pd.DataFrame(hx))
    else:
        st.info("No history found.")
    if st.button("Back to Main"):
        st.session_state.nav="Tabs"


#####################################################################
# 18) MAIN TABS (Train Mode or Test Mode)
#####################################################################
elif st.session_state.main_mode=="Train Mode":
    st.header("Train Mode")
    chosen_industry = st.selectbox("Select Your Industry", industry_options, key="chosen_industry_train")
    c1,c2 = st.columns(2)
    with c1:
        up_train = st.file_uploader("Upload Training (CSV/Excel)", type=["csv","xlsx"], key="train_file")
    with c2:
        st.markdown("### Training File Guide")
        st.markdown("""
        Must have a **target column** (e.g. 'Attrition').  
        Feature columns: Age, Gender, Tenure (Months), etc.
        """)
        st.download_button(
            label="Download Dummy Training File",
            data=generate_dummy_training_file(),
            file_name="dummy_training_file.csv",
            mime="text/csv"
        )

    target_col = st.text_input("Target Column Name", value="Attrition")
    if up_train:
        try:
            if up_train.name.endswith(".csv"):
                df_tr = pd.read_csv(up_train)
            else:
                df_tr = pd.read_excel(up_train)
            st.write("### Training Data Preview:")
            st.dataframe(df_tr.head())
            if st.button("Train Model"):
                train_model(df_tr, target_col, chosen_industry)
                st.session_state.training_data = df_tr.copy()
        except Exception as e:
            st.error(f"Error reading file: {e}")

else:
    # TEST MODE => Bulk
    st.header("Bulk Employee Attrition Prediction")
    chosen_industry = st.selectbox("Select Your Industry", industry_options, key="test_industry")
    up_bulk = st.file_uploader("Upload Bulk Data (CSV/Excel)", type=["csv","xlsx"], key="bulk_file")
    if up_bulk:
        try:
            if up_bulk.name.endswith(".csv"):
                df_bulk = pd.read_csv(up_bulk)
            else:
                df_bulk = pd.read_excel(up_bulk)
            st.write("### Bulk Data Preview:")
            st.dataframe(df_bulk.head())

            needed_cols = [
                "Name","Employee Age","Gender","Tenure (Months)","Pulse",
                "Hasn't been promoted","Minimum Promotion Cycle","College Tier",
                "Industry","Company Type","Last Performance Rating","Compa Ratio"
            ]
            missing = [c for c in needed_cols if c not in df_bulk.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("🚀 Run Bulk Prediction"):
                        final_scores=[]
                        final_triggers=[]
                        for idx,row in df_bulk.iterrows():
                            rowd = row.to_dict()
                            rowd["Average Employee Age"] = st.session_state.global_avg_age
                            rowd["Female Employee Ratio"] = st.session_state.global_female_ratio

                            # College Tier
                            c_tier = rowd.get("College Tier","Tier 3")
                            if c_tier=="Tier 1":
                                rowd["College Tier Retention"] = st.session_state.bulk_tier1
                            elif c_tier=="Tier 2":
                                rowd["College Tier Retention"] = st.session_state.bulk_tier2
                            elif c_tier=="Tier 3":
                                rowd["College Tier Retention"] = st.session_state.bulk_tier3
                            else:
                                rowd["College Tier Retention"] = 40

                            # Industry
                            i_val = rowd.get("Industry","Tech")
                            rowd["Industry Retention"] = st.session_state.get(f"bulk_ind_{i_val}",50)

                            # Company Type
                            ctype = rowd.get("Company Type","Startup").lower()
                            if "startup" in ctype:
                                rowd["Company Type Retention"] = st.session_state.bulk_startup
                            elif "small" in ctype:
                                rowd["Company Type Retention"] = st.session_state.bulk_small
                            elif "mid" in ctype:
                                rowd["Company Type Retention"] = st.session_state.bulk_mid
                            elif "mnc" in ctype or "giant" in ctype:
                                rowd["Company Type Retention"] = st.session_state.bulk_mnc
                            else:
                                rowd["Company Type Retention"] = 50

                            try:
                                score, triggers, _ = predict_attrition(rowd, chosen_industry)
                            except Exception as exc:
                                st.error(f"Row {idx} => {exc}")
                                final_scores.append(None)
                                final_triggers.append("Prediction Failed")
                                continue
                            final_scores.append(score)
                            neg_trigs = [t for t in triggers if t in TRIGGER_DETAILS]
                            final_triggers.append(", ".join(neg_trigs) if neg_trigs else "None")

                        df_bulk["Attrition Score"] = final_scores
                        df_bulk["Negative Triggers"] = final_triggers
                        st.session_state.bulk_result = df_bulk.copy()
                        st.session_state.bulk_prediction_complete = True
                        st.session_state.bulk_result["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_user_event(st.session_state.user["email"], "bulk_prediction", {"rows":len(df_bulk)})

                with bc2:
                    if st.session_state.bulk_prediction_complete:
                        st.session_state.enable_what_if = st.checkbox("Enable What-If Analysis", key="whatif_toggle")

                if st.session_state.bulk_prediction_complete:
                    st.markdown("### Bulk Analysis")
                    flt_vals = horizontal_filters(st.session_state.bulk_result)
                    filtered_df = apply_filters(st.session_state.bulk_result, flt_vals)

                    st.write("#### Filtered Data")
                    st.dataframe(filtered_df)

                    if "Attrition Score" in filtered_df.columns and not filtered_df.empty:
                        hi = (filtered_df["Attrition Score"]>=75).sum()
                        mod = ((filtered_df["Attrition Score"]>=60)&(filtered_df["Attrition Score"]<75)).sum()
                        mid= ((filtered_df["Attrition Score"]>=35)&(filtered_df["Attrition Score"]<60)).sum()
                        lo = (filtered_df["Attrition Score"]<35).sum()
                        dist_df = pd.DataFrame({
                            "Risk Category":["High(>=75)","Mod-High(60-74)","Moderate(35-59)","Low(<35)"],
                            "Count":[hi, mod, mid, lo]
                        })
                        st.markdown("##### Risk Distribution")
                        st.bar_chart(dist_df.set_index("Risk Category"))

                    if st.session_state.enable_what_if:
                        st.markdown("#### What-If Analysis (Sliders, etc.)")
                        st.warning("Fill your custom logic here. Omitted for brevity.")
                    else:
                        st.markdown("#### Quick Charts & Custom Graph Builder")
                        ccol1, ccol2 = st.columns([0.5,0.5])
                        with ccol1:
                            st.markdown("##### Custom Graph Builder")
                            with st.form("custom_chart_form"):
                                if filtered_df.empty:
                                    st.warning("No data after filters => no charts.")
                                    xcol=None
                                    ycol=None
                                    data_label=None
                                    subchart=False
                                else:
                                    xcol = st.selectbox("X Axis", filtered_df.columns)
                                    ycol = st.selectbox("Y Axis", filtered_df.columns)
                                    data_label = st.selectbox("Data Label (color)", ["None"]+list(filtered_df.columns))
                                    subchart = st.form_submit_button("Generate Custom Chart")

                            if subchart and xcol and ycol:
                                config = {
                                    "x_axis": xcol,
                                    "y_axis": ycol,
                                    "data_label": data_label
                                }
                                st.session_state.custom_charts.insert(0, config)

                            if st.session_state.custom_charts:
                                st.markdown("### Custom Charts")
                                for i,cfg in enumerate(st.session_state.custom_charts):
                                    st.markdown(f"#### Chart {i+1}")
                                    ch = generate_custom_chart(cfg, filtered_df)
                                    st.altair_chart(ch, use_container_width=True)

                        with ccol2:
                            st.markdown("##### Additional Quick Charts or Distribution")
                            st.write("Similar to your original distribution/comparative/correlation sections...")

                            # You can replicate your expansions here if you want.

                        # Cohort Analysis
                        st.markdown("### Cohort Analysis")
                        do_cohort = st.checkbox("Enable Cohort Analysis", key="cohort_toggle")
                        if do_cohort:
                            if st.session_state.training_data is None:
                                st.error("No training data available. Train your model first.")
                            else:
                                cdf = st.session_state.training_data.copy()
                                st.markdown("#### Training Data (First 10 Rows):")
                                st.dataframe(cdf.head(10))

                                # Filter the training data
                                st.markdown("#### Filter Training Data for Cohort Analysis")
                                tflt = horizontal_filters(cdf)
                                cdf_filt = apply_filters(cdf, tflt)
                                st.write("Filtered Training Data:")
                                st.dataframe(cdf_filt)

                                # Pick column for cohorts
                                possible_cohort_cols = [
                                    c for c in cdf_filt.columns
                                    if c not in ["Name","Attrition Score","Attrition","What-If Attrition Score","What-If Negative Triggers","Prediction Time"]
                                ]
                                if possible_cohort_cols:
                                    cohort_col = st.selectbox("Column for Cohorts", possible_cohort_cols)
                                    if pd.api.types.is_numeric_dtype(cdf_filt[cohort_col]):
                                        binopt = st.checkbox("Bin numeric column?", key="bin_numeric_cohort")
                                        if binopt:
                                            bsize = st.number_input("Bin size", min_value=1, value=5)
                                            mnv = int(cdf_filt[cohort_col].min())
                                            mxv = int(cdf_filt[cohort_col].max())
                                            edges = list(range(mnv, mxv+bsize, bsize))
                                            cdf_filt["Cohort"] = pd.cut(cdf_filt[cohort_col], bins=edges)
                                        else:
                                            cdf_filt["Cohort"] = cdf_filt[cohort_col].astype(str)
                                    else:
                                        cdf_filt["Cohort"] = cdf_filt[cohort_col]

                                    metric_options = ["Count","Average Employee Age","Average Tenure (Months)",
                                                      "Average Compa Ratio","Average Last Performance Rating"]
                                    if "Attrition" in cdf_filt.columns:
                                        metric_options.append("Attrition Rate")
                                    chosen_metric = st.selectbox("Select Cohort Metric", metric_options)

                                    # aggregator
                                    if chosen_metric=="Count":
                                        grp = cdf_filt.groupby("Cohort").size().reset_index(name="Metric")
                                    elif chosen_metric=="Average Employee Age":
                                        grp = cdf_filt.groupby("Cohort")["Employee Age"].mean().reset_index(name="Metric")
                                    elif chosen_metric=="Average Tenure (Months)":
                                        grp = cdf_filt.groupby("Cohort")["Tenure (Months)"].mean().reset_index(name="Metric")
                                    elif chosen_metric=="Average Compa Ratio":
                                        grp = cdf_filt.groupby("Cohort")["Compa Ratio"].mean().reset_index(name="Metric")
                                    elif chosen_metric=="Average Last Performance Rating":
                                        grp = cdf_filt.groupby("Cohort")["Last Performance Rating"].mean().reset_index(name="Metric")
                                    elif chosen_metric=="Attrition Rate":
                                        grp = cdf_filt.groupby("Cohort")["Attrition"].mean().reset_index(name="Metric")
                                    else:
                                        st.warning("Select a valid metric.")
                                        grp = pd.DataFrame(columns=["Cohort","Metric"])

                                    st.markdown("#### Cohort Metric Visualization")
                                    if not grp.empty:
                                        viz_choice = st.selectbox("Chart Type", ["Bar","Line","Pie","Area"])
                                        if viz_choice=="Bar":
                                            cchart = alt.Chart(grp).mark_bar().encode(
                                                x=alt.X("Cohort:N", title="Cohort"),
                                                y=alt.Y("Metric:Q", title=chosen_metric),
                                                tooltip=["Cohort","Metric"]
                                            )
                                            st.altair_chart(cchart, use_container_width=True)
                                        elif viz_choice=="Line":
                                            cchart = alt.Chart(grp).mark_line(point=True).encode(
                                                x=alt.X("Cohort:N"),
                                                y=alt.Y("Metric:Q", title=chosen_metric),
                                                tooltip=["Cohort","Metric"]
                                            )
                                            st.altair_chart(cchart, use_container_width=True)
                                        elif viz_choice=="Pie":
                                            cchart = alt.Chart(grp).mark_arc().encode(
                                                theta=alt.Theta(field="Metric", type="quantitative"),
                                                color=alt.Color(field="Cohort", type="nominal"),
                                                tooltip=["Cohort","Metric"]
                                            )
                                            st.altair_chart(cchart, use_container_width=True)
                                        else:
                                            cchart = alt.Chart(grp).mark_area(opacity=0.5).encode(
                                                x=alt.X("Cohort:N"),
                                                y=alt.Y("Metric:Q", title=chosen_metric),
                                                tooltip=["Cohort","Metric"]
                                            )
                                            st.altair_chart(cchart, use_container_width=True)

                                        # EXAMPLE: Show negative triggers or pay gap per cohort
                                        st.markdown("#### Show Cohort Issues")
                                        if "Negative Triggers" in cdf_filt.columns:
                                            if st.button("Analyze Negative Triggers per Cohort"):
                                                # parse triggers for each row, group by cohort
                                                # build freq table for each cohort
                                                all_rows = []
                                                for _, rrow in cdf_filt.iterrows():
                                                    clabel = rrow["Cohort"]
                                                    trig_str = rrow.get("Negative Triggers","").strip()
                                                    if trig_str and trig_str!="None":
                                                        # parse
                                                        for t in trig_str.split(","):
                                                            t = t.strip()
                                                            if t:
                                                                all_rows.append({"Cohort":clabel,"Trigger":t})
                                                if all_rows:
                                                    cdfx = pd.DataFrame(all_rows)
                                                    freq = cdfx.groupby(["Cohort","Trigger"]).size().reset_index(name="Count")
                                                    st.write("Negative Trigger Frequencies by Cohort:")
                                                    st.dataframe(freq)
                                                else:
                                                    st.info("No triggers found in the training data for these cohorts.")

                                        if "Gender" in cdf_filt.columns and "Compa Ratio" in cdf_filt.columns:
                                            st.markdown("##### Example: Check Pay Gap (Compa Ratio) by Gender & Cohort")
                                            if st.button("Show Pay Gap by Cohort & Gender"):
                                                # group by cohort + gender => average compa
                                                paygrp = cdf_filt.groupby(["Cohort","Gender"])["Compa Ratio"].mean().reset_index(name="AvgCompa")
                                                st.dataframe(paygrp)
                                                # might chart it
                                                cchart = alt.Chart(paygrp).mark_bar().encode(
                                                    x=alt.X("Cohort:N", title="Cohort"),
                                                    y=alt.Y("AvgCompa:Q", title="Avg Compa Ratio"),
                                                    color=alt.Color("Gender:N"),
                                                    tooltip=["Cohort","Gender","AvgCompa"]
                                                )
                                                st.altair_chart(cchart, use_container_width=True)

                                    else:
                                        st.warning("No data in aggregated table.")
                                else:
                                    st.warning("No columns to define cohorts.")
