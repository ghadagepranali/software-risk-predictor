import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config(page_title="Software Risk Predictor", page_icon="🚀")
st.title("🚀 Software Project Risk Predictor")
st.write("Explainable AI System - Tumcha project cha Risk check kara")

@st.cache_resource
def load_model():
    df = pd.read_csv("dataset/software_projects.csv")
    X = df.drop("risk_level", axis=1)
    y = df["risk_level"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X, y_enc)
    return model, le

model, le = load_model()

st.sidebar.header("Project Details Taka")
team_size = st.sidebar.slider("Team Size", 2, 15, 5)
bug_count = st.sidebar.slider("Bug Count", 0, 50, 10)
code_complexity = st.sidebar.slider("Code Complexity (1-10)", 1, 10, 5)
req_changes = st.sidebar.slider("Requirement Changes", 0, 20, 3)
deadline_pressure = st.sidebar.slider("Deadline Pressure (1-10)", 1, 10, 5)

if st.button("Risk Predict Kara"):
    input_data = pd.DataFrame([[team_size, bug_count, code_complexity, req_changes, deadline_pressure]],
                              columns=["team_size", "bug_count", "code_complexity", "req_changes", "deadline_pressure"])
    pred = model.predict(input_data.values)
    risk = le.inverse_transform(pred)[0]

    if risk == "High":
        st.error(f"🔴 Result: {risk} Risk - Sambhalun kaam kara!")
    elif risk == "Medium":
        st.warning(f"🟡 Result: {risk} Risk - Lakshya dya")
    else:
        st.success(f"🟢 Result: {risk} Risk - Project Safe aahe!")

st.divider()
st.write("Graphs:")
col1, col2 = st.columns(2)
with col1:
    st.image("risk_distribution.png", caption="Risk Distribution")
with col2:
    st.image("shap_summary.png", caption="SHAP - Why Risk?")