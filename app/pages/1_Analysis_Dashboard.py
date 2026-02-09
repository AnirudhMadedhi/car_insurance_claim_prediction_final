import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analysis Dashboard", layout="wide")

st.title("📊 Claim Risk Analysis Dashboard")

# Load model
model = joblib.load("models/XGBoost.pkl")

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.drop(columns=["policy_id"], errors="ignore")

    # Encode categoricals (same logic as app.py)
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    probs = model.predict_proba(X)[:, 1]

    dashboard_df = pd.DataFrame({
        "Claim Probability": probs
    })

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.25, 0.05)
    dashboard_df["Prediction"] = (dashboard_df["Claim Probability"] >= threshold).astype(int)

    # ---------------- KPIs ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Policies", len(dashboard_df))
    col2.metric("Predicted Claims", dashboard_df["Prediction"].sum())
    col3.metric("Avg Claim Probability", round(dashboard_df["Claim Probability"].mean(), 3))

    st.divider()

    # ---------------- Probability Distribution ----------------
    st.subheader("Claim Probability Distribution")

    fig, ax = plt.subplots()
    ax.hist(dashboard_df["Claim Probability"], bins=50)
    ax.set_xlabel("Claim Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ---------------- Risk Segmentation ----------------
    st.subheader("Risk Segmentation")

    def risk_bucket(p):
        if p < 0.2:
            return "Low"
        elif p < 0.5:
            return "Medium"
        else:
            return "High"

    dashboard_df["Risk Level"] = dashboard_df["Claim Probability"].apply(risk_bucket)
    st.bar_chart(dashboard_df["Risk Level"].value_counts())

else:
    st.info("Please upload a CSV file to view analysis.")
