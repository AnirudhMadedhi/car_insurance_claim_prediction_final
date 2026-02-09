import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Insurance Claim Prediction", layout="centered")

# --------------------------------------------------
# Load model
# --------------------------------------------------
MODEL_PATH = "models/XGBoost.pkl"
model = joblib.load(MODEL_PATH)

st.title("🚗 Car Insurance Claim Prediction")
st.write(
    "This app predicts whether a customer is likely to file an insurance claim "
    "based on vehicle, policy, and demographic details."
)

# --------------------------------------------------
# Upload input CSV
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file (same structure as training data, without `is_claim`)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Drop policy_id if present
    if "policy_id" in df.columns:
        df = df.drop(columns=["policy_id"])

    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result_df = pd.DataFrame({
        "Prediction (0 = No Claim, 1 = Claim)": predictions,
        "Claim Probability": probabilities
    })

    st.subheader("Prediction Results")
    st.dataframe(result_df)

    st.success("Prediction completed successfully!")

else:
    st.info("Please upload a CSV file to get predictions.")
