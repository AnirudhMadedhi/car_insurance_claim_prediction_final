import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Feature Importance", layout="wide")

st.title("🔍 Top Predictive Features")

# Load model
model = joblib.load("models/XGBoost.pkl")

# Load training data (to get feature names)
train_df = pd.read_csv("data/train.csv")
X = train_df.drop(columns=["is_claim", "policy_id"])

# Encode categoricals same as training
from sklearn.preprocessing import LabelEncoder
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_n = st.slider("Select number of top features", 5, 30, 15)

st.dataframe(importance_df.head(top_n), use_container_width=True)

# Plot
st.subheader("Feature Importance Plot")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    importance_df.head(top_n)["Feature"][::-1],
    importance_df.head(top_n)["Importance"][::-1]
)
ax.set_xlabel("Importance Score")
st.pyplot(fig)
