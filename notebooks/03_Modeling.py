# 03_Model_Training.py

import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
DATA_PATH = "data/train.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_claim", "policy_id"])
y = df["is_claim"]


# --------------------------------------------------
# 2. Encode categorical variables
# --------------------------------------------------
label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # stored if needed later


# --------------------------------------------------
# 3. Train-test split (stratified)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create models directory
# --------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------------------------
# 5. Handle class imbalance
# --------------------------------------------------
# Ratio for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# --------------------------------------------------
# 6. Define models (IMBALANCE FIXED)
# --------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),

    "DecisionTree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",  # 🔥 KEY FIX
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,  # 🔥 KEY FIX
        random_state=42
    )
}


# --------------------------------------------------
# 7. Train, evaluate, save models
# --------------------------------------------------
results = {}

print("\nStarting model training...\n")

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    # --- Probabilities (IMPORTANT) ---
    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold (app will tune this later)
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "ROC_AUC": roc,
        "F1_Score": f1
    }

    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    print(f"{name} saved to {model_path}")
    print(f"ROC-AUC: {roc:.4f} | F1-Score: {f1:.4f}\n")


# --------------------------------------------------
# 8. Final comparison
# --------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nModel Comparison:")
print(results_df)

print("\nTraining complete. Models saved successfully.")
