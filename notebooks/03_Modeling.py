# 03_Model_Training.py

import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    label_encoders[col] = le


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
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# --------------------------------------------------
# 6. Define baseline models (unchanged)
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
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 7. XGBoost with RandomizedSearchCV
# --------------------------------------------------
xgb = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# --------------------------------------------------
# 8. Train, evaluate, save models
# --------------------------------------------------
results = {}

print("\nStarting model training...\n")

# ---- Train baseline models ----
for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
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


# ---- Train XGBoost with RandomizedSearchCV ----
print("Training XGBoost with RandomizedSearchCV...")

xgb_random.fit(X_train, y_train)

best_xgb = xgb_random.best_estimator_

y_prob = best_xgb.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

results["XGBoost"] = {
    "ROC_AUC": roc,
    "F1_Score": f1
}

model_path = os.path.join(MODELS_DIR, "XGBoost.pkl")
joblib.dump(best_xgb, model_path)

print(f"XGBoost saved to {model_path}")
print(f"ROC-AUC: {roc:.4f} | F1-Score: {f1:.4f}")
print("Best XGBoost Params:", xgb_random.best_params_)


# --------------------------------------------------
# 9. Final comparison
# --------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nModel Comparison:")
print(results_df)

print("\nTraining complete. Models saved successfully.")
