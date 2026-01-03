# 03_Model_Training.py

import os
import pandas as pd
import joblib

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
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# --------------------------------------------------
# 3. Train-test split
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
# 5. Define models
# --------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}


# --------------------------------------------------
# 6. Train, evaluate, save models
# --------------------------------------------------
results = {}

print("\nStarting model training...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    roc = roc_auc_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results[name] = {
        "ROC_AUC": roc,
        "F1_Score": f1
    }
    
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    
    print(f"{name} saved to {model_path}")
    print(f"ROC-AUC: {roc:.4f} | F1: {f1:.4f}\n")


# --------------------------------------------------
# 7. Final comparison
# --------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nModel Comparison:")
print(results_df)

print("\nTraining complete. Models saved successfully.")
