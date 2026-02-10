# 04_Model_Evaluation.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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
# 3. Train-test split (same as training)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------
MODEL_PATH = "models/XGBoost.pkl"
model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 6. Metrics
# --------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation Metrics (XGBoost)")
print("----------------------------------------")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc:.4f}")


# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.show()
