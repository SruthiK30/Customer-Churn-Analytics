import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

# --- Paths ---
DATA_DIR = r"C:\Users\Sruthi K\OneDrive\Desktop\Customer-Churn-Analytics\data"
MODEL_DIR = r"C:\Users\Sruthi K\OneDrive\Desktop\Customer-Churn-Analytics\model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load processed data ---
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print("Data loaded:", X_train.shape, X_test.shape)

# ============================================================
# MODEL 1: Logistic Regression (baseline)
# ============================================================
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# ============================================================
# MODEL 2: Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'  # handles churn imbalance
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# ============================================================
# Compare + pick best model (by ROC-AUC)
# ============================================================
auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_rf = roc_auc_score(y_test, y_proba_rf)

if auc_rf >= auc_lr:
    best_model = rf
    best_name = "Random Forest"
else:
    best_model = log_reg
    best_name = "Logistic Regression"

print(f"\n🏆 Best model: {best_name} (ROC-AUC: {max(auc_lr, auc_rf):.4f})")

# ============================================================
# Feature importance (Random Forest only)
# ============================================================
if best_name == "Random Forest":
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 features driving churn:")
    print(importances.head(10))

# ============================================================
# Save the best model
# ============================================================
joblib.dump(best_model, os.path.join(MODEL_DIR, "churn_model.pkl"))
joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))
print(f"\n✅ Saved {best_name} to model/churn_model.pkl")

# ============================================================
# THRESHOLD TUNING for Random Forest
# ============================================================
print("\n=== Threshold Tuning (Random Forest) ===")
thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

for t in thresholds:
    y_pred_t = (y_proba_rf >= t).astype(int)
    prec = precision_score(y_test, y_pred_t)
    rec = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    print(f"Threshold {t}: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

# ============================================================
# MODEL 2b: Random Forest (tuned)
# ============================================================
rf_tuned = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42,
    class_weight='balanced'
)
rf_tuned.fit(X_train, y_train)

y_pred_rf2 = rf_tuned.predict(X_test)
y_proba_rf2 = rf_tuned.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (Tuned) ===")
print(classification_report(y_test, y_pred_rf2))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf2))