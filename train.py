import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/telecom_churn.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ============================================================
# SELECT IMPORTANT FEATURES
# ============================================================

selected_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport"
]

X = df[selected_features]
y = df["Churn"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ]
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

# ============================================================
# PREDICT
# ============================================================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

# ============================================================
# EVALUATION
# ============================================================

print("=" * 60)

print("Classification Report\n")

print(classification_report(y_test, predictions))

print("=" * 60)

accuracy = accuracy_score(y_test, predictions)

roc = roc_auc_score(y_test, probabilities)

print(f"Accuracy : {accuracy:.4f}")

print(f"ROC AUC  : {roc:.4f}")

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

feature_importance = pd.Series(
    pipeline.named_steps[
        "classifier"
    ].feature_importances_,
    index=feature_names
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\nTop 10 Important Features\n")

print(feature_importance.head(10))

# ============================================================
# SAVE PIPELINE
# ============================================================

joblib.dump(
    pipeline,
    os.path.join(
        MODEL_DIR,
        "churn_pipeline.pkl"
    )
)

print("\nModel Saved Successfully!")

print("Location : model/churn_pipeline.pkl")