"""
train_model.py
Trains and packages a model + metadata artifact.
Run first: python src/train_model.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_DIR = "model"
MODEL_VERSION = "1.0.0"

FEATURE_NAMES = [
    "income",
    "age",
    "credit_score",
    "loan_amount",
    "employment_years",
]


def build_dataset(n_samples: int = 2000, seed: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    df["income"] = (df["income"] * 20000 + 60000).clip(10000, 300000)
    df["age"] = (df["age"] * 10 + 40).clip(18, 90).round()
    df["credit_score"] = (df["credit_score"] * 100 + 650).clip(300, 850).round()
    df["loan_amount"] = (df["loan_amount"] * 15000 + 25000).clip(1000, 200000)
    df["employment_years"] = (df["employment_years"] * 5 + 6).clip(0, 40).round()

    return df, y


def main():
    df, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC-AUC : {auc:.4f}")

    feature_schema = {
        "income": {"dtype": "float", "min": 10000, "max": 300000},
        "age": {"dtype": "int", "min": 18, "max": 90},
        "credit_score": {"dtype": "int", "min": 300, "max": 850},
        "loan_amount": {"dtype": "float", "min": 1000, "max": 200000},
        "employment_years": {"dtype": "int", "min": 0, "max": 40},
    }

    metadata = {
        "model_name": "loan_approval_classifier",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn",
        "algorithm": "RandomForestClassifier",
        "feature_order": FEATURE_NAMES,
        "feature_schema": feature_schema,
        "metrics": {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)},
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/model_{MODEL_VERSION}.joblib")
    with open(f"{MODEL_DIR}/metadata_{MODEL_VERSION}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(f"{MODEL_DIR}/latest.json", "w") as f:
        json.dump({"latest_version": MODEL_VERSION}, f, indent=2)

    print(f"Packaged model -> {MODEL_DIR}/model_{MODEL_VERSION}.joblib")
    print(f"Packaged metadata -> {MODEL_DIR}/metadata_{MODEL_VERSION}.json")


if __name__ == "__main__":
    main()