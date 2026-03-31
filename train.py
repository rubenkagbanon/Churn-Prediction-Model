"""
train.py — Entraînement des modèles d'ensemble pour la prédiction du churn
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.database import init_db, save_metrics

PROC_PATH  = os.path.join("data", "processed")
MODEL_PATH = "models"


def load_processed():
    X_train = pd.read_csv(os.path.join(PROC_PATH, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROC_PATH, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC_PATH, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROC_PATH, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def apply_smote(X, y):
    """Rééquilibrer les classes avec SMOTE."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[SMOTE] Distribution après rééquilibrage : {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def build_models():
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
        voting="soft",
        weights=[2, 3, 1],
    )
    return {"random_forest": rf, "xgboost": xgb, "logistic_regression": lr, "voting_classifier": voting}


def train():
    init_db()
    X_train, X_test, y_train, y_test = load_processed()
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    models = build_models()
    os.makedirs(MODEL_PATH, exist_ok=True)

    for name, model in models.items():
        print(f"\n[TRAIN] Entraînement : {name} …")
        model.fit(X_train_res, y_train_res)

        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring="accuracy")
        print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        model_file = os.path.join(MODEL_PATH, f"{name}.pkl")
        joblib.dump(model, model_file)
        print(f"  Modèle sauvegardé → {model_file}")

    print("\n[TRAIN] Tous les modèles entraînés avec succès.")
    return models, X_test, y_test


if __name__ == "__main__":
    # Lancer le prétraitement si nécessaire
    if not os.path.exists(os.path.join(PROC_PATH, "X_train.csv")):
        from src.preprocess import load_data, preprocess
        df = load_data()
        preprocess(df)
    train()