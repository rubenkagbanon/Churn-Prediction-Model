"""
preprocess.py — Pipeline de prétraitement des données Churn
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RAW_DATA_PATH  = os.path.join("data", "raw", "telco_churn.csv")
PROC_DATA_PATH = os.path.join("data", "processed")
ENCODER_PATH   = os.path.join("models", "encoders.pkl")
SCALER_PATH    = os.path.join("models", "scaler.pkl")


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """Générer un dataset synthétique Telco-Churn si les données réelles sont absentes."""
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id":       [f"C{i:05d}" for i in range(n)],
        "gender":            np.random.choice(["Male", "Female"], n),
        "senior_citizen":    np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "partner":           np.random.choice(["Yes", "No"], n),
        "dependents":        np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure":            np.random.randint(0, 72, n),
        "phone_service":     np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "multiple_lines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "internet_service":  np.random.choice(["DSL", "Fiber optic", "No"], n),
        "online_security":   np.random.choice(["Yes", "No", "No internet service"], n),
        "tech_support":      np.random.choice(["Yes", "No", "No internet service"], n),
        "contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
        "paperless_billing": np.random.choice(["Yes", "No"], n),
        "payment_method":    np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "monthly_charges":   np.round(np.random.uniform(18, 118, n), 2),
        "total_charges":     np.round(np.random.uniform(18, 8500, n), 2),
    })
    # Logique de churn réaliste
    churn_prob = (
        0.05
        + 0.30 * (df["contract"] == "Month-to-month").astype(int)
        + 0.15 * (df["internet_service"] == "Fiber optic").astype(int)
        - 0.10 * (df["tenure"] > 36).astype(int)
        + 0.05 * np.random.normal(0, 1, n)
    )
    churn_prob = churn_prob.clip(0, 1)
    df["churn"] = np.where(np.random.uniform(0, 1, n) < churn_prob, "Yes", "No")
    return df


def load_data() -> pd.DataFrame:
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        print("[INFO] Données réelles non trouvées → génération synthétique.")
        df = generate_synthetic_data()
        df.to_csv(RAW_DATA_PATH, index=False)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Nettoyage
    df.drop(columns=["customer_id"], errors="ignore", inplace=True)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encodage des variables catégorielles
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "churn"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Target
    df["churn"] = (df["churn"] == "Yes").astype(int)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Sauvegarde
    os.makedirs(PROC_DATA_PATH, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler,   SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    X_train.to_csv(os.path.join(PROC_DATA_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROC_DATA_PATH, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(PROC_DATA_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROC_DATA_PATH, "y_test.csv"),  index=False)

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    print("[INFO] Prétraitement terminé.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    preprocess(df)