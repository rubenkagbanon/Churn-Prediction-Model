"""
database.py — Gestion de la base de données SQLite pour le projet Churn Prediction
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "churn.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Créer toutes les tables si elles n'existent pas."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     TEXT UNIQUE NOT NULL,
            gender          TEXT,
            senior_citizen  INTEGER,
            partner         TEXT,
            dependents      TEXT,
            tenure          INTEGER,
            phone_service   TEXT,
            multiple_lines  TEXT,
            internet_service TEXT,
            online_security TEXT,
            tech_support    TEXT,
            contract        TEXT,
            paperless_billing TEXT,
            payment_method  TEXT,
            monthly_charges REAL,
            total_charges   REAL,
            churn           TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     TEXT,
            churn_prob      REAL,
            prediction      TEXT,
            model_version   TEXT,
            predicted_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            accuracy        REAL,
            auc_roc         REAL,
            f1_score        REAL,
            precision_score REAL,
            recall_score    REAL,
            trained_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print("[DB] Base de données initialisée.")


def insert_customers(df: pd.DataFrame):
    """Insérer des clients dans la table customers."""
    conn = get_connection()
    df.to_sql("customers", conn, if_exists="append", index=False)
    conn.close()
    print(f"[DB] {len(df)} clients insérés.")


def save_prediction(customer_id: str, churn_prob: float, prediction: str, model_version: str = "v1.0"):
    """Sauvegarder une prédiction."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions (customer_id, churn_prob, prediction, model_version, predicted_at)
           VALUES (?, ?, ?, ?, ?)""",
        (customer_id, churn_prob, prediction, model_version, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def save_metrics(model_name, accuracy, auc_roc, f1, precision, recall):
    """Enregistrer les métriques d'un modèle."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO model_metrics (model_name, accuracy, auc_roc, f1_score, precision_score, recall_score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_name, accuracy, auc_roc, f1, precision, recall)
    )
    conn.commit()
    conn.close()


def get_predictions_history() -> pd.DataFrame:
    """Récupérer l'historique des prédictions."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY predicted_at DESC", conn)
    conn.close()
    return df


def get_model_metrics() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM model_metrics ORDER BY trained_at DESC", conn)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
