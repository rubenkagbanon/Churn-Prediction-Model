"""
evaluate.py — Évaluation complète des modèles de churn
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report, roc_curve
)
from src.database import save_metrics

PROC_PATH  = os.path.join("data", "processed")
MODEL_PATH = "models"
REPORT_DIR = os.path.join("data", "reports")


def load_test_data():
    X_test = pd.read_csv(os.path.join(PROC_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROC_PATH, "y_test.csv")).squeeze()
    return X_test, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]
    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_proba)
    f1          = f1_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred)
    rec         = recall_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    save_metrics(name, acc, auc, f1, prec, rec)
    return {"name": name, "accuracy": acc, "auc": auc, "f1": f1,
            "precision": prec, "recall": rec, "y_pred": y_pred, "y_proba": y_proba}


def plot_results(results, y_test):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Confusion Matrix ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, r in enumerate(results):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        axes[i].set_title(f"{r['name']} — Acc: {r['accuracy']:.2%}")
    plt.suptitle("Matrices de Confusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrices.png"), dpi=120)
    plt.close()

    # --- ROC Curves ---
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbes ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=120)
    plt.close()

    # --- Metrics Comparison Bar Chart ---
    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if k in ["name","accuracy","auc","f1","precision","recall"]}
        for r in results
    ]).set_index("name")
    metrics_df.plot(kind="bar", figsize=(10, 5), ylim=(0.85, 1.0), colormap="viridis")
    plt.title("Comparaison des Métriques")
    plt.ylabel("Score")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "metrics_comparison.png"), dpi=120)
    plt.close()

    print(f"\n[INFO] Graphiques sauvegardés dans {REPORT_DIR}/")


def evaluate_all():
    X_test, y_test = load_test_data()
    results = []
    model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pkl")
                and "encoder" not in f and "scaler" not in f]
    for mf in model_files:
        name  = mf.replace(".pkl", "")
        model = joblib.load(os.path.join(MODEL_PATH, mf))
        r     = evaluate_model(name, model, X_test, y_test)
        results.append(r)

    if results:
        plot_results(results, y_test)
    return results


if __name__ == "__main__":
    evaluate_all()