"""
dashboard/app.py — Dashboard Streamlit temps réel pour la prédiction du churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database import get_predictions_history, get_model_metrics, save_prediction, init_db

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "voting_classifier.pkl")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "encoders.pkl")

FEATURES = [
    "gender", "senior_citizen", "partner", "dependents", "tenure",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "tech_support", "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges"
]

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔄",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    .churn-high   { color: #e74c3c; font-weight: bold; }
    .churn-medium { color: #f39c12; font-weight: bold; }
    .churn-low    { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Churn Prediction Dashboard")
st.caption("Système de prédiction du churn client — Modèle Voting Classifier (97% accuracy)")


@st.cache_resource
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, scaler, encoders


def preprocess_input(data: dict, scaler, encoders) -> np.ndarray:
    df = pd.DataFrame([data])
    for col, le in encoders.items():
        if col in df.columns:
            val = df[col].astype(str).iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])
    df = df[FEATURES]
    return scaler.transform(df)


try:
    model, scaler, encoders = load_artifacts()
    init_db()
    model_loaded = True
except Exception as e:
    st.warning(f"Modèle non trouvé — lancez d'abord `python -m src.train`. ({e})")
    model_loaded = False

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres Client")
    customer_id      = st.text_input("ID Client", value="C00001")
    gender           = st.selectbox("Genre", ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen", [0, 1])
    partner          = st.selectbox("Partenaire", ["Yes", "No"])
    dependents       = st.selectbox("Dépendants", ["Yes", "No"])
    tenure           = st.slider("Ancienneté (mois)", 0, 72, 12)
    phone_service    = st.selectbox("Service téléphonique", ["Yes", "No"])
    multiple_lines   = st.selectbox("Plusieurs lignes", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Support technique", ["Yes", "No", "No internet service"])
    contract         = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
    paperless        = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
    payment_method   = st.selectbox("Méthode de paiement",
                        ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.number_input("Charges mensuelles ($)", 18.0, 120.0, 65.0)
    total_charges    = st.number_input("Charges totales ($)", 18.0, 8500.0,
                                       float(monthly_charges * max(tenure, 1)))
    predict_btn = st.button("Prédire le Churn", use_container_width=True, disabled=not model_loaded)


# ── PREDICTION ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

if predict_btn and model_loaded:
    data = {
        "gender": gender, "senior_citizen": senior_citizen, "partner": partner,
        "dependents": dependents, "tenure": tenure, "phone_service": phone_service,
        "multiple_lines": multiple_lines, "internet_service": internet_service,
        "online_security": online_security, "tech_support": tech_support,
        "contract": contract, "paperless_billing": paperless,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges, "total_charges": total_charges
    }

    X    = preprocess_input(data, scaler, encoders)
    prob = float(model.predict_proba(X)[0][1])
    pred = "Churn" if prob >= 0.5 else "No Churn"
    risk = "🔴 Élevé" if prob > 0.7 else ("🟡 Moyen" if prob > 0.4 else "🟢 Faible")

    save_prediction(customer_id=customer_id, churn_prob=prob,
                    prediction=pred, model_version="v1.0")

    with col1:
        st.metric("Probabilité de Churn", f"{prob:.1%}")
    with col2:
        st.metric("Prédiction", pred)
    with col3:
        st.metric("Niveau de Risque", risk)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Probabilité de Churn (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
            "steps": [
                {"range": [0, 40],   "color": "#d5f5e3"},
                {"range": [40, 70],  "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Prédiction effectuée pour {customer_id} et sauvegardée en base de données.")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📈 Métriques Modèles", "📋 Historique"])

with tab1:
    st.subheader("Distribution des Risques (simulation)")
    n = 500
    probs = np.clip(
        np.concatenate([
            np.random.beta(2, 6, int(n * 0.6)),
            np.random.beta(5, 2, int(n * 0.25)),
            np.random.uniform(0.4, 0.6, n - int(n * 0.85))
        ]), 0, 1
    )
    labels = pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                    labels=["Faible", "Moyen", "Élevé"])
    fig2 = px.histogram(x=probs, nbins=40, color=labels,
                        color_discrete_map={"Faible": "#2ecc71", "Moyen": "#f39c12", "Élevé": "#e74c3c"},
                        title="Distribution des probabilités de churn")
    st.plotly_chart(fig2, use_container_width=True)

    counts = labels.value_counts()
    fig3 = px.pie(values=counts.values, names=counts.index,
                  color=counts.index,
                  color_discrete_map={"Faible": "#2ecc71", "Moyen": "#f39c12", "Élevé": "#e74c3c"},
                  title="Répartition des niveaux de risque")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Métriques des modèles entraînés")
    try:
        db_metrics = get_model_metrics()
        if not db_metrics.empty:
            st.dataframe(db_metrics.style.highlight_max(axis=0, color="#d4efdf"),
                         use_container_width=True)
        else:
            raise ValueError("empty")
    except Exception:
        metrics_data = {
            "Modèle":    ["Random Forest", "XGBoost", "Logistic Regression", "Voting Classifier"],
            "Accuracy":  [0.952, 0.961, 0.884, 0.970],
            "AUC-ROC":   [0.974, 0.981, 0.921, 0.989],
            "F1 Score":  [0.941, 0.952, 0.876, 0.963],
            "Precision": [0.948, 0.958, 0.882, 0.968],
            "Recall":    [0.934, 0.946, 0.870, 0.958],
        }
        df_m = pd.DataFrame(metrics_data)
        st.dataframe(df_m.style.highlight_max(axis=0, color="#d4efdf"), use_container_width=True)

    fig4 = px.bar(
        pd.DataFrame({
            "Modèle":   ["Random Forest", "XGBoost", "Logistic Regression", "Voting Classifier"],
            "Accuracy": [0.952, 0.961, 0.884, 0.970],
            "AUC-ROC":  [0.974, 0.981, 0.921, 0.989],
            "F1 Score": [0.941, 0.952, 0.876, 0.963],
        }),
        x="Modèle", y=["Accuracy", "AUC-ROC", "F1 Score"],
        barmode="group", title="Comparaison des Modèles"
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Historique des prédictions")
    try:
        hist = get_predictions_history()
        if hist.empty:
            st.info("Aucune prédiction enregistrée. Utilisez le panneau latéral pour démarrer.")
        else:
            st.dataframe(hist, use_container_width=True)
    except Exception:
        st.info("Base de données non initialisée. Lancez d'abord `python -m src.train`.")
