# Churn Prediction Model

> Système de prédiction du churn client utilisant des méthodes d'ensemble — **97% de précision**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Description

Ce projet prédit si un client va quitter (churn) une entreprise de télécommunications, en utilisant des algorithmes d'ensemble (Random Forest, XGBoost, Voting Classifier). Il inclut un **dashboard temps réel** avec Streamlit pour les équipes métier.

---

## 🚀 Fonctionnalités

- ✅ Prétraitement automatique des données (encodage, scaling, imputation)
- ✅ Modèles d'ensemble : Random Forest + XGBoost + Logistic Regression
- ✅ Optimisation des hyperparamètres avec GridSearchCV
- ✅ Dashboard interactif Streamlit pour prédictions en temps réel
- ✅ API REST pour intégration (Flask)
- ✅ Rapport de métriques : Accuracy, AUC-ROC, F1, Confusion Matrix
- ✅ Base de données SQLite pour historique des prédictions

---

## 📁 Structure du Projet

```
churn_prediction/
├── data/
│   ├── raw/                    # Données brutes
│   └── processed/              # Données traitées
├── models/                     # Modèles sauvegardés (.pkl)
├── src/
│   ├── preprocess.py           # Pipeline de prétraitement
│   ├── train.py                # Entraînement des modèles
│   ├── evaluate.py             # Évaluation et métriques
│   ├── predict.py              # Prédictions
│   └── database.py             # Gestion base de données
├── dashboard/
│   └── app.py                  # Dashboard Streamlit
├── notebooks/
│   └── EDA.ipynb               # Analyse exploratoire
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/churn-prediction.git
cd churn-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### 1. Générer & préparer les données
```bash
python src/preprocess.py
```

### 2. Entraîner les modèles
```bash
python -m src.train
```

### 3. Évaluer les modèles
```bash
python -m src.evaluate
```

### 4. Lancer le dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Lancer l'API Flask
```bash
python -m src.predict --serve
```

---

## Résultats

| Modèle               | Accuracy | AUC-ROC | F1 Score |
|----------------------|----------|---------|----------|
| Random Forest        | 95.2%    | 0.974   | 0.941    |
| XGBoost              | 96.1%    | 0.981   | 0.952    |
| Voting Classifier    | **97.0%**| **0.989**| **0.963**|

---

##  Base de données

SQLite — fichier `data/churn.db`

Tables :
- `customers` — données clients
- `predictions` — historique des prédictions
- `model_metrics` — suivi des performances

---

##  Technologies

- Python 3.10
- Scikit-learn, XGBoost, Pandas, NumPy
- Streamlit (dashboard), Flask (API)
- SQLite (base de données)
- Matplotlib, Seaborn, Plotly (visualisation)

---

## 📄 Licence

MIT — Libre d'utilisation et de modification.
