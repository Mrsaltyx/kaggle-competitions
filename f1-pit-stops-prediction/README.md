# Predicting F1 Pit Stops — Kaggle Tabular Playground

> 📦 Contenu historique importé depuis `Mrsaltyx/Predicting-F1-Pit-Stops`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Tabular%20Playground-20BEFF.svg)](https://www.kaggle.com/competitions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Pipeline de Machine Learning pour prédire les arrêts aux stands en Formule 1, sur des données synthétiques à haute fidélité imitant les comportements réels des écuries.

## Ce que fait le projet

- Feature engineering métier : variables liées à l'usure des pneumatiques, à la dégradation de l'essence et aux deltas de performance par tour.
- Compare les modèles de Gradient Boosting de référence sur données tabulaires : LightGBM, XGBoost et CatBoost.
- Recherche d'hyperparamètres par optimisation bayésienne (Optuna).
- Validation robuste par Cross-Validation stratifiée (Stratified K-Fold) pour éviter le sur-apprentissage.
- Ensembling par stacking ou blending pour maximiser le score final.

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/f1-pit-stops-prediction

pip install pandas numpy scikit-learn lightgbm xgboost optuna jupyter
```

## Utilisation

Les analyses et prototypes sont dans le répertoire `Predicting f1/`. Ouvrir les notebooks avec Jupyter :

```bash
jupyter notebook
```

## Structure du projet

```text
Predicting-F1-Pit-Stops/
  Predicting f1/  # Notebooks : EDA, feature engineering, modélisation, ensembling
```

## État du projet

- Pipeline de Gradient Boosting fonctionnel, optimisation Optuna en cours d'affinage.
