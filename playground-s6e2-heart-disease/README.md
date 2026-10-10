# Predicting Heart Disease — Kaggle Playground S6E2

> 📦 Contenu historique importé depuis `Mrsaltyx/Predicting_Heart_Disease`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E2-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Public LB](https://img.shields.io/badge/rank-839%20%2F%20top%201000-brightgreen.svg)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition Kaggle Playground Series S6E2 : prédire la présence de maladies cardiaques en maximisant l'AUC-ROC.

## Ce que fait le projet

- Feature engineering physiologique : `HR_diff` (écart fréquence cardiaque théorique / observée), `BP_Age_ratio` (tension / âge), `High_Risk_Blood` (cholestérol > 240 et hypertension > 130).
- Entraîne le « trio de fer » du boosting tabulaire : LightGBM, XGBoost et CatBoost.
- Optimise les hyperparamètres de chaque modèle avec Optuna (recherche bayésienne, 50 trials, Stratified 5-Fold CV).
- Optimise les poids du blending des trois modèles via Optuna sur les prédictions OOF (Out-of-Fold).

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition (`train.csv`, `test.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/playground-s6e2-heart-disease

pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

## Utilisation

Placer `train.csv` et `test.csv` à la racine, puis lancer le pipeline complet (HPO + blending) :

```bash
python test_optuna.py
```

Le script `test.py` contient une version de base sans optimisation bayésienne.

## Structure du projet

```text
Predicting_Heart_Disease/
  test.py         # Baseline : trio de boosting, paramètres standards
  test_optuna.py  # Pipeline complet : HPO Optuna + blending pondéré OOF
```

## État du projet

- **Top 1000 (rang 839)** sur le leaderboard public.
- Perspectives : stacking avec meta-learner, analyse des outliers du cholestérol, target encoding des variables catégorielles.
