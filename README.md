# Kaggle Competitions — Mrsaltyx

[![Kaggle](https://img.shields.io/badge/Kaggle-Profil-20BEFF.svg)](https://www.kaggle.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Monorepo regroupant mes participations aux compétitions [Kaggle](https://www.kaggle.com/competitions) : classification tabulaire, NLP, computer vision et séries temporelles. Chaque dossier est un projet autonome avec son README, ses notebooks et son historique de scores.

## Compétitions

| Compétition | Dossier | Type | Métrique | Résultat |
|---|---|---|---|---|
| Playground S6E8 — Smartphone Addiction | [`playground-s6e8-smartphone-addiction/`](playground-s6e8-smartphone-addiction/) | Tabulaire binaire | ROC AUC | **Top 3,5 % — #121 / 3532 (private 0.97104)** |
| Playground S6E7 — Student Health Risk | [`playground-s6e7-student-health-risk/`](playground-s6e7-student-health-risk/) | Tabulaire déséquilibré | Balanced accuracy | **0.95307** (public) |
| Playground S6E2 — Heart Disease | [`playground-s6e2-heart-disease/`](playground-s6e2-heart-disease/) | Tabulaire binaire | AUC-ROC | Top 1000 (rang 839) |
| Playground S5E7 — Introvert Prediction | [`playground-s5e7-introvert-prediction/`](playground-s5e7-introvert-prediction/) | Tabulaire binaire | AUC-ROC | Blending moyenne / early stopping |
| F1 Pit Stops | [`f1-pit-stops-prediction/`](f1-pit-stops-prediction/) | Tabulaire / feature engineering métier | — | Trio LGBM / XGB / CatBoost + Optuna |
| Loan Payback Scoring | [`loan-payback-scoring/`](loan-payback-scoring/) | Scoring crédit | AUC-ROC | LightGBM optimisé |
| NLP Disaster Tweets | [`nlp-disaster-tweets/`](nlp-disaster-tweets/) | NLP binaire | F1-score | TF-IDF / GloVe + XGBoost |
| CIFAR-10 | [`cifar-10-image-classification/`](cifar-10-image-classification/) | Vision / CNN | Accuracy | CNN + augmentation + early stopping |
| Titanic | [`titanic/`](titanic/) | Intro ML | Accuracy | EDA + feature engineering + comparaison de modèles |

## Méthodologie

Principes appliqués sur les compétitions récentes (S6E7 → S6E8), formalisés dans le `PLAYBOOK.md` du dossier S6E7 :

1. **Baseline honnête précoce** — un modèle simple soumis dès le jour 1 pour ancrer le leaderboard.
2. **Journal d'expérimentation** — chaque soumission documentée avec approche, score et verdict mesuré.
3. **Sélection finale au meilleur OOF honnête**, pas au meilleur public LB — sur S6E8, cette doctrine a évité tout shakeup (−0.00025 public → private).
4. **La métrique est tout** — correction de prior calibrée sur OOF (S6E7, +0.06 en balanced accuracy), folding fold-safe massif sur les méta-ensembles (S6E8).

## Stack technique récurrente

LightGBM / XGBoost (CUDA) / CatBoost · Optuna (TPE) · scikit-learn · pandas / numpy · PyTorch / TensorFlow (vision, NLP) · stratified K-Fold OOF · blending & vote pondéré.

## Structure

```
kaggle-competitions/
├── playground-s6e8-smartphone-addiction/   # top 3,5 % — kernels v1→v4, méta-stack, journal
├── playground-s6e7-student-health-risk/    # PLAYBOOK.md, 20+ soumissions, correction de prior
├── playground-s6e2-heart-disease/          # feature engineering physiologique + Optuna
├── playground-s5e7-introvert-prediction/   # XGBoost + early stopping + blending
├── f1-pit-stops-prediction/                # feature engineering F1, stacking
├── loan-payback-scoring/                   # scoring crédit LightGBM
├── nlp-disaster-tweets/                    # TF-IDF / GloVe, Naive Bayes → XGBoost
├── cifar-10-image-classification/          # CNN Keras, augmentation de données
└── titanic/                                # EDA, feature engineering, comparaison de modèles
```

## Remarque

Ces dossiers ont été importés depuis des dépôts individuels historiques (fusionnés ici en octobre 2026). Les jeux de données Kaggle (`train.csv`, `test.csv`, etc.) ne sont pas versionnés — les télécharger depuis la page de chaque compétition.
