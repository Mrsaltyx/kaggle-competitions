# Introvert — Prédiction de personnalité (Kaggle Playground S5E7)

> 📦 Contenu historique importé depuis `Mrsaltyx/introvert`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S5E7-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s5e7)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition Kaggle Playground Series S5E7 : prédire le trait de personnalité (introverti / extroverti) avec un classificateur XGBoost.

## Ce que fait le projet

- Prétraitement : encodage de la cible et des variables binaires (`Stage_fear`, `Drained_after_socializing`), imputation des valeurs manquantes par la médiane.
- Entraîne un XGBoost Classifier avec validation croisée et early stopping pour déterminer le nombre optimal d'estimateurs.
- Génère trois soumissions complémentaires :
  - `submission_best.csv` — modèle optimisé par early stopping ;
  - `submission_full.csv` — modèle entraîné sur l'intégralité du train ;
  - `submission_avg.csv` — blending par moyenne des probabilités des deux modèles, pour un score plus stable.

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition (`train.csv`, `test.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/playground-s5e7-introvert-prediction

pip install pandas numpy xgboost scikit-learn jupyter
```

## Utilisation

Placer les jeux de données à la racine (ou ajuster les chemins dans le notebook), puis exécuter :

```bash
jupyter notebook test.ipynb
```

## Structure du projet

```text
introvert/
  test.ipynb  # Pipeline complet : prétraitement, XGBoost + early stopping, soumissions
```
