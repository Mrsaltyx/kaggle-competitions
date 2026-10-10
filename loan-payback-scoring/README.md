# Predicting Loan Payback — Kaggle

> 📦 Contenu historique importé depuis `Mrsaltyx/PredictingLoanPayback`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground-20BEFF.svg)](https://www.kaggle.com/competitions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour une compétition Kaggle : prédire la probabilité qu'un emprunteur rembourse son prêt, en maximisant l'AUC-ROC.

## Ce que fait le projet

- Analyse exploratoire : compréhension des variables clients (revenus, historique) et nettoyage des données manquantes ou erronées.
- Préparation intelligente : transformation et regroupement de catégories pour rendre les features plus utiles au modèle.
- Entraîne un modèle LightGBM, particulièrement adapté aux données tabulaires.
- Optimise finement les hyperparamètres pour maximiser l'AUC sur validation.
- Produit un score de probabilité entre 0 et 1 par client (ex. 0.80 = 80 % de chances de remboursement).

## Contexte

Les données sont tabulaires et générées synthétiquement (aucune information confidentielle réelle). Ce type de modèle de scoring est essentiel dans le secteur financier pour la gestion du risque de crédit.

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition (`train.csv`, `test.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/loan-payback-scoring

pip install pandas numpy scikit-learn lightgbm jupyter
```

## Utilisation

Placer `train.csv` et `test.csv` à la racine, puis exécuter le notebook :

```bash
jupyter notebook test.ipynb
```

## Structure du projet

```text
PredictingLoanPayback/
  test.ipynb  # Pipeline complet : EDA, prétraitement, LightGBM, optimisation AUC
```

## État du projet

- Modèle robuste validé sur la plateforme Kaggle.
- Perspectives : modèles plus complexes, analyse des erreurs de prédiction pour cibler les améliorations.
