# Titanic — Machine Learning from Disaster

> 📦 Contenu historique importé depuis `Mrsaltyx/Titanic`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition Kaggle « Titanic - Machine Learning from Disaster » : prédire la survie des passagers en classification binaire (accuracy).

## Ce que fait le projet

- Analyse exploratoire (EDA) : distribution des variables, valeurs manquantes, corrélations avec la survie (âge, sexe, classe).
- Prétraitement : imputation des valeurs manquantes (`Age`, `Embarked`) et encodage des variables catégorielles (`Sex`, `Embarked`).
- Feature engineering : extraction des titres (`Mr`, `Mrs`, `Master`) depuis le nom et création de groupes de familles.
- Entraîne et compare plusieurs modèles de classification (régression logistique, Random Forest, Gradient Boosting).
- Génère le fichier de soumission au format Kaggle.

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition (`train.csv`, `test.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/titanic

pip install pandas numpy scikit-learn jupyter
```

## Utilisation

Placer `train.csv` et `test.csv` à la racine, puis exécuter le notebook :

```bash
jupyter notebook titanic.ipynb
```

Le notebook produit `submission.csv`, prêt à être soumis sur Kaggle.

## Structure du projet

```text
Titanic/
  titanic.ipynb  # Pipeline complet : EDA, prétraitement, feature engineering, modélisation
```
