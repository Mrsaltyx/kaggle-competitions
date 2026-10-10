# CIFAR-10 — Classification d'images (Kaggle)

> 📦 Contenu historique importé depuis `Mrsaltyx/Cifar-10`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-CIFAR--10-20BEFF.svg)](https://www.kaggle.com/c/cifar-10)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition Kaggle « CIFAR-10 - Classification » : classifier des images 32×32 en 10 catégories avec un réseau de neurones convolutif (accuracy).

## Ce que fait le projet

- Analyse exploratoire : visualisation d'échantillons et vérification de l'équilibre des classes.
- Prétraitement : normalisation des pixels et augmentation de données (rotation, zoom, flip) pour renforcer la robustesse du modèle.
- Conçoit un CNN complet : couches de convolution (Conv2D) pour l'extraction de caractéristiques, MaxPooling pour la réduction de dimensionnalité, Dropout contre le sur-apprentissage, couches denses pour la classification finale.
- Entraîne le modèle avec l'optimiseur Adam et early stopping.
- Évalue la précision sur un jeu de validation et génère le fichier de soumission.

## Prérequis

- Python 3.10 ou plus récent
- Les données de la compétition (`train.zip`, `test.zip`, `sample_submission.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/cifar-10-image-classification

pip install tensorflow keras pandas numpy scikit-learn matplotlib
```

## Utilisation

Extraire les archives d'images à la racine, puis lancer :

```bash
python Cifar-10.py
```

Le script entraîne le CNN et produit `submission.csv` au format Kaggle.

## Structure du projet

```text
Cifar-10/
  Cifar-10.py  # Pipeline complet : prétraitement, augmentation, CNN, entraînement, soumission
```
