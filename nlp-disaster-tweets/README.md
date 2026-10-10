# Disaster Tweets — NLP (Kaggle Getting Started)

> 📦 Contenu historique importé depuis `Mrsaltyx/Disaster_Tweet`, conservé dans ce monorepo.

[![Kaggle](https://img.shields.io/badge/Kaggle-Disaster%20Tweets-20BEFF.svg)](https://www.kaggle.com/competitions/nlp-getting-started)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition Kaggle « Natural Language Processing with Disaster Tweets » : distinguer les tweets signalant une catastrophe réelle de ceux qui n'en signalent pas (F1-score).

## Ce que fait le projet

- Analyse exploratoire : distribution des tweets, mots-clés et localisations sur ~10 000 tweets annotés.
- Prétraitement du texte : nettoyage (URL, mentions, ponctuation), tokenisation, lemmatisation et gestion des stop words.
- Vectorisation : TF-IDF ou embeddings de mots (GloVe) pour transformer le texte en features numériques.
- Compare plusieurs familles de modèles : Naive Bayes et régression logistique (baselines), XGBoost (boosting), approches séquentielles type RNN / LSTM / Transformers (BERT).
- Optimise les hyperparamètres au F1-score sur un jeu de validation et génère la soumission Kaggle.

## Prérequis

- Python 3.10 ou plus récent
- Les jeux de données de la compétition (`train.csv`, `test.csv`)

## Installation

```bash
git clone https://github.com/Mrsaltyx/kaggle-competitions.git
cd kaggle-competitions/nlp-disaster-tweets

pip install pandas numpy scikit-learn nltk xgboost jupyter
# Pour les modèles de deep learning : tensorflow ou pytorch
```

## Utilisation

Placer `train.csv` et `test.csv` à la racine, puis exécuter le notebook pas à pas :

```bash
jupyter notebook sample.ipynb
```

## Structure du projet

```text
Disaster_Tweet/
  sample.ipynb  # Pipeline NLP complet : EDA, prétraitement, vectorisation, modélisation
```
