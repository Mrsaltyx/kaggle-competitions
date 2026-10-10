# Kaggle Playground S6E7 — Predicting Student Health Risk

> 📦 Contenu historique importé depuis `Mrsaltyx/kaggle-s6e7-student-health-risk`, conservé dans ce monorepo.

[![Kaggle Public](https://img.shields.io/badge/Kaggle%20Public-0.95307-brightgreen)](https://www.kaggle.com/competitions/playground-series-s6e7)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Solution pour la compétition [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) : prédire le risque de santé étudiant en **balanced accuracy**, sur une cible très déséquilibrée (`at-risk` ≈ 86 %).

**Meilleur score public : 0.95307** (C26, fichier Naji reproduit le jour J) — C1 0.95267 reste la référence robuste pour le private split.

## Ce que fait le projet

- Pipeline GPU poussé via kernels Kaggle : CatBoost ×5 seeds, XGBoost CUDA, KNN-lookup features, calibration OOF.
- Méta-ensemble local : collecte de soumissions publiques, déduplication (accord > 0.999), vote majoritaire pondéré `weight = (lb − 0.948)^4`, variantes de calibration.
- Correction de prior calibrée sur l'OOF (`proba / prior**beta`) — indispensable sur métrique balanced accuracy (+0.06 vs argmax naïf).
- Journal d'expérimentation complet : 20+ soumissions documentées avec verdicts mesurés.

## Historique des scores (public LB)

| Version | Approche | LB |
|---|---|---|
| v1 | LightGBM 10-fold, argmax naïf | 0.87868 |
| v3 | Ensemble XGB + Cat + LGB GPU, prior-correction OOF | 0.94902 |
| v6 | CatBoost GPU ×5 seeds, synth-only, calibration OOF | 0.94908 |
| v8 | XGB / Cat non pondérés + TargetEncoder + beta = 1.0 | 0.94956 |
| v9 | Seed-bag 3+3, XGB non pondéré + beta 1.05 | 0.94975 |
| v10 | Optuna XGB (60 trials TPE) + 5×3 seeds + seuils par classe | 0.95004 |
| **C1** | **Vote pondéré exponentiel de 12 notebooks publics + v6** | **0.95267** |
| **C26** | **Fichier Naji post-processing calibration (probé, jour J)** | **0.95307** |

## Enseignements clés

1. **La métrique est tout** : la balanced accuracy punit l'argmax naïf sur données déséquilibrées ; la prior-correction vaut ~+0.06.
2. **Le test suit la distribution du train synthétique**, pas celle du dataset original (LB ≈ OOF synthétique).
3. **La règle source** : `sleep_duration` (~6h) + `stress_level` + `physical_activity_level` expliquent ~0.99 du dataset source ; le plafond ~0.95 vient des valeurs manquantes dans ces variables génératrices.
4. **Le plafond GBM est ~0.949–0.950** en single pipeline (record maison : 0.95004 via Optuna TPE). Au-delà : méta-ensemble des meilleures soumissions publiques.
5. **OOF ≠ LB à cette marge** : au-delà de 0.9499, le sur-ajustement OOF devient contre-productif ; l'écart au top (0.00043) est dans le bruit du public split (σ ≈ 0.0005).

## Pistes mesurées et fermées

- Vote générateur (clés sleep / stress / activité) : signal absorbé par C1 à 99.5 %, ne transfère pas sur test (−0.0007).
- Pseudo-labeling @0.995 : dégrade (−0.0004).
- Target encoding numérique valeur-exacte : OOF record mais transfert négatif (−0.0008).
- Flips minés au public LB : risqués pour le private split (rang public anti-corrélé au rang final sur S6E6).

## Structure du projet

```text
kaggle-s6e7-student-health-risk/
  notebooks/
    s6e7-health-risk.ipynb    # Pipeline GPU (CatBoost ×5 + XGBoost cuda, calibration)
  kaggle/
    kernel-metadata.json      # Config kernel Kaggle (GPU, dataset original attaché)
  kernels/                    # Sources des kernels v10 / v10b Optuna
  scripts/
    blend.py                  # Méta-ensemble : collecte, déduplication, vote pondéré
  results/
    best-submission-0.95267.csv  # Meilleure soumission à date
```

## Crédits

Blend construit sur des soumissions publiques : amanatar (solution 0.95267), souvikdbiswas, chakrabhuanavdeva, masayakawamata, jjsdfd22, amerhu, artkomissar, et analyse AleD (règle source & missingness).

## État du projet
- **31/07 (jour J)** : le fichier Naji « post-processing calibration » reproduit **0.95307** (= C26) — nouveau meilleur score public. Attention : fichier explicitement « probé » sur l'oracle public (222 flips minés ligne par ligne), espérance private ≈ C1 avec variance accrue.
- Compétition terminée côté exploration : après 21 soumissions documentées, toutes les pistes mesurables ont été testées.
- Sélection finale : **C1 (0.95267)** robuste + au choix **S3 (0.95004, hedge honnête indépendant)** ou **C26 (0.95307, double exposition au bandeau de tête)**. Défaut Kaggle sans action : auto-sélection C26 + C1.

## Résultat final — private split (01/08)

**Classement final : 596e / 3356 équipes (top 18 %) — score privé 0.95020.**

### Le shake-down prédit s'est produit, presque exactement comme mesuré
| Entrée | Public | Privé | Verdict |
|---|---|---|---|
| Top public (khante) | 0.95340 | 0.95025 (440e) | effondrement du bandeau |
| C26 naji (probé) | 0.95307 | 0.95020 (605e) | le probing a été intégralement puni (−0.0029) |
| **C1 (notre sélection)** | 0.95267 | **0.95020 (596e)** | le consensus tient, mais ne gagne pas non plus |
| v10 TE num (notre OOF record) | 0.94958 | **0.95039** | notre meilleur privé ! Public pessimiste, privé optimiste |
| Vainqueur (Shantanu Chopra) | ~0.950 | 0.95085 | soumission du 12/07, modèle honnête et précoce |

### Enseignements finaux (validés par le private split)
1. **Le cadre georgymamarin était exact** : 1 score public = σ 0.0005, le bandeau de tête public s'est réordonné quasi aléatoirement, les fichiers probés ont régressé à leur espérance (~0.9502).
2. **Le gagnant venait d'un modèle honnête soumis tôt** — reproduction du pattern S6E6 (vainqueur venu du midfield public).
3. **Ironie mesurée** : notre v10 « échec public » (0.94958) était notre meilleur fichier privé (0.95039, ~212e potentiel). La sélection C26+C1 était rationnelle ex-ante (espérance max), la réalisation a favorisé les modèles honnêtes.
4. **Plafond réel des données confirmé** : ~0.9509 en privé. Tout l'écart public au-delà était du bruit miné.
5. Sur 21 soumissions documentées, la démarche (consensus robuste + hedge honnête + exploration exhaustive) a tenu la route : top 18 % final sur 3356 équipes.
