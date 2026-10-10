# Kaggle Playground S6E8 — Predicting Smartphone Addiction

> 📦 Contenu historique importé depuis `Mrsaltyx/kaggle-s6e8-smartphone-addiction`, conservé dans ce monorepo.

Compétition : https://www.kaggle.com/competitions/playground-series-s6e8
**Métrique : ROC AUC** — classification binaire `addicted_label` (71 % / 29 %), 691k train / 296k test.
Deadline : 31/08/2026. Repo créé le 03/08 (jour 3 de la compète).

## 🏁 Résultat final

**#121 / 3532 équipes (top 3,5 %) — private 0.97104, public 0.97129.**
Aucun shakeup subi (−0.00025 public→private) : la sélection à l'OOF honnête a tenu, conforme à la doctrine
(voir « Sélection finale » plus bas). Le top 10 public (0.97135+) et le top 3 (0.97149+) sont restés stables eux aussi —
cette compète récompensait les stacks fold-safe massifs, pas les hacks LB.

| Soumission | Public | Private | Commentaire |
|---|---|---|---|
| v4 (fusion 226 membres) | 0.97129 | **0.97104** | Finaliste 1 — meilleur private |
| v3 (fusion 205 membres) | 0.97125 | 0.97102 | Finaliste 2 |
| v2 (méta-stack 55 membres) | 0.97064 | 0.97042 | Jour 3 |
| v1 (baseline XGB) | 0.96595 | 0.96567 | Jour 3 |

**Playbook S6E7 appliqué** (voir `Mrsaltyx/kaggle-s6e7-student-health-risk/PLAYBOOK.md`) :
baseline honnête précoce, journal de chaque soumission, sélection finale = consensus + meilleur OOF honnête.

## EDA (03/08)

- **Signal concentré sur 5 features monotones fortes** (taux d'addiction 1er → dernier décile) :
  `daily_screen_time_hours` 0.265 → 1.000, `social_media_hours` 0.250 → 0.999,
  `weekend_screen_time` 0.264 → 1.000, `work_study_hours` 0.581 → 0.936, `gaming_hours` 0.638 → 0.922.
- **Catégorielles quasi nulles** : gender 0.70–0.72, stress_level 0.71, academic_work_impact 0.71.
  À garder (coût nul pour un arbre) mais aucun espoir de levier.
- **Features faibles** : age, notifications_per_day, app_opens_per_day (quasi plates).
- **Missingness 4–20 %** sur presque toutes les colonnes, train ≈ test → NaN natifs, pas d'imputation.
- 0 doublon exact train↔test sur toutes les features (rééchantillonnage bruité comme S6E7).
- Top LB au 05/08 : **0.97115** (442 équipes, compète jeune).

## Journal des expériences

| # | Version | Approche | OOF AUC | LB | Verdict |
|---|---|---|---|---|---|
| 1 | v1 | Baseline XGB(cuda)+CatBoost(GPU) 5-fold × 2 seeds, cats ordinales, blend OOF-optimal | 0.96465 | **0.96595** | ✅ LB > OOF, transfert honnête. XGB seul > blend (CatBoost 0.95967 écarté par le blend OOF) |
| 2 | v2 | Méta-stack logistique 55 membres (bibliothèque OOF szymonkapiski + Naji + notre v1), blend percentile conditionné par missingness | 0.969485 | **0.97064** | ✅ +0.0047 vs v1. Le méta-ensemble est le jeu (leçon S6E7/C1). À 0.0004 du top (0.97102) au jour 3 |
| 3 | **v3** | **Repro honnête fusion hboyang 205 membres (rank+logit+ régime, LBFGS), VALIDATE=True, alpha nested = 0.5 fold-consistent** | 0.970394 | **0.97125** | ✅ +0.00061 vs v2, repro exacte du LB public hboyang. Blend ancres honnêtes testés et rejetés en OOF : v2 (Spearman 0.990) et tamerlan full-blend (0.989) n'ajoutent rien — pool saturé sur ces familles |
| 4 | **v4** | **Pool enrichi 226 membres (v3 + aadijoshi mask-aug 9 + paiky 11 neural + masayakawamata, dedup Spearman 0.9995), fits float32 (OOM fix)** | 0.970330 | **0.97129** | ✅ LB +0.00004 vs v3. OOF −0.00006 = dans le bruit float32/float64 → v3 et v4 statistiquement équivalents, v4 plus diversifié. **Finalistes : v3 + v4** |

## Sélection finale (verrouillée le 29/08, J−2)

**Finalistes choisis : v3 (0.97125) + v4 (0.97129)** — les deux meilleurs OOF honnêtes, tous deux fold-safe, aucun hack LB/ID.
- Auto-sélection Kaggle (2 meilleurs scores publics) retombe sur cette paire même sans action manuelle ; vérifier dans l'UI avant la deadline (31/08).
- Refusés : blends Frankenstein LB-only (leçon Naji « Overfitting Trap » : le micro-sorting inversé gagne +0.00001 public et scrambles le private), ancres sans OOF (anthonytherrien vault, thisray), override d'IDs « duplicate magic » (amanatar) — tous non validables honnêtement.
- v2 conservée en repli (OOF 0.969485, LB 0.97064) si un finaliste devait être invalidé.

## Structure

- `kernel/s6e8-baseline.ipynb` — baseline GPU poussée via `kaggle kernels push`
- `kernel/kernel-metadata.json` — config kernel (GPU activé)
- `kernel-v2/` — méta-stack 55 membres (v2)
- `kernel-v3/` — fusion rank-logit-régime 205 membres (v3, `mrsaltyx/s6e8-v3-rank-logit-regime-fusion-205-membres`)
- `eval_blend_v3b.py` — évaluation OOF honnête des blends d'ancres (numpy/pandas only)
- `pub/` — notebooks et datasets publics de référence (non versionnés)
- `data/` — données de la compétition (non versionnées)

## Sélection finale — règle verrouillée d'avance

1. Meilleur consensus/méta-ensemble robuste
2. Meilleur OOF honnête, **même si son score public déçoit** (leçon v10/S6E7 = 400 places)

## Enseignements S6E8 (jour 3)

1. **Le single model n'est pas le jeu** : notre XGB seul = 0.96465 OOF, le stack public = 0.96950. Comme sur S6E7 (0.950 vs 0.9527), le plafond single est loin du plafond consensus. Priorité au méta-ensemble dès le jour 3, pas en semaine 4.
2. **La bibliothèque OOF publique est le levier** : szymonkapiski publie les OOF/test de 47+ modèles (TabM, RealMLP, NN, GBM). Stack logistique en espace logit, déduplication corr > 0.9995, CV 5-fold honnête → reproduction de riponce à 0.00001 près.
3. **Missingness = régime de blending, pas feature prédictive** : aucun signal direct (Δ ≤ 0.004, contrôle fait), mais le poids du blend Naji↔stack optimal dépend du régime de missingness (rayk : 10 % lignes complètes / 40 % incomplètes). Validé fold-consistant, retenu en CV.
4. **AUC → blend en percentile-rank** (pas en probas brutes : AUC ne dépend que de l'ordre).
5. **Prochaines étapes** : la bibliothèque grandit (25 → 47 modèles en 2 jours) → re-stack hebdomadaire ; ajouter nos propres familles (TabM/RealMLP) si elles apportent de la diversité mesurable en CV ; Optuna single model = faible valeur (écart au stack trop grand).

## Surveillance (auto-journal)

| Date | Bibliothèque OOF | Top LB | Notre LB | Écart | Action |
|---|---|---|---|---|---|
| 2026-08-03 19:37 | 47 modèles (szymonkapiski, maj 08:28) | 0.97102 (MILANFX) | 0.97064 | −0.00038 | Aucune — pas de changement détecté |
| 2026-08-05 19:38 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97115 (MILANFX) | 0.97064 | −0.00051 | Aucune — bib inchangée, LB stable (+0.00013), nouveaux datasets détectés (raykkretzschmar FM lattice, dariushafshar Golem 4 modèles) |
| 2026-08-07 19:37 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97120 (MILANFX) | 0.97064 | −0.00056 | Aucune — bib inchangée, LB stable (+0.00005), nouveaux OOF individuels (beicicc 5 artefacts, mohankrishnathalla 3 modèles, boltuzamaki mirror 47) |
| 2026-08-15 19:37 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97124 (MILANFX) | 0.97064 | −0.00060 | Aucune — bib principale inchangée, LB stable (+0.00004), nouveaux datasets : adarsh1077 (11 modèles), dariushafshar golem (4→7 modèles), najiama psa + maj OOF, raykkretzschmar transductive, beicicc crossfit-logitlr, kenchanhodgkin 9 child exps, anthonytherrien vault |
| 2026-08-15 21:37 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97124 (MILANFX + Utkarsh) | 0.97064 | −0.00060 | Aucune — inchangé depuis 19h37, Utkarsh rejoint le top à 0.97124 |
| 2026-08-17 19:37 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97132 (MILANFX) | 0.97064 | −0.00068 | Aucune — bib principale inchangée, LB stable (+0.00008), maj Naji (OOF 17/08 + psa 16/08), nouveau masayakawamata catstr-aug16, 6 kenchanhodgkin child exps, georgymamarin leaderboards |
| 2026-08-19 19:37 | 47 modèles (szymonkapiski, maj 04/08 10:51) | 0.97134 (MILANFX) | 0.97064 | −0.00070 | Aucune — bib principale inchangée, LB stable (+0.00002), nouveau hboyang 150-fusion-local-members, thisray our-component (0.97120 LB), tamerlanomralinov full-best-blend-npy, maj raykkretzschmar transductive (size x2), qamrodz submission |
| 2026-08-24 16:05 | 47 + 50 weak (szymonkapiski 21/08), Naji maj 23/08, hboyang catstrall maj 24/08 | 0.97168 (cdeotte) | **0.97125** | −0.00043 | ✅ v3 soumise : repro honnête fusion hboyang 205 membres (VALIDATE=True, OOF nested 0.970394, alpha=0.5), +0.00061. Ancres v2/tamerlan rejetées en OOF |
| 2026-08-29 13:05 | + aadijoshi mask-aug (28/08), paiky1995 11 neural (25/08), Naji psa maj 27/08 | 0.97207 (cdeotte) | **0.97129** | −0.00078 | ✅ v4 soumise : pool 226 membres, OOF 0.970330 ≈ v3 (bruit float32), LB +0.00004. 3 runs OOM avant fix float32. Finalistes verrouillés : v3 + v4. Naji confirme la doctrine anti-shakeup (notebook « Overfitting Trap ») |
| 2026-09-09 19:37 | Compétition terminée — bibliothèques figées, datasets post-mortem publiés | 0.97176 (Chris Deotte, private) | **0.97104** (private) | −0.00072 | 🏁 **FIN** : #121 / 3532 (top 3,5 %). Aucun shakeup subi (−0.00025 public→private). Sélection v3+v4 validée par le privé. Doctrine S6E7 confirmée : OOF honnête > hacks LB. |
| 2026-09-23 19:37 | Compétition terminée depuis le 31/08 — toutes bibliothèques figées | 0.97176 (Chris Deotte, private) | **0.97104** (private) | −0.00072 | 🏁 **CLÔTURE** : leaderboard figé, résultat final confirmé. Aucune action possible. Repo archivé comme référence pour S6E9+. |
| 2026-09-25 19:37 | Compétition terminée depuis le 31/08 — toutes bibliothèques figées | 0.97176 (Chris Deotte, private) | **0.97104** (private) | −0.00072 | 🏁 **CLÔTURE** : leaderboard figé, résultat final confirmé. Aucune action possible. Repo archivé. |
| 2026-10-05 19:37 | Compétition terminée depuis le 31/08 — toutes bibliothèques figées | 0.97176 (Chris Deotte, private) | **0.97104** (private) | −0.00072 | 🏁 **CLÔTURE** : leaderboard figé, résultat final confirmé. Aucune action possible. Repo archivé. |
| 2026-10-09 19:37 | Compétition terminée depuis le 31/08 — toutes bibliothèques figées | 0.97176 (Chris Deotte, private) | **0.97104** (private) | −0.00072 | 🏁 **CLÔTURE** : leaderboard figé, résultat final confirmé. Aucune action possible. Repo archivé. |

## Enseignements S6E8 (jour 21, reprise)

5. **La fusion publique la plus forte est rank+logit+régime sur ~205 membres** (hboyang, LB 0.97125) : pool communautaire 145 + 50 modèles *faibles* CC0 (la diversité compte plus que la force individuelle, leçon szymonkapiski) + 6 extras + refit méta-stack de base. Reproduite à l'identique : LB 0.97125 exactement.
6. **VALIDATE=True > alpha fixe** : l'alpha nested honnête ressort à 0.5 sur les 5 folds (fold-consistent), et le LB n'en souffre pas. Toujours payer le coût de la validation.
7. **Les ancres corrélées n'ajoutent rien une fois le pool saturé** : v2 (Spearman 0.990) et tamerlan full-blend (0.989) — grille de poids OOF monotone décroissante, w*=0. Ne pas gaspiller de soumissions sur des blends non validés.
8. **Piège d'infra** : le pool GPU Kaggle attribue parfois un P100 (sm_60) incompatible avec le torch actuel → smoke test CUDA + fallback CPU obligatoire (run v3 : ~1h27 sur CPU).
9. **Plafond RAM ~13 Go sur les workers** (CPU ou GPU-P100) : au-delà de ~205 membres, les matrices de fusion float64 (>10 Go) font tuer le kernel silencieusement (« Kernel died while waiting for execute reply ») → **float32 + libération tôt des DataFrames**. Coût : bruit numérique ~1e-5 sur l'OOF, négligeable.
10. **Enrichir un pool saturé ne change rien en OOF** (+21 membres décorrélés à 0.96-0.99 : ΔOOF −0.00006) — mais la diversité supplémentaire est une assurance private à coût nul, d'où la paire v3+v4 en finale.
11. **La fin de compète récompense la discipline, pas les hacks** : Naji (auteur de la bibliothèque OOF la plus utilisée) refuse lui-même son blend 0.97129 LB-probé pour un modèle CV honnête. Nous faisons pareil.

