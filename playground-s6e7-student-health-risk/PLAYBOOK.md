# Playbook Kaggle — leçons de Playground S6E7 (juillet 2026)

Contexte : compétition « Predicting Student Health Risk » (owner = mrsaltyx, repo
`Mrsaltyx/kaggle-s6e7-student-health-risk`). Résultat final : **596e / 3356 (top 18 %)**,
privé 0.95020, meilleur public 0.95307. Métrique : balanced accuracy, cible déséquilibrée
(at-risk ≈ 86 %). Deadline → private shake-down massif, documenté dans le README du repo.

## Leçons stratégiques (validées par le private split)

1. **σ(score public) ≈ 0.0005 sur un split ~59k lignes.** Tout écart public < 0.001 est du
   bruit : ne jamais optimiser dessus. Mesuré par georgymamarin, confirmé par nos 21
   soumissions.
2. **Le bandeau de tête public se réordonne quasi aléatoirement au privé** (S6E6 : rang
   public/final anti-corrélé −0.50 ; S6E7 : top public 0.9534 → 440e privé). Le vainqueur
   vient typiquement d'un modèle honnête soumis tôt (S6E7 : soumis le 12/07).
3. **Le probing/mining de l'oracle public est intégralement puni** : le fichier naji
   (0.95307 public, 222 flips minés) → 0.95020 privé, −0.0029. Ne JAMAIS sélectionner un
   fichier probé en final.
4. **Sélection finale optimale (2 slots)** : 1 consensus robuste (méta-ensemble de
   notebooks publics dédupliqués, vote pondéré) + 1 modèle maison honnête au meilleur OOF,
   MÊME si son score public déçoit. Erreur S6E7 : on a écarté v10 (OOF record 0.95034,
   public 0.94958) qui était notre meilleur fichier privé (0.95039, ~212e au lieu de 596e).
   → Le score public d'un modèle honnête ne prédit PAS son rang privé à marge fine.
5. **Le plafond est dans les données, pas dans les modèles.** Toutes les familles (XGB,
   LGBM, CatBoost, RealMLP, FT-Transformer, TabPFN) convergent au même score ±0.0006.
   Au-delà du plafond single-model, seuls le consensus (+0.002) et le bruit miné (punition
   privée) « montent ».

## Leçons techniques

6. **Métrique balanced accuracy sur cible déséquilibrée** : la prior-correction
   `argmax(proba / prior**beta)` vaut ~+0.06. Entraîner NON pondéré + corriger post-hoc ;
   class weights + early stopping = bug (eval non pondéré → stop immédiat, OOF 0.14).
   beta optimal ≈ 1 (théorique) ; tuner beta au-delà = bruit.
7. **Datasets Playground synthétiques** : chercher les variables génératrices (ici
   sleep+stress+activity, règle quasi-déterministe, vote majoritaire BA 0.9704 en heldout
   train). MAIS le test est rééchantillonné bruité : les clés ne transfèrent pas (C25
   −0.0007). Tester l'hypothèse coûte 1 soumission, la fermer proprement évite des
   semaines d'acharnement.
8. **Le test suit la distribution synthétique**, pas le dataset original : ajouter
   l'original au train fait perdre ~0.001.
9. **Ce qui n'apporte rien de mesurable** : pseudo-labeling @0.995 (−0.0004), seuils par
   classe (neutre), rule-features dérivées des coupures du générateur (0, contrôle),
   TE numérique valeur-exacte (OOF +0.0005 mais transfert public négatif… et meilleur
   privé ! voir leçon 4 — à garder dans la sélection malgré tout).
10. **Optuna TPE casse le plafond GBM de +0.0003** (depth 7, lr 0.032, min_child 9.5,
    colsample 0.67, reg_lambda 0.36, gamma 0.30). Un 2e round Optuna sur-ajuste l'OOF
    (OOF monte, LB descend).
11. **Ensembling de modèles qui récupèrent la même règle** : accord ~99 %, gain ~+0.0001
    — négligeable. L'ensemble ne paie que sur erreurs décorrélées.

## Processus qui a marché

12. **Journal d'expérimentation systématique** : chaque soumission documentée (hypothèse,
    score, verdict) dans le README Git → zéro test redondant, décisions traçables.
13. **Méta-ensemble C1** : collecte de 12+ notebooks publics, déduplication (accord >
    0.999), vote pondéré `weight = (lb − 0.948)^4`. Le consensus moyenne les
    sur-ajustements individuels → plus robuste au privé que chaque membre.
14. **Validation leak-free** : StratifiedKFold, TE cross-fitté, jamais d'imputation avant
    split. Un OOF honnête > un LB flatteur pour prédire le privé.
15. **Quotas** : 5-10 soumissions/jour ; réserver les tests pour les hypothèses fondées,
    1 soumission = 1 hypothèse mesurable. GPU Kaggle via `kaggle kernels push` (XGB cuda
    + CatBoost GPU OK ; TabPFN inutilisable : licence + CUDA T4 incompatible).
16. **Timing** : soumettre un modèle honnête TÔT dans la compétition (le vainqueur S6E7
    datait du 12/07 sur une compète finissant le 31/07 — la variance privée récompense la
    précocité, pas le raffinement tardif).

## Checklist de démarrage d'une nouvelle compétition

- [ ] Lire la métrique et simuler son comportement (baselines : classe majoritaire, etc.)
- [ ] EDA : drift train/test (adversarial validation), missingness, variables génératrices
- [ ] Baseline leak-free en 1 journée + soumission précoce (ticket privé)
- [ ] Lire les 5-10 notebooks publics les plus votés + les discussions (règle source)
- [ ] Monter le pipeline GPU (XGB/CatBoost) + prior-correction calibrée OOF
- [ ] Construire le méta-ensemble des publics (dédup + vote pondéré) en cours de compète
- [ ] Journal Git à chaque soumission : hypothèse → score → verdict
- [ ] 1 semaine avant la fin : arrêter les micro-tests, verrouiller la sélection
      (consensus + meilleur OOF honnête), ignorer les fichiers probés même à +0.001 public
