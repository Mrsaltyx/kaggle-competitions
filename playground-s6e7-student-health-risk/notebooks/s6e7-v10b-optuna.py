# S6E7 v10 — Optuna-tuned XGBoost (GPU) + correction beta/seuils + pseudo-labeling
# Recette de base v9 (record single 0.94975) : TargetEncoder sklearn, NON pondere,
# correction post-hoc. Ameliorations :
#   1. Optuna (TPE) sur les hyperparams XGB, objectif = BAcc OOF apres correction beta
#   2. Correction finale : beta scalaire VS seuils par classe (Nelder-Mead), garde-fou OOF
#   3. Pseudo-labeling @0.995 + reentrainement, mix fixe 0.75/0.25
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 5
SEEDS = [42, 1, 2026]
OPTUNA_TRIALS = 80
OPTUNA_TIMEOUT = 5400          # 90 min de tuning max
TUNE_FOLDS = 3                 # CV rapide pendant le tuning
PSEUDO_THRESHOLD = 0.995
PL_MIX = 0.25

CANDIDATES = ["/kaggle/input/competitions/playground-series-s6e7",
              "/kaggle/input/playground-series-s6e7", "./data", "."]
DATA_DIR = next(p for p in CANDIDATES if os.path.exists(os.path.join(p, "train.csv")))
print("DATA_DIR:", DATA_DIR)
TARGET = "health_condition"; ID_COL = "id"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != TARGET]
num_cols = [c for c in train.columns if c not in [TARGET, ID_COL] + cat_cols]

target_le = LabelEncoder()
y = target_le.fit_transform(train[TARGET])
N_CLASSES = len(target_le.classes_)
prior = np.bincount(y) / len(y)
print("classes:", list(target_le.classes_), "| prior:", prior.round(4))

X = train[num_cols + cat_cols].copy()
X_test = test[num_cols + cat_cols].copy()
print(X.shape, X_test.shape)

BETAS = np.arange(0.0, 1.51, 0.05)

def best_beta_score(oof):
    """BAcc max sur scan beta scalaire (rapide, argmax seulement)."""
    return max(balanced_accuracy_score(y, (oof / prior ** bb).argmax(1)) for bb in BETAS)

def run_xgb_cv(params, n_folds, seeds, n_estimators=3000, es_rounds=200):
    """CV XGB avec TargetEncoder par fold. Renvoie (oof, tst|None)."""
    oof = np.zeros((len(X), N_CLASSES))
    tst = np.zeros((len(X_test), N_CLASSES))
    n_runs = n_folds * len(seeds)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            enc = TargetEncoder(target_type="multiclass", smooth="auto",
                                random_state=seed, cv=5)
            Xtr = np.hstack([X.iloc[tr][num_cols].to_numpy(float),
                             enc.fit_transform(X.iloc[tr][cat_cols], y[tr])])
            Xva = np.hstack([X.iloc[va][num_cols].to_numpy(float),
                             enc.transform(X.iloc[va][cat_cols])])
            Xte = np.hstack([X_test[num_cols].to_numpy(float),
                             enc.transform(X_test[cat_cols])])
            m = xgb.XGBClassifier(
                objective="multi:softprob", num_class=N_CLASSES,
                tree_method="hist", device="cuda",
                n_estimators=n_estimators, early_stopping_rounds=es_rounds,
                random_state=seed, n_jobs=-1, **params)
            m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
            oof[va] += m.predict_proba(Xva) / len(seeds)
            tst += m.predict_proba(Xte) / n_runs
    return oof, tst

# ================= 1. OPTUNA =================
t0 = time.time()
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }
    oof, _ = run_xgb_cv(params, TUNE_FOLDS, [42], n_estimators=2000, es_rounds=150)
    raw = balanced_accuracy_score(y, oof.argmax(1))
    corr = best_beta_score(oof)
    print(f"trial {trial.number:3d} | raw {raw:.5f} | beta-corr {corr:.5f}")
    return corr

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=7))
study.enqueue_trial({"max_depth": 7, "learning_rate": 0.03235006746937468,
                     "min_child_weight": 9.523353984919787, "subsample": 0.8068832385649766,
                     "colsample_bytree": 0.6682247247167337, "reg_lambda": 0.3638218974136179,
                     "reg_alpha": 0.004987824133203836, "gamma": 0.29629275615574696})  # warm-start = optimum v10 (OOF 0.94993)
study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
best_params = study.best_params
print(f"OPTUNA DONE ({time.time()-t0:.0f}s) | best beta-corr OOF = {study.best_value:.5f}")
print("best params:", json.dumps(best_params, indent=1))
with open("optuna_best.json", "w") as f:
    json.dump({"value": study.best_value, "params": best_params}, f, indent=1)

# ================= 2. ENTRAINEMENT FINAL =================
oof, tst = run_xgb_cv(best_params, N_FOLDS, SEEDS)
raw_oof = balanced_accuracy_score(y, oof.argmax(1))
print(f"[FINAL] OOF brut (argmax) = {raw_oof:.5f}")

# ================= 3. CORRECTION : beta VS seuils =================
beta_scores = {bb: balanced_accuracy_score(y, (oof / prior ** bb).argmax(1)) for bb in BETAS}
best_beta = max(beta_scores, key=beta_scores.get)
print(f"beta optimal = {best_beta:.2f} | OOF {beta_scores[0.0]:.5f} -> {beta_scores[best_beta]:.5f}")

def neg_bacc_thresh(t):
    t = np.abs(t)
    return -balanced_accuracy_score(y, (oof * t).argmax(1))

res = minimize(neg_bacc_thresh, x0=prior ** (-best_beta), method="Nelder-Mead",
               options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-7})
learned_t = np.abs(res.x); learned_t /= learned_t.mean()
thresh_score = balanced_accuracy_score(y, (oof * learned_t).argmax(1))
print(f"seuils appris {np.round(learned_t, 4)} | OOF seuils = {thresh_score:.5f}")

if thresh_score > beta_scores[best_beta]:
    correction = {"type": "thresholds", "t": learned_t.tolist(), "oof": thresh_score}
    tst_corr = tst * learned_t
    print("correction retenue : SEUILS")
else:
    correction = {"type": "beta", "beta": float(best_beta), "oof": beta_scores[best_beta]}
    tst_corr = tst / prior ** best_beta
    print("correction retenue : BETA")

np.save("oof.npy", oof)
np.save("test_proba.npy", tst)
with open("correction.json", "w") as f:
    json.dump(correction, f, indent=1)

pred = target_le.inverse_transform(tst_corr.argmax(1))
sub = sample.copy()
sub[TARGET] = pred
sub.to_csv("submission.csv", index=False)
print(sub[TARGET].value_counts(normalize=True))
print("OK submission.csv (pre-PL)")

# ================= 4. PSEUDO-LABELING =================
conf = tst_corr.max(1)
mask = conf >= PSEUDO_THRESHOLD
n_pseudo = int(mask.sum())
print(f"Pseudo-labels @ {PSEUDO_THRESHOLD}: {n_pseudo} ({100*n_pseudo/len(X_test):.1f}%)")
if n_pseudo > 1000:
    py = tst_corr[mask].argmax(1)
    X_aug = pd.concat([X, X_test.iloc[mask]], ignore_index=True)
    y_aug = np.concatenate([y, py])
    oof_pl = np.zeros((len(X_aug), N_CLASSES))
    tst_pl = np.zeros((len(X_test), N_CLASSES))
    n_runs = N_FOLDS
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=84)
    for fold, (tr, va) in enumerate(skf.split(X_aug, y_aug)):
        enc = TargetEncoder(target_type="multiclass", smooth="auto", random_state=84, cv=5)
        Xtr = np.hstack([X_aug.iloc[tr][num_cols].to_numpy(float),
                         enc.fit_transform(X_aug.iloc[tr][cat_cols], y_aug[tr])])
        Xva = np.hstack([X_aug.iloc[va][num_cols].to_numpy(float),
                         enc.transform(X_aug.iloc[va][cat_cols])])
        Xte = np.hstack([X_test[num_cols].to_numpy(float),
                         enc.transform(X_test[cat_cols])])
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=N_CLASSES,
            tree_method="hist", device="cuda",
            n_estimators=3000, early_stopping_rounds=200,
            random_state=84, n_jobs=-1, **best_params)
        m.fit(Xtr, y_aug[tr], eval_set=[(Xva, y_aug[va])], verbose=False)
        oof_pl[va] = m.predict_proba(Xva)
        tst_pl += m.predict_proba(Xte) / n_runs
        print(f"[PL] fold {fold+1} done")
    np.save("test_proba_pl.npy", tst_pl)
    tst_mix = (1 - PL_MIX) * tst + PL_MIX * tst_pl
    if correction["type"] == "thresholds":
        tst_mix_corr = tst_mix * np.array(correction["t"])
    else:
        tst_mix_corr = tst_mix / prior ** correction["beta"]
    pred_pl = target_le.inverse_transform(tst_mix_corr.argmax(1))
    sub_pl = sample.copy()
    sub_pl[TARGET] = pred_pl
    sub_pl.to_csv("submission_pl.csv", index=False)
    n_flip = int((sub_pl[TARGET] != sub[TARGET]).sum())
    print(f"OK submission_pl.csv | flips vs pre-PL = {n_flip}")
    print(sub_pl[TARGET].value_counts(normalize=True))
print("V10 TERMINE")
