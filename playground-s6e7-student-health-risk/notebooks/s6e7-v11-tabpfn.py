# S6E7 v11 — TabPFN-3 + XGBoost (GPU) blend, correction beta/seuils
# Inspire d'assemelqirsh N17 (TabPFN balance_probabilities, PAS de double correction)
# mais base XGB non pondere style v9 + correction beta/seuils apres fusion.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.0.9"], check=False)
import gc, json, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize
import xgboost as xgb

warnings.filterwarnings("ignore")

N_FOLDS = 5
SEEDS = [42, 2026]
TABPFN_ESTIMATORS = 2
TABPFN_SUBSAMPLE = 50_000

import glob as _glob
for root in ["/kaggle/input/models", "/kaggle/input"]:
    for dp, dn, fn in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth <= 4:
            print("DIR", dp, "->", fn[:8])
        if depth > 4:
            dn[:] = []
_ckpts = _glob.glob("/kaggle/input/models/**/*.ckpt", recursive=True) +          _glob.glob("/kaggle/input/**/tabpfn*.ckpt", recursive=True)
print("CKPTS:", _ckpts)
TABPFN_CKPT = _ckpts[0] if _ckpts else None
if TABPFN_CKPT:
    os.environ["TABPFN_MODEL_CACHE_DIR"] = os.path.dirname(TABPFN_CKPT)

GPU = torch.cuda.is_available()
print("GPU:", GPU, "| ckpt:", TABPFN_CKPT)

CANDIDATES = ["/kaggle/input/competitions/playground-series-s6e7",
              "/kaggle/input/playground-series-s6e7", "./data", "."]
DATA_DIR = next(p for p in CANDIDATES if os.path.exists(os.path.join(p, "train.csv")))
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
BETAS = np.arange(0.0, 1.51, 0.05)

def best_beta_score(oof):
    return max(balanced_accuracy_score(y, (oof / prior ** bb).argmax(1)) for bb in BETAS)

# ================= 1. BASE XGB (recette v9) =================
xgb_params = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
                  min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0)
oof_x = np.zeros((len(X), N_CLASSES))
tst_x = np.zeros((len(X_test), N_CLASSES))
n_runs = N_FOLDS * len(SEEDS)
t0 = time.time()
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        enc = TargetEncoder(target_type="multiclass", smooth="auto", random_state=seed, cv=5)
        Xtr = np.hstack([X.iloc[tr][num_cols].to_numpy(float),
                         enc.fit_transform(X.iloc[tr][cat_cols], y[tr])])
        Xva = np.hstack([X.iloc[va][num_cols].to_numpy(float),
                         enc.transform(X.iloc[va][cat_cols])])
        Xte = np.hstack([X_test[num_cols].to_numpy(float),
                         enc.transform(X_test[cat_cols])])
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=N_CLASSES,
                              tree_method="hist", device="cuda",
                              early_stopping_rounds=200, random_state=seed,
                              n_jobs=-1, **xgb_params)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
        oof_x[va] += m.predict_proba(Xva) / len(SEEDS)
        tst_x += m.predict_proba(Xte) / n_runs
print(f"[XGB] OOF brut {balanced_accuracy_score(y, oof_x.argmax(1)):.5f} | "
      f"beta-corr {best_beta_score(oof_x):.5f} ({time.time()-t0:.0f}s)")
np.save("oof_xgb.npy", oof_x); np.save("tst_xgb.npy", tst_x)

# ================= 2. TABPFN-3 =================
imp = SimpleImputer(strategy="median")
X_num = imp.fit_transform(train[num_cols]); X_num_te = imp.transform(test[num_cols])
X_tab = X_num.copy(); X_tab_te = X_num_te.copy()
for c in cat_cols:
    cats = pd.Categorical(train[c].fillna("Missing").astype(str))
    X_tab = np.hstack([X_tab, cats.codes.astype(float).reshape(-1, 1)])
    X_tab_te = np.hstack([X_tab_te,
        pd.Categorical(test[c].fillna("Missing").astype(str),
                       categories=cats.categories).codes.astype(float).reshape(-1, 1)])

def batch_predict(clf, Xa, bs=4096):
    Xa = np.asarray(Xa)
    return np.vstack([clf.predict_proba(Xa[i:i+bs]) for i in range(0, len(Xa), bs)])

oof_t = np.zeros((len(X), N_CLASSES))
tst_t = np.zeros((len(X_test), N_CLASSES))
TABPFN_OK = True
try:
    from tabpfn import TabPFNClassifier
    print("tabpfn import OK")
except Exception as e:
    print("IMPORT TABPFN FAILED:", repr(e))
    TABPFN_OK = False

def make_tabpfn():
    kw = dict(device="cpu", n_estimators=TABPFN_ESTIMATORS,
              ignore_pretraining_limits=True)
    if TABPFN_CKPT:
        kw["model_path"] = TABPFN_CKPT
        kw["fit_mode"] = "fit_with_cache"
    try:
        return TabPFNClassifier(balance_probabilities=True, **kw)
    except TypeError as e:
        print("balance_probabilities non supporte:", e)
        return TabPFNClassifier(**kw)
if TABPFN_OK:
  try:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X_tab, y)):
        sub_idx, _ = train_test_split(np.arange(len(tr)),
                                      train_size=min(TABPFN_SUBSAMPLE, len(tr)),
                                      stratify=y[tr], random_state=42 + fold)
        clf = make_tabpfn()
        clf.fit(X_tab[tr[sub_idx]], y[tr[sub_idx]])
        va_p = batch_predict(clf, X_tab[va])
        oof_t[va] = va_p
        tst_t += batch_predict(clf, X_tab_te) / N_FOLDS
        print(f"[TabPFN] fold {fold+1} | BAcc {balanced_accuracy_score(y[va], va_p.argmax(1)):.5f}")
        del clf; gc.collect()
        if GPU: torch.cuda.empty_cache()
    print(f"[TabPFN] OOF {balanced_accuracy_score(y, oof_t.argmax(1)):.5f} ({time.time()-t0:.0f}s)")
    np.save("oof_tabpfn.npy", oof_t); np.save("tst_tabpfn.npy", tst_t)
  except Exception as e:
    import traceback
    with open("tabpfn_error.txt", "w") as f:
        f.write(traceback.format_exc())
    print("TABPFN CRASH:", repr(e))
    TABPFN_OK = False
else:
    print("TabPFN indisponible, blend XGB seul")

# ================= 3. FUSION + CORRECTION =================
best_a, best_s = 0.0, best_beta_score(oof_x)
if TABPFN_OK:
    for a in np.arange(0.02, 0.61, 0.02):
        s = best_beta_score(a * oof_t + (1 - a) * oof_x)
        if s > best_s:
            best_s, best_a = s, a
print(f"fusion : tabpfn={best_a:.2f} xgb={1-best_a:.2f} | OOF beta-corr = {best_s:.5f}")
oof = best_a * oof_t + (1 - best_a) * oof_x
tst = best_a * tst_t + (1 - best_a) * tst_x

beta_scores = {bb: balanced_accuracy_score(y, (oof / prior ** bb).argmax(1)) for bb in BETAS}
best_beta = max(beta_scores, key=beta_scores.get)

def neg_bacc_thresh(t):
    t = np.abs(t)
    return -balanced_accuracy_score(y, (oof * t).argmax(1))

res = minimize(neg_bacc_thresh, x0=prior ** (-best_beta), method="Nelder-Mead",
               options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-7})
learned_t = np.abs(res.x); learned_t /= learned_t.mean()
thresh_score = balanced_accuracy_score(y, (oof * learned_t).argmax(1))
print(f"beta {best_beta:.2f} -> {beta_scores[best_beta]:.5f} | seuils {np.round(learned_t,4)} -> {thresh_score:.5f}")

if thresh_score > beta_scores[best_beta]:
    correction = {"type": "thresholds", "t": learned_t.tolist(), "oof": thresh_score}
    tst_corr = tst * learned_t
else:
    correction = {"type": "beta", "beta": float(best_beta), "oof": beta_scores[best_beta]}
    tst_corr = tst / prior ** best_beta
print("correction:", correction)

np.save("oof.npy", oof); np.save("test_proba.npy", tst)
with open("correction.json", "w") as f:
    json.dump({**correction, "tabpfn_weight": float(best_a)}, f, indent=1)

pred = target_le.inverse_transform(tst_corr.argmax(1))
sub = sample.copy()
sub[TARGET] = pred
sub.to_csv("submission.csv", index=False)
print(sub[TARGET].value_counts(normalize=True))
print("V11 TERMINE")
