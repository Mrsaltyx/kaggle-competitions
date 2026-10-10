# Blend local des meilleures soumissions publiques S6E7 + notre v6
# Stratégie inspirée d'amanatar (0.95267) et amerhu (0.95088)
import os, re, glob
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Salty\Downloads\Predicting Student Health Risk"
PUB = os.path.join(ROOT, "pub-files")
CLASSES = ["at-risk", "fit", "unhealthy"]  # ordre alphabétique = notre LabelEncoder
C2I = {c: i for i, c in enumerate(CLASSES)}

# LB publics connus (titres/slugs). Inconnus -> 0.94900 conservateur
KNOWN_LB = {
    "amanatar_medal-solution-s6e7-smart-ensemble-lb-0-95267": 0.95267,
    "amanatar_ultimate-ensemble-top-15-ground-truth-flips": 0.95260,
    "crystalbaby_lb-95-267": 0.95267,
    "navazshfathi_predicting-student-health-risk-95-267": 0.95267,
    "jjsdfd22_0-95238-starter": 0.95238,
    "chakrabhuanavdeva_gbdt-lb-0-94986": 0.94986,
    "souvikdbiswas_s6e7-quit-wasting-4-hours-on-training-0-95-in-5": 0.95050,
    "masayakawamata_s6e7-xgb-1-cv-0-94986": 0.94986,
    "masayakawamata_s6e7-repleaf-1-cv-0-94964": 0.94964,
    "liamob96_realmlp-nn-0-94929-oof-feature-engineering": 0.94929,
}
DEFAULT_LB = 0.94900

sample = pd.read_csv(os.path.join(ROOT, "kaggle-output-v4", "submission.csv"))["id"]  # ids de référence
N = len(sample)

members = {}  # name -> (labels_idx, lb)
for d in sorted(os.listdir(PUB)):
    f = os.path.join(PUB, d, "submission.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    if "health_condition" not in df.columns or len(df) != N:
        print(f"SKIP {d}: format inattendu {df.columns.tolist()} len={len(df)}")
        continue
    lab = df["health_condition"].map(C2I).values
    lb = KNOWN_LB.get(d, DEFAULT_LB)
    members[d] = (lab, lb)

# notre v6
v6 = pd.read_csv(os.path.join(ROOT, "kaggle-output-v6", "submission.csv"))["health_condition"].map(C2I).values
members["mrsaltyx_v6_catboost"] = (v6, 0.94908)

print(f"{len(members)} membres chargés")

# --- déduplication (accord argmax > 0.999) en gardant le meilleur LB ---
kept = []
for n in sorted(members, key=lambda k: -members[k][1]):
    if all((members[n][0] == members[k][0]).mean() < 0.999 for k in kept):
        kept.append(n)
    else:
        print(f"drop quasi-doublon: {n} (LB {members[n][1]})")
print(f"{len(kept)} membres conservés:")
for n in kept:
    print(f"  {members[n][1]:.5f}  {n}")

# --- poids exponentiels (amanatar-style): (lb - seuil)^4 ---
SEUIL = 0.94800
lbs = np.array([members[n][1] for n in kept])
w = np.clip(lbs - SEUIL, 1e-6, None) ** 4
w = w / w.sum()
print("\npoids:", {n.split('_')[0]: round(x, 3) for n, x in zip(kept, w)})

V = np.zeros((N, 3))
for n, wi in zip(kept, w):
    V[np.arange(N), members[n][0]] += wi

# --- candidats ---
MULT_NONE  = np.array([1.0, 1.0, 1.0])
MULT_V6    = np.array([0.7381, 1.0, 0.9524])          # calibrés sur notre OOF synth
MULT_AMERHU = np.array([1.0, 1.125, 1.273])            # validés LB par amerhu

def write(pred_idx, name):
    sub = pd.DataFrame({"id": sample, "health_condition": [CLASSES[i] for i in pred_idx]})
    path = os.path.join(ROOT, f"blend-{name}.csv")
    sub.to_csv(path, index=False)
    d = sub["health_condition"].value_counts(normalize=True)
    print(f"{name}: " + "  ".join(f"{c}={d.get(c, 0):.3f}" for c in CLASSES))
    return path

write((V * MULT_NONE).argmax(1), "C1-vote-brut")
write((V * MULT_V6).argmax(1), "C2-vote-mult-v6")
write((V * MULT_AMERHU).argmax(1), "C3-vote-mult-amerhu")

# --- C4: blend géométrique des probas dispo (masaya xgb/repleaf + notre tst_mix) ---
probs = []
for slug, f, lb in [("xgb", r"pub-files\masayakawamata_s6e7-xgb-1-cv-0-94986\testpred_xgb.csv", 0.94986),
                    ("repleaf", r"pub-files\masayakawamata_s6e7-repleaf-1-cv-0-94964\testpred_repleaf.csv", 0.94964)]:
    p = pd.read_csv(os.path.join(ROOT, f))
    pcols = [c for c in p.columns if c.lower() in CLASSES or c in CLASSES]
    if len(pcols) == 3:
        arr = p[pcols].values
        order = [pcols.index(c) for c in CLASSES]
        probs.append((slug, arr[:, order], lb))
        print(f"probas chargées: {slug} cols={pcols}")
    else:
        print(f"format probas inattendu pour {slug}: {p.columns.tolist()[:6]}")

t6 = os.path.join(ROOT, "kaggle-output-v6", "tst_mix.npy")
if os.path.exists(t6):
    probs.append(("v6", np.load(t6), 0.94908))
    print("probas chargées: v6 (tst_mix.npy)")

if probs:
    wl = np.clip(np.array([p[2] for p in probs]) - SEUIL, 1e-6, None) ** 4
    wl = wl / wl.sum()
    Z = sum(wi * np.log(np.clip(p[1], 1e-7, 1)) for wi, p in zip(wl, probs))
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    write((P * MULT_AMERHU).argmax(1), "C4-geometrique-mult-amerhu")
print("terminé")
