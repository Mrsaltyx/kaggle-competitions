# Blend v2 — variantes systématiques pour dépasser 0.95267
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Salty\Downloads\Predicting Student Health Risk"
PUB = os.path.join(ROOT, "pub-files")
CLASSES = ["at-risk", "fit", "unhealthy"]
C2I = {c: i for i, c in enumerate(CLASSES)}

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

sample = pd.read_csv(os.path.join(ROOT, "kaggle-output-v4", "submission.csv"))["id"]
N = len(sample)

members = {}
for d in sorted(os.listdir(PUB)):
    ddir = os.path.join(PUB, d)
    if not os.path.isdir(ddir):
        continue
    f = os.path.join(ddir, "submission.csv")
    if d.startswith("liamob96"):
        f = os.path.join(ddir, "submission_realmlp_style_v1.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    if "health_condition" not in df.columns or len(df) != N:
        continue
    members[d] = (df["health_condition"].map(C2I).values, KNOWN_LB.get(d, DEFAULT_LB))

v6 = pd.read_csv(os.path.join(ROOT, "kaggle-output-v6", "submission.csv"))["health_condition"].map(C2I).values
members["mrsaltyx_v6_catboost"] = (v6, 0.94908)
print(f"{len(members)} membres")

def dedup(names, thr):
    kept = []
    for n in sorted(names, key=lambda k: -members[k][1]):
        if all((members[n][0] == members[k][0]).mean() < thr for k in kept):
            kept.append(n)
    return kept

def vote(kept, expo, seuil=0.948):
    lbs = np.array([members[n][1] for n in kept])
    w = np.clip(lbs - seuil, 1e-6, None) ** expo
    w /= w.sum()
    V = np.zeros((N, 3))
    for n, wi in zip(kept, w):
        V[np.arange(N), members[n][0]] += wi
    return V, dict(zip(kept, w.round(3)))

def write(pred_idx, name):
    sub = pd.DataFrame({"id": sample, "health_condition": [CLASSES[i] for i in pred_idx]})
    sub.to_csv(os.path.join(ROOT, f"blend-{name}.csv"), index=False)
    d = sub["health_condition"].value_counts(normalize=True)
    print(f"{name}: " + "  ".join(f"{c}={d.get(c, 0):.3f}" for c in CLASSES))

# C6 : 13 membres (dedup 0.999) incluant realmlp, poids ^4
k6 = dedup(members.keys(), 0.999)
print("C6 membres:", len(k6))
V6, w6 = vote(k6, 4)
write(V6.argmax(1), "C6-realmlp-p4")

# C7 : dedup 0.9995 (garde les quasi-doublons et leurs micro-flips), poids ^4
k7 = dedup(members.keys(), 0.9995)
print("C7 membres:", len(k7))
V7, _ = vote(k7, 4)
write(V7.argmax(1), "C7-dedup-lache-p4")

# C8 : mêmes membres que C6, poids ^2 (plus doux -> plus de flips minoritaires)
V8, w8 = vote(k6, 2)
write(V8.argmax(1), "C8-realmlp-p2")

# C9 : C6 + tilt minoritaire sur les votes serrés (marge < 0.12 -> 2e choix si minoritaire)
srt = np.sort(V6, axis=1)
margin = srt[:, 2] - srt[:, 1]
pred = V6.argmax(1)
second = np.argsort(-V6, axis=1)[:, 1]
flip = (margin < 0.12) & (pred == 0) & (second != 0)
pred9 = pred.copy(); pred9[flip] = second[flip]
print(f"C9: {flip.sum()} flips at-risk -> minoritaire")
write(pred9, "C9-margin-tilt")
print("terminé")
