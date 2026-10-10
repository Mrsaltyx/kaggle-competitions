# Blend v7 (probas brutes) + C1 (one-hot) — variantes d'injection
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Salty\Downloads\Predicting Student Health Risk"
CLASSES = ["at-risk", "fit", "unhealthy"]
C2I = {c: i for i, c in enumerate(CLASSES)}

base = pd.read_csv(os.path.join(ROOT, "blend-C1-vote-brut.csv"))
ref = base["health_condition"].map(C2I).values
N = len(ref)

P7 = np.load(os.path.join(ROOT, "kaggle-output-v7", "test_proba.npy"))
assert P7.shape == (N, 3), P7.shape

V = np.zeros((N, 3)); V[np.arange(N), ref] += 1.0
for w, name in [(0.20, "C16-v7-20"), (0.35, "C17-v7-35")]:
    mix = (1 - w) * V + w * P7
    pred = mix.argmax(1)
    nd = (pred != ref).sum()
    sub = base.copy(); sub["health_condition"] = [CLASSES[i] for i in pred]
    sub.to_csv(os.path.join(ROOT, f"blend-{name}.csv"), index=False)
    print(f"{name}: {nd} flips vs C1 | dist:",
          sub["health_condition"].value_counts(normalize=True).round(3).to_dict())
print("terminé")
