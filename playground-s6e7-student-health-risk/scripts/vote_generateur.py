"""Analyse du vote générateur (28/07) — piste 0.953+ explorée et fermée.

Hypothèse : le label est quasi-déterminé par (sleep_duration, stress_level,
physical_activity_level). Un vote majoritaire des lignes train partageant la
même clé pourrait prédire le test.

Résultats :
- Heldout train (80/20) : BA = 0.9704, couverture 74 % -> signal RÉEL sur train.
- Accord avec C1 sur test : 99.53 % -> C1 a déjà absorbé le signal.
- C25 = C1 + 95 flips haute confiance (pureté >= 97 %, n >= 30 doublons) :
  LB 0.95195 < C1 0.95267. Les flips étaient majoritairement FAUX.
- Conclusion : sur test, les clés génératrices sont décorrélées du label
  (rééchantillonnage bruité) ; le vote ne transfère pas. Piste FERMÉE.
- Combo de clés optimal = les 3 variables ensemble (en retirer une effondre le BA).
"""
import pandas as pd
import numpy as np

GEN = ["sleep_duration", "stress_level", "physical_activity_level"]
T = "health_condition"


def balanced_acc(y_true, y_pred):
    classes = pd.unique(y_true)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in classes]))


def main():
    train = pd.read_csv("comp-data/train.csv")
    test = pd.read_csv("comp-data/test.csv")
    c1 = pd.read_csv("blend-C1-vote-brut.csv")

    # 1) Validation heldout sur train
    tr_s = train.sample(frac=1, random_state=42).reset_index(drop=True)
    cut = int(len(tr_s) * 0.8)
    tr_a, tr_b = tr_s.iloc[:cut], tr_s.iloc[cut:]
    major_a = tr_a.groupby(GEN)[T].agg(lambda s: s.value_counts().idxmax()).rename("vote")
    hb = tr_b.merge(major_a, on=GEN, how="left")
    m = hb["vote"].notna()
    print(f"Heldout train : BA={balanced_acc(hb.loc[m, T], hb.loc[m, 'vote']):.4f} "
          f"couverture={m.mean():.3f}")

    # 2) Vote full-train appliqué au test, comparé à C1
    g = train.groupby(GEN)[T]
    stats = pd.concat([
        g.agg(lambda s: s.value_counts().idxmax()).rename("gen_label"),
        g.agg(lambda s: s.value_counts(normalize=True).iloc[0]).rename("gen_conf"),
        train.groupby(GEN).size().rename("gen_n"),
    ], axis=1)
    tt = test.merge(stats, on=GEN, how="left").merge(c1, on="id")
    matched = tt["gen_label"].notna()
    print(f"Test matché : {matched.sum()}/{len(tt)} | "
          f"accord C1={(tt.loc[matched, 'gen_label'] == tt.loc[matched, T]).mean():.4f}")

    # 3) C25 : flips haute confiance uniquement (LB 0.95195 — ne pas réutiliser)
    flip = matched & (tt["gen_label"] != tt[T]) & (tt["gen_conf"] >= 0.97) & (tt["gen_n"] >= 30)
    sub = c1.copy()
    fmap = dict(zip(tt.loc[flip, "id"], tt.loc[flip, "gen_label"]))
    sub.loc[sub["id"].isin(fmap), T] = sub["id"].map(fmap).fillna(sub[T])
    sub.to_csv("blend-C25-vote-generateur.csv", index=False)
    print(f"C25 écrit : {len(fmap)} flips vs C1 (LB mesuré : 0.95195, perte -0.0007)")


if __name__ == "__main__":
    main()
