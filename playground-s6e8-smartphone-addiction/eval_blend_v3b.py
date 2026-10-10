# Evaluation honnete du blend v3b : endpoint nested (v3) + ancre diversite (notre v2)
# Style atakan : blend en percentile-rank, poids choisi sur OOF uniquement.
# numpy/pandas uniquement (pas de scipy/sklearn dans le runtime).
import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv', usecols=['addicted_label'])
y = train['addicted_label'].to_numpy()

oof_nested = np.load('out-v3/oof_nested.npy')   # OOF honnete du kernel v3
oof_v2 = np.load('out-v2/oof_stack.npy')        # OOF honnete de notre v2
test_nested = np.load('out-v3/test_nested.npy')
test_v2 = np.load('out-v2/test_stack.npy')

def pct(x):
    return pd.Series(x).rank(method='average').to_numpy() / len(x)

def auc(y_true, score):
    r = pd.Series(score).rank(method='average').to_numpy()  # rangs 1..n
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

r_n, r_2 = pct(oof_nested), pct(oof_v2)
print('AUC nested seul :', round(auc(y, oof_nested), 6))
print('AUC v2 seul     :', round(auc(y, oof_v2), 6))
print('spearman nested/v2:', round(np.corrcoef(r_n, r_2)[0, 1], 5))

best = (None, -1)
for w in np.arange(0.0, 0.51, 0.025):  # poids de l'ancre v2
    a = auc(y, (1 - w) * r_n + w * r_2)
    print(f'w_v2={w:.3f}  AUC OOF blend = {a:.6f}')
    if a > best[1]:
        best = (w, a)

w, a = best
print(f'\nmeilleur: w_v2={w:.3f}, AUC OOF = {a:.6f}')
if w > 0:
    sub = pd.read_csv('data/sample_submission.csv')
    blend_test = pct((1 - w) * pct(test_nested) + w * pct(test_v2))
    sub['addicted_label'] = blend_test
    sub.to_csv('out-v3/submission_v3b_blend.csv', index=False)
    print('ecrit out-v3/submission_v3b_blend.csv')
else:
    print("l'ancre v2 n'apporte rien en OOF -> on reste sur nested pur")
