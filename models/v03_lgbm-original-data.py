import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = tr.pop("satisfaction").astype(int)

# ORIGINAL DATA: THE REAL SURVEY THE SYNTHETIC SET WAS GENERATED FROM (129,880 ROWS); SAME COLUMNS, LABELS MAPPED
og = pd.concat([pd.read_csv("data/original/train.csv"), pd.read_csv("data/original/test.csv")])
yo = (og.pop("satisfaction") == "satisfied").astype(int)
og = og[tr.columns.drop("id")]

# FEATURES: SAME AS v00-v02, BUILT ON ALL ROWS TOGETHER SO THE ONE-HOT COLUMNS MATCH
X = pd.concat([tr.drop(columns="id"), te.drop(columns="id"), og])
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
X = pd.get_dummies(X, drop_first=True, dtype=float)
n, k = len(tr), len(te)
Xtr, Xte, Xog = X[:n], X[n:n + k], X[n + k:]

# MODEL: v02's TUNED PARAMETERS, UNCHANGED
P = dict(num_leaves=156, min_child_samples=27, subsample=0.994, subsample_freq=1, colsample_bytree=0.785, reg_lambda=1.771)

# CV: EACH FOLD TRAINS ON 4/5 SYNTHETIC + ALL ORIGINAL, SCORES ON 1/5 SYNTHETIC ONLY
aucs, pred = [], np.zeros(k)
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y):
    m = LGBMClassifier(n_estimators=10000, learning_rate=0.03, verbose=-1, **P)
    m.fit(pd.concat([Xtr.iloc[a], Xog]), pd.concat([y.iloc[a], yo]), eval_set=[(Xtr.iloc[b], y.iloc[b])],
          eval_metric="auc", callbacks=[early_stopping(100, verbose=False)])
    aucs.append(m.best_score_["valid_0"]["auc"])
    pred += m.predict_proba(Xte)[:, 1] / 5
print(f"CV AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
pd.DataFrame({"id": te["id"], "satisfaction": pred}).to_csv("submissions/v03.csv", index=False)
