import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = tr.pop("satisfaction").astype(int)

# FEATURES: SAME AS v02, PLUS THE 14 SERVICE RATINGS (0-5) AS CATEGORIES, NOT NUMBERS
# (A 0 OFTEN MEANS "NOT USED", SO THE VALUES NEED NOT BE ORDERED; LIGHTGBM GROUPS CATEGORY VALUES FREELY)
X = pd.concat([tr, te]).drop(columns="id")
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
X = pd.get_dummies(X, drop_first=True, dtype=float)
R = [c for c in tr.columns.drop("id") if c in X and X[c].between(0, 5).all() and X[c].nunique() <= 6]
X[R] = X[R].astype(int).astype("category")
n, k = len(tr), len(te)
Xtr, Xte = X[:n], X[n:]

# MODEL: v02's TUNED PARAMETERS, UNCHANGED
P = dict(num_leaves=156, min_child_samples=27, subsample=0.994, subsample_freq=1, colsample_bytree=0.785, reg_lambda=1.771)

# CV: SAME 5 FOLDS AS v02
aucs, pred = [], np.zeros(k)
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y):
    m = LGBMClassifier(n_estimators=10000, learning_rate=0.03, verbose=-1, **P)
    m.fit(Xtr.iloc[a], y.iloc[a], eval_set=[(Xtr.iloc[b], y.iloc[b])],
          eval_metric="auc", callbacks=[early_stopping(100, verbose=False)])
    aucs.append(m.best_score_["valid_0"]["auc"])
    pred += m.predict_proba(Xte)[:, 1] / 5
print(f"{len(R)} rating columns as categories")
print(f"CV AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
pd.DataFrame({"id": te["id"], "satisfaction": pred}).to_csv("submissions/v04.csv", index=False)
