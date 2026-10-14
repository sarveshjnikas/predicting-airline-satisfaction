import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = tr.pop("satisfaction").astype(int)

# FEATURES: SAME AS v00/v01
X = pd.concat([tr, te]).drop(columns="id")
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
X = pd.get_dummies(X, drop_first=True, dtype=float)
Xtr, Xte = X[: len(tr)], X[len(tr):]


def run(p, lr, k, test=False):
    """k-FOLD CV WITH EARLY STOPPING; RETURNS MEAN AUC (+ AVERAGED TEST PREDICTIONS)"""
    aucs, pred = [], np.zeros(len(Xte))
    for a, b in StratifiedKFold(k, shuffle=True, random_state=0).split(Xtr, y):
        m = LGBMClassifier(n_estimators=10000, learning_rate=lr, verbose=-1, **p)
        m.fit(Xtr.iloc[a], y.iloc[a], eval_set=[(Xtr.iloc[b], y.iloc[b])], eval_metric="auc",
              callbacks=[early_stopping(100, verbose=False)])
        aucs.append(m.best_score_["valid_0"]["auc"])
        if test: pred += m.predict_proba(Xte)[:, 1] / k
    return np.mean(aucs), pred


# TUNE: TREE SIZE + RANDOMNESS + PENALTY, FAST (LR 0.1, 3 FOLDS)
def objective(t):
    p = dict(num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
             min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
             subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    return run(p, 0.1, 3)[0]

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
best = study.best_params | {"subsample_freq": 1}
print("best (3-fold, lr 0.1):", round(study.best_value, 4), best)

# FINAL: BEST PARAMS, SMALLER LR, 5 FOLDS, TEST = MEAN OF THE 5 FOLD MODELS
auc, pred = run(best, 0.03, 5, test=True)
print(f"CV AUC {auc:.4f}")
pd.DataFrame({"id": te["id"], "satisfaction": pred}).to_csv("submissions/v02.csv", index=False)
