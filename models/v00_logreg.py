import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = tr.pop("satisfaction").astype(int)

# FEATURES: FILL MISSING ARRIVAL DELAY WITH DEPARTURE DELAY, ONE-HOT THE CATEGORICALS
X = pd.concat([tr, te]).drop(columns="id")
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
X = pd.get_dummies(X, drop_first=True, dtype=float)
Xtr, Xte = X[: len(tr)], X[len(tr):]

# MODEL: SCALE + LOGISTIC REGRESSION, 5-FOLD STRATIFIED CV AUC
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = cross_val_score(m, Xtr, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
print(f"CV AUC {cv.mean():.4f} ± {cv.std():.4f}")

# SUBMISSION: FIT ON ALL TRAIN, PREDICT TEST PROBABILITIES
m.fit(Xtr, y)
pd.DataFrame({"id": te["id"], "satisfaction": m.predict_proba(Xte)[:, 1]}).to_csv("submissions/v00.csv", index=False)
