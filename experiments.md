# Experiments

| id | parent | one change | CV AUC | LB AUC | keep? |
|---|---|---|---|---|---|
| v00 | — | logistic regression baseline (one-hot, scaled, arrival delay filled from departure delay) | 0.9216 | 0.9223 | ✅ |
| v01 | v00 | LightGBM, all defaults (same features) | 0.9576 | — | ✅ |
