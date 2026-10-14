# Experiments

| id | parent | one change | CV AUC | LB AUC | keep? |
|---|---|---|---|---|---|
| v00 | — | logistic regression baseline (one-hot, scaled, arrival delay filled from departure delay) | 0.9216 | 0.9223 | ✅ |
| v01 | v00 | LightGBM, all defaults (same features) | 0.9576 | — | ✅ |
| v02 | v01 | LightGBM tuned: optuna 30 trials (num_leaves 156, min_child 27, subsample 0.99, colsample 0.79, lambda 1.77), lr 0.03, 5-fold avg | 0.9590 | 0.9584 | ✅ |
| v03 | v02 | + original dataset (129,880 real rows) in training only; CV on synthetic only | 0.9587 | 0.9580 | ❌ |
| v04 | v02 | 13 service ratings as LightGBM categories (not numbers) | 0.9589 | 0.9584 | ❌ (no change) |
