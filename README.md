# Playground S6E10: Predicting Airline Satisfaction

- **Competition:** https://www.kaggle.com/competitions/playground-series-s6e10
- **Task:** binary classification. For each `id`, predict the probability of `satisfaction`.
- **Metric:** ROC AUC.
- **Submission:** `id,satisfaction` (a probability per test id), with a header row.
- **Deadline:** 2026-10-31 23:59 UTC.
- **Data:** synthetic, generated from a real airline-satisfaction dataset.
  - `train.csv`: 67 MB
  - `test.csv`: 27 MB
  - `sample_submission.csv`: 8 MB

## Layout
- `data/`: competition files (not committed).
- `models/`: one script per version, `vNN_slug.py` (e.g. `v01_lgbm-baseline.py`).
- `submissions/`: one file per version, `vNN.csv` (not committed).
- `experiments.md`: one row per version.

## Versions
- **One change per version**, built on a named parent.
- **One ID everywhere:** model `models/v03_target-encoding.py`, submission `submissions/v03.csv`, Kaggle message `v03 | target encoding | CV 0.xxxx`.

## Get the data (after joining the competition on Kaggle)
```bash
cd data && kaggle competitions download -c playground-series-s6e10 -p . && unzip -o -q *.zip && rm *.zip
```
