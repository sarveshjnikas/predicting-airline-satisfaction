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
- `notebooks/`: your work.
- `submissions/`: submission CSVs.

## Get the data (after joining the competition on Kaggle)
```bash
cd data && kaggle competitions download -c playground-series-s6e10 -p . && unzip -o -q *.zip && rm *.zip
```
