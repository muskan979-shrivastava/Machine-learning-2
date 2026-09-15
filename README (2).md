# Rossmann Store Sales Forecasting

End-to-end sales forecasting project for Rossmann Pharmaceuticals (NextHikes IT Solutions
internship challenge): EDA, a Random Forest regression pipeline, an LSTM deep-learning
model, MLflow tracking, and a Streamlit dashboard for serving predictions.

## Project structure

```
.
├── Rossmann_Store_Sales_Complete_Notebook.ipynb   # Task 1 & Task 2 (EDA, ML, DL)
├── app.py                                         # Task 3: Streamlit prediction dashboard
├── requirements.txt
├── data/
│   └── raw/            # train.csv, test.csv, store.csv, sample_submission.csv
├── models/             # timestamped .pkl models saved by the notebook
└── reports/            # submission.csv, plots, etc.
```

## How to run

1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put `train.csv`, `test.csv`, `store.csv`, `sample_submission.csv` in `data/raw/`.
3. Run the notebook top to bottom. It will:
   - Perform the Task 1 EDA (promotions, holidays, seasonality, assortment,
     competition distance, weekday-opening stores, competitor-opening impact).
   - Engineer features, train a Random Forest pipeline, evaluate with
     MAE/RMSE/R², compute feature importance and a 95% prediction interval.
   - Save a timestamped model to `models/rossmann_rf_<timestamp>.pkl`.
   - Build and evaluate a 2-layer LSTM on the aggregated daily sales series.
   - Log the Random Forest run to MLflow (`mlflow ui --port 5000` to view it).
   - Write `reports/submission.csv` for the competition test set.
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
   The app auto-loads the newest model in `models/`. If none exists yet it runs
   in a clearly-labelled demo mode so the UI can still be reviewed.

## What's done vs. still outstanding

**Done (in the notebook):**
- Task 1: full EDA answering every question in the brief, including the two
  that were previously missing — "which stores open on all weekdays and how
  does that affect weekend sales" and "impact of a competitor opening during
  the training window" (plus a proximity-based proxy for the city-centre
  question, since the dataset has no location field).
- Task 2.1–2.7: feature engineering, sklearn pipeline, Random Forest model,
  loss-function justification (MAE/RMSE/R²), feature importance, prediction
  confidence interval, timestamped model serialization, 2-layer LSTM with
  stationarity/ACF-PACF checks and sliding-window scaling, MLflow logging.

**Added in this pass:**
- `app.py` — the Task 3 Streamlit dashboard (manual input + CSV upload,
  prediction plot, CSV download).
- `requirements.txt`, this `README.md`.

**Still needed from you (outside notebook scope):**
- Push the repo to GitHub (with the notebook, `app.py`, this README).
- Set up DVC for data versioning and take screenshots of multiple data
  versions, as required by the "to be categorised" submission checklist.
- Take MLflow dashboard screenshots showing multiple model runs (run the
  notebook a few times with different hyperparameters to generate these).
- Deploy `app.py` (Heroku, Streamlit Community Cloud, Render, etc.) and grab
  the live link or a screenshot.
- 3–5 slide interim presentation summarizing the Task 1 findings.
- Final PDF "blog-style" report (with extra emphasis on the deep learning
  section) due **15 Sep 2026**.
- Optional but recommended: a `Dockerfile`, a `tests/` folder, and a
  `.github/workflows/ci.yml` for CI, since these were flagged in the
  notebook's own checklist and score well under "Quality of code".
