"""
Rossmann Store Sales - Prediction Dashboard (Task 3)
-----------------------------------------------------
A Streamlit web app that lets a store manager:
  1. Enter a Store ID and a few parameters (promo, holiday, weekend, date range)
     OR upload a CSV with a Date/IsHoliday/IsWeekend/IsPromo column set.
  2. Get predicted Sales (and, if the notebook's Customers-aware pipeline was
     used, an approximate Customers estimate) for each requested date.
  3. See a plot of the predictions.
  4. Download the predictions as a CSV.

Run with:
    streamlit run app.py

The app looks for the most recent timestamped model file produced by the
notebook (models/rossmann_rf_<timestamp>.pkl). If no model is found it runs
in DEMO MODE with a simple heuristic so the UI can still be shown end-to-end.
"""

import glob
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

MODEL_DIR = "models"


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_latest_model():
    """Load the most recently saved timestamped model, if one exists."""
    if not os.path.isdir(MODEL_DIR):
        return None
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "rossmann_rf_*.pkl")))
    if not candidates:
        return None
    latest = candidates[-1]
    bundle = joblib.load(latest)
    bundle["path"] = latest
    return bundle


model_bundle = load_latest_model()
DEMO_MODE = model_bundle is None


# ----------------------------------------------------------------------
# Feature engineering (must mirror the notebook's clean_and_engineer)
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df["IsWeekend"] = df["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsMonthMid"] = df["Day"].between(11, 20).astype(int)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run the real model if available, otherwise a transparent demo heuristic."""
    feats = engineer_features(df)

    if not DEMO_MODE:
        pipeline = model_bundle["pipeline"]
        feature_cols = model_bundle["feature_columns"]
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = np.nan
        X = feats[feature_cols]
        preds = np.maximum(pipeline.predict(X), 0)
    else:
        # Simple, clearly-labelled placeholder so the dashboard is demonstrable
        # even before a trained model file exists.
        base = 6000
        promo_lift = feats.get("IsPromo", 0).astype(int) * 1800
        weekend_drop = feats["IsWeekend"] * -900
        holiday_drop = feats.get("IsHoliday", 0).astype(int) * -2500
        noise = np.random.default_rng(42).normal(0, 150, size=len(feats))
        preds = np.maximum(base + promo_lift + weekend_drop + holiday_drop + noise, 0)

    feats["PredictedSales"] = preds
    feats["PredictedCustomers"] = (preds / 9.0).round().astype(int)  # rough sales/customer ratio
    return feats


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏪 Rossmann Store Sales Forecast Dashboard")
st.caption("NextHikes / Rossmann Pharmaceuticals — 6-week sales forecasting tool for store managers.")

if DEMO_MODE:
    st.warning(
        "No trained model file was found in `models/`. Running in **DEMO MODE** with a "
        "placeholder formula so you can preview the dashboard. Run the notebook's "
        "modelling section first to save a real `rossmann_rf_*.pkl` model here."
    )
else:
    st.success(f"Loaded model: `{model_bundle['path']}`")

tab1, tab2 = st.tabs(["🔢 Manual Input", "📄 Upload CSV"])

with tab1:
    st.subheader("Enter parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
        promo = st.selectbox("Promo running?", ["No", "Yes"])
    with col2:
        start_date = st.date_input("Forecast start date", datetime.today())
        horizon = st.slider("Forecast horizon (days, up to 6 weeks)", 1, 42, 14)
    with col3:
        is_holiday = st.selectbox("State holiday?", ["No", "Yes"])
        is_school_holiday = st.selectbox("School holiday?", ["No", "Yes"])

    if st.button("Generate Forecast", type="primary"):
        dates = [start_date + timedelta(days=i) for i in range(horizon)]
        input_df = pd.DataFrame({
            "Date": dates,
            "Store": store_id,
            "IsPromo": 1 if promo == "Yes" else 0,
            "IsHoliday": 1 if is_holiday == "Yes" else 0,
            "SchoolHoliday": 1 if is_school_holiday == "Yes" else 0,
            "Open": 1,
        })
        result = predict(input_df)

        st.line_chart(result.set_index("Date")[["PredictedSales", "PredictedCustomers"]])

        display_cols = ["Date", "Store", "PredictedSales", "PredictedCustomers"]
        st.dataframe(result[display_cols], use_container_width=True)

        csv_bytes = result[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download predictions as CSV",
            data=csv_bytes,
            file_name=f"store_{store_id}_forecast.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("Upload a CSV with future dates")
    st.write(
        "Required columns: `Date`. Optional columns: `Store`, `IsHoliday`, `IsWeekend`, "
        "`IsPromo`, `SchoolHoliday`. Missing optional columns default to 0."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        if "Date" not in raw.columns:
            st.error("The uploaded file must contain a 'Date' column.")
        else:
            for col, default in [("Store", store_id if "store_id" in dir() else 1),
                                  ("IsHoliday", 0), ("IsPromo", 0), ("SchoolHoliday", 0), ("Open", 1)]:
                if col not in raw.columns:
                    raw[col] = default

            result = predict(raw)
            st.line_chart(result.set_index("Date")[["PredictedSales", "PredictedCustomers"]])

            display_cols = ["Date", "Store", "PredictedSales", "PredictedCustomers"]
            st.dataframe(result[display_cols], use_container_width=True)

            csv_bytes = result[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download predictions as CSV",
                data=csv_bytes,
                file_name="uploaded_forecast.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Model: RandomForestRegressor inside a scikit-learn Pipeline, trained in "
    "Rossmann_Store_Sales_Complete_Notebook.ipynb. Deploy this app with "
    "`streamlit run app.py`, or containerize it with the provided Dockerfile."
)
