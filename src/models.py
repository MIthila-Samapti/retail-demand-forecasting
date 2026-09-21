"""
Forecasting models and rolling-origin backtesting.

Three approaches are compared on the same target (daily unit sales per
product family, summed across the 10 scoped stores):

1. Seasonal-naive baseline  -- "demand next Tuesday = demand last Tuesday"
2. Prophet                  -- additive time-series model with weekly/yearly
                                seasonality plus oil-price, promotion, and
                                holiday regressors
3. XGBoost                  -- a single global gradient-boosted model
                                trained jointly across all families, using
                                lag/rolling-window features plus calendar,
                                promotion, and oil-price signals

All three are evaluated the same way: rolling-origin (walk-forward)
backtesting, so accuracy reflects what each model would have achieved
forecasting forward in time, not a single lucky train/test split.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def make_family_panel(store_family_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the store x family x day panel down to family x day, summed
    across the 10 scoped stores, with every calendar day present.

    Important: the raw Kaggle data only includes rows where a sale
    happened, so a missing date for a family isn't missing data -- it's
    zero units sold that day. We reindex to the full date range and fill
    with 0 so the models see true zero-demand days instead of gaps.
    """
    df = store_family_df.copy()
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    out = []
    for family, g in df.groupby("family"):
        daily = (
            g.groupby("date")
            .agg(
                unit_sales=("unit_sales", "sum"),
                n_items_on_promo=("n_items_on_promo", "sum"),
                oil_price=("oil_price", "mean"),
                is_holiday=("is_holiday", "max"),
            )
            .reindex(full_dates)
        )
        daily["unit_sales"] = daily["unit_sales"].fillna(0.0)
        daily["n_items_on_promo"] = daily["n_items_on_promo"].fillna(0.0)
        daily["oil_price"] = daily["oil_price"].ffill().bfill()
        daily["is_holiday"] = daily["is_holiday"].fillna(False).astype(bool)
        daily["family"] = family
        daily.index.name = "date"
        out.append(daily.reset_index())

    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Model 1: seasonal-naive baseline
# ---------------------------------------------------------------------------

def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last `season`-day pattern forward for `horizon` days."""
    last_season = history.values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


# ---------------------------------------------------------------------------
# Model 2: Prophet
# ---------------------------------------------------------------------------

def prophet_forecast(train: pd.DataFrame, future_regressors: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    train: columns ds, y, oil_price, n_items_on_promo, is_holiday
    future_regressors: the same regressor columns for the `horizon` days
    being forecast (in a real deployment these would be your own
    forecasts/plans for promotions, holidays are known in advance, and
    oil price would need its own forecast -- here we use the realized
    values, which is standard practice for backtesting a model's demand-
    forecasting skill in isolation from a separate regressor-forecasting
    problem).
    """
    from prophet import Prophet

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    for reg in ["oil_price", "n_items_on_promo", "is_holiday"]:
        m.add_regressor(reg)

    fit_df = train.copy()
    fit_df["is_holiday"] = fit_df["is_holiday"].astype(int)
    m.fit(fit_df[["ds", "y", "oil_price", "n_items_on_promo", "is_holiday"]])

    future = future_regressors.copy()
    future["is_holiday"] = future["is_holiday"].astype(int)
    fc = m.predict(future[["ds", "oil_price", "n_items_on_promo", "is_holiday"]])
    return fc["yhat"].clip(lower=0).values[:horizon]


# ---------------------------------------------------------------------------
# Model 3: XGBoost (global model across all families)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "dow", "month", "is_holiday", "n_items_on_promo", "oil_price",
]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling/calendar features, computed separately per family
    so no family's history leaks into another's lags."""
    df = panel.sort_values(["family", "date"]).copy()
    g = df.groupby("family")["unit_sales"]

    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    for win in (7, 14, 28):
        df[f"roll_mean_{win}"] = g.shift(1).rolling(win).mean().reset_index(level=0, drop=True)
    df["roll_std_7"] = g.shift(1).rolling(7).std().reset_index(level=0, drop=True)

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["family_code"] = df["family"].astype("category").cat.codes
    return df


def train_xgboost(train_features: pd.DataFrame):
    from xgboost import XGBRegressor

    cols = FEATURE_COLS + ["family_code"]
    train_features = train_features.dropna(subset=cols)
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train_features[cols], train_features["unit_sales"])
    return model


def xgboost_forecast(model, panel: pd.DataFrame, family: str, cutoff_date, horizon: int) -> np.ndarray:
    """
    Recursive multi-step forecast: predict one day ahead, append that
    prediction to the working history, recompute lag/rolling features,
    predict the next day -- repeated for `horizon` days. This mirrors how
    the model would actually be used in production.
    """
    cols = FEATURE_COLS + ["family_code"]
    hist = panel[(panel["family"] == family) & (panel["date"] <= cutoff_date)].copy()
    future_dates = pd.date_range(cutoff_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_known = panel[(panel["family"] == family) & (panel["date"].isin(future_dates))].set_index("date")

    preds = []
    working = hist.copy()
    for d in future_dates:
        row = {
            "date": d, "family": family,
            "n_items_on_promo": future_known.loc[d, "n_items_on_promo"] if d in future_known.index else 0,
            "oil_price": future_known.loc[d, "oil_price"] if d in future_known.index else working["oil_price"].iloc[-1],
            "is_holiday": future_known.loc[d, "is_holiday"] if d in future_known.index else False,
            "unit_sales": np.nan,
        }
        working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)
        feat = add_features(working)
        last = feat.iloc[[-1]].copy()
        last["family_code"] = working["family"].astype("category").cat.codes.iloc[-1]
        pred = max(0.0, float(model.predict(last[cols])[0]))
        preds.append(pred)
        working.loc[working.index[-1], "unit_sales"] = pred

    return np.array(preds)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    wmape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-9)

    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
        if nonzero.any() else np.nan
    )

    bias = np.sum(y_pred - y_true) / max(np.sum(np.abs(y_true)), 1e-9)

    return {"MAPE": mape, "WMAPE": wmape, "bias": bias}


# ---------------------------------------------------------------------------
# Rolling-origin backtest orchestration
# ---------------------------------------------------------------------------

def generate_backtest_predictions(panel: pd.DataFrame, horizon: int = 28, n_folds: int = 4) -> pd.DataFrame:
    """
    Walk-forward backtest: n_folds non-overlapping `horizon`-day test
    windows at the end of the series, each with an expanding training
    window (everything before that window). For each fold, each family,
    and each model, forecast the window day by day.

    Returns one row per (fold, family, model, date) with the actual and
    predicted value -- the day-level detail that both the accuracy
    metrics and the inventory simulation (Phase 3) are built from.
    """
    max_date = panel["date"].max()
    families = sorted(panel["family"].unique())
    rows = []

    for fold in range(n_folds):
        cutoff = max_date - pd.Timedelta(days=horizon * (n_folds - fold))
        window_end = cutoff + pd.Timedelta(days=horizon)

        train_all = panel[panel["date"] <= cutoff]
        feat_all = add_features(train_all)
        xgb_model = train_xgboost(feat_all)

        for family in families:
            hist = train_all[train_all["family"] == family]
            actual_df = panel[
                (panel["family"] == family) & (panel["date"] > cutoff) & (panel["date"] <= window_end)
            ].sort_values("date")
            if len(actual_df) < horizon:
                continue
            y_true = actual_df["unit_sales"].values
            dates = actual_df["date"].values

            base_pred = seasonal_naive_forecast(hist.set_index("date")["unit_sales"], horizon)
            train_p = hist.rename(columns={"date": "ds", "unit_sales": "y"})
            future_p = actual_df.rename(columns={"date": "ds"})
            proph_pred = prophet_forecast(train_p, future_p, horizon)
            xgb_pred = xgboost_forecast(xgb_model, panel, family, cutoff, horizon)

            for model_name, y_pred in [
                ("Seasonal naive", base_pred),
                ("Prophet", proph_pred),
                ("XGBoost", xgb_pred),
            ]:
                for day_offset, (d, actual, pred) in enumerate(zip(dates, y_true, y_pred), start=1):
                    rows.append({
                        "fold": fold, "family": family, "model": model_name,
                        "date": d, "day_offset": day_offset,
                        "actual": actual, "predicted": pred,
                    })

    return pd.DataFrame(rows)


def run_backtest(panel: pd.DataFrame, horizon: int = 28, n_folds: int = 4) -> pd.DataFrame:
    """Aggregate MAPE/WMAPE/bias per (fold, family, model), built on top
    of generate_backtest_predictions so metrics and simulation always
    agree with each other."""
    detailed = generate_backtest_predictions(panel, horizon, n_folds)
    rows = []
    for (fold, family, model), g in detailed.groupby(["fold", "family", "model"]):
        metrics = evaluate(g["actual"].values, g["predicted"].values)
        rows.append({"fold": fold, "family": family, "model": model, **metrics})
    return pd.DataFrame(rows)
