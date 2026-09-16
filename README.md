# Retail Demand Forecasting & Inventory Optimization

Predicting product-level retail demand and translating those forecasts into
inventory decisions (safety stock, reorder points, service levels) using the
Corporación Favorita grocery sales dataset.

This project is a working proof-of-concept for the demand forecasting and
inventory optimization components of my AI-Powered Predictive Supply Chain
Resilience Framework — see [Methodology](#methodology) below.

## Problem

Retailers lose sales to stockouts and tie up cash in overstock because
demand forecasts are often built on intuition rather than evidence. This
project builds and evaluates predictive models against that problem on a
real, public retail dataset, then shows what the forecast improvement is
worth in inventory terms.

## Data

Corporación Favorita Grocery Sales Forecasting (Kaggle) — several years of
daily unit sales across thousands of products and dozens of stores in
Ecuador, plus store, item, holiday, oil-price, and transaction metadata.
See `data/README.md` for how to download it.

## Methodology

1. **Data consolidation & cleaning** — merge sales, store, item, holiday,
   and oil-price data; handle missing values and outliers; validate
   consistency.
2. **Forecasting models**, compared on the same holdout:
   - Naive / seasonal-naive baseline
   - Statistical model (SARIMA or Prophet)
   - Machine-learning model (XGBoost or LightGBM) with engineered features
     (lags, rolling stats, calendar/holiday effects, price/promo signals)
3. **Evaluation** — rolling-origin (walk-forward) backtesting, scored on
   MAPE, WMAPE, and bias, so accuracy is measured the way it would hold up
   in production, not on a single lucky train/test split.
4. **Inventory optimization** — convert forecast + forecast error into
   safety stock and reorder points at a target service level, then compare
   projected stockout and overstock cost against a naive-forecast baseline.
5. **Reporting** — a results dashboard (rebuilt in Tableau) and a written
   summary of what the better forecast is worth in dollar terms.

## Results

_To be filled in as the project progresses._

| Model | MAPE | WMAPE | Bias |
|---|---|---|---|
| Naive baseline | — | — | — |
| SARIMA / Prophet | — | — | — |
| XGBoost / LightGBM | — | — | — |

Estimated inventory cost impact vs. baseline: _TBD_

## Project structure

```
data/           raw (not committed) and processed data
notebooks/      exploratory analysis
src/            reusable pipeline code (data prep, models, inventory, evaluation)
reports/        figures and written results
tests/          unit tests for src/
```

## Setup

```
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Author

Mithila Zaman Samapti — Supply Chain Analytics & Procurement Manager
