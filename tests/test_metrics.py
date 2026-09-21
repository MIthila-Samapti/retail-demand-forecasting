import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.models import evaluate, seasonal_naive_forecast
import pandas as pd


def test_evaluate_perfect_forecast_is_zero_error():
    y_true = np.array([100.0, 120.0, 90.0])
    metrics = evaluate(y_true, y_true.copy())
    assert metrics["MAPE"] == 0
    assert metrics["WMAPE"] == 0
    assert metrics["bias"] == 0


def test_evaluate_known_percentage_error():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])  # +10% and -10%, cancels out in bias
    metrics = evaluate(y_true, y_pred)
    assert round(metrics["MAPE"], 4) == 0.10
    assert round(metrics["bias"], 4) == 0.0


def test_evaluate_bias_direction():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 110.0])  # consistently over-forecasting
    metrics = evaluate(y_true, y_pred)
    assert metrics["bias"] > 0  # positive bias = over-forecasting


def test_seasonal_naive_repeats_last_week():
    history = pd.Series([10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71])
    forecast = seasonal_naive_forecast(history, horizon=7, season=7)
    assert list(forecast) == [11, 21, 31, 41, 51, 61, 71]
