"""
Inventory optimization: turns a demand forecast (and its error) into a
safety stock and reorder point, then simulates a simple replenishment
policy to quantify stockouts and excess inventory against actual demand.

Note on costs: Favorita has no price/cost data, so holding and
stockout costs below use illustrative per-unit dollar figures, not
real ones - flagged wherever a dollar figure shows up. Point is to
demonstrate the method (forecast -> safety stock -> reorder point ->
simulated cost); same method applies once real unit costs go in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Illustrative unit economics (documented, not fitted to real data)
HOLDING_COST_PER_UNIT_PER_DAY = 0.05
STOCKOUT_COST_PER_UNIT = 2.00

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95


def compute_daily_error_std(detailed: pd.DataFrame) -> pd.DataFrame:
    # std of (actual - predicted) per family+model across all backtest
    # days - real out-of-sample error, not a training residual, so it
    # reflects actual production uncertainty
    g = detailed.copy()
    g["error"] = g["actual"] - g["predicted"]
    return (
        g.groupby(["family", "model"])["error"]
        .std()
        .rename("sigma_daily")
        .reset_index()
    )


def safety_stock(sigma_daily: float, lead_time: int = LEAD_TIME_DAYS, service_level: float = SERVICE_LEVEL) -> float:
    # classic newsvendor safety stock: z * sigma_L, sigma_L scaled by
    # sqrt(lead_time) assuming independent day-to-day errors. standard
    # textbook simplification - real demand errors are often
    # autocorrelated, which would push the true number higher. limitation,
    # not hidden.
    z = norm.ppf(service_level)
    return max(0.0, z * sigma_daily * np.sqrt(lead_time))


def reorder_point(avg_daily_forecast: float, ss: float, lead_time: int = LEAD_TIME_DAYS) -> float:
    # expected demand over the lead time, plus the safety buffer
    return avg_daily_forecast * lead_time + ss


def simulate_policy(
    dates: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    lead_time: int = LEAD_TIME_DAYS,
    service_level: float = SERVICE_LEVEL,
) -> dict:
    # daily-review, order-up-to-reorder-point policy over one family's
    # backtest window: each day, receive any order placed lead_time
    # days ago, fill demand from on-hand (shortfall = stockout), then if
    # inventory position (on-hand + on-order) is at or below the reorder
    # point, order back up to it. uses the model's own forecasts to set
    # the reorder point, tested against what actually happened.
    sigma_daily = float(np.std(actual - forecast, ddof=1)) if len(actual) > 1 else 0.0
    ss = safety_stock(sigma_daily, lead_time, service_level)
    avg_daily_forecast = float(np.mean(forecast))
    rop = reorder_point(avg_daily_forecast, ss, lead_time)

    n = len(actual)
    on_hand = rop  # start at the target level
    pipeline = {}  # arrival_day -> quantity
    stockout_units = 0.0
    total_demand = 0.0
    on_hand_history = []

    for day in range(n):
        if day in pipeline:
            on_hand += pipeline.pop(day)

        demand = actual[day]
        total_demand += demand
        shortfall = max(0.0, demand - on_hand)
        stockout_units += shortfall
        on_hand = max(0.0, on_hand - demand)

        on_order = sum(pipeline.values())
        inventory_position = on_hand + on_order
        if inventory_position <= rop:
            order_qty = max(0.0, rop - inventory_position)
            arrival_day = day + lead_time
            pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order_qty

        on_hand_history.append(on_hand)

    avg_on_hand = float(np.mean(on_hand_history))
    fill_rate = 1 - (stockout_units / total_demand if total_demand > 0 else 0.0)
    holding_cost = avg_on_hand * HOLDING_COST_PER_UNIT_PER_DAY * n
    stockout_cost = stockout_units * STOCKOUT_COST_PER_UNIT
    total_cost = holding_cost + stockout_cost

    return {
        "safety_stock": ss,
        "reorder_point": rop,
        "avg_on_hand": avg_on_hand,
        "total_demand": total_demand,
        "stockout_units": stockout_units,
        "fill_rate": fill_rate,
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "total_cost": total_cost,
    }


def run_inventory_simulation(detailed: pd.DataFrame, lead_time: int = LEAD_TIME_DAYS, service_level: float = SERVICE_LEVEL) -> pd.DataFrame:
    # simulate_policy for every family+model, driven by that model's
    # full backtest-window forecasts (all folds concatenated, date order)
    rows = []
    for (family, model), g in detailed.sort_values(["family", "model", "date"]).groupby(["family", "model"]):
        result = simulate_policy(
            g["date"].values, g["actual"].values, g["predicted"].values,
            lead_time=lead_time, service_level=service_level,
        )
        rows.append({"family": family, "model": model, **result})
    return pd.DataFrame(rows)
