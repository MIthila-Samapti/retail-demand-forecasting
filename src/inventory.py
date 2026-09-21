"""
Inventory optimization: turns a demand forecast (and its error) into a
safety stock and reorder point, then simulates a simple replenishment
policy to quantify stockouts and excess inventory against actual demand.

Note on costs: the Favorita dataset has no price/cost data, so holding
and stockout costs below use illustrative per-unit dollar figures rather
than real ones. That's stated explicitly wherever a dollar figure
appears -- the point is to demonstrate the method (forecast -> safety
stock -> reorder point -> simulated cost), which is the same method
you'd apply once real unit costs are plugged in.
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
    """
    Std deviation of (actual - predicted) per family+model, computed
    across all backtest days. This is the model's real, out-of-sample
    forecast error -- not a training-set residual -- so it reflects how
    uncertain the forecast actually is in production.
    """
    g = detailed.copy()
    g["error"] = g["actual"] - g["predicted"]
    return (
        g.groupby(["family", "model"])["error"]
        .std()
        .rename("sigma_daily")
        .reset_index()
    )


def safety_stock(sigma_daily: float, lead_time: int = LEAD_TIME_DAYS, service_level: float = SERVICE_LEVEL) -> float:
    """
    Classic newsvendor-style safety stock: z * sigma_L, where sigma_L is
    the standard deviation of demand *over the lead time*. Assuming
    day-to-day forecast errors are roughly independent, sigma_L scales
    with sqrt(lead_time) -- a standard, if simplifying, textbook
    assumption (real demand errors are often autocorrelated, which would
    push the true number higher; noted as a limitation, not hidden).
    """
    z = norm.ppf(service_level)
    return max(0.0, z * sigma_daily * np.sqrt(lead_time))


def reorder_point(avg_daily_forecast: float, ss: float, lead_time: int = LEAD_TIME_DAYS) -> float:
    """Expected demand during the lead time, plus the safety buffer."""
    return avg_daily_forecast * lead_time + ss


def simulate_policy(
    dates: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    lead_time: int = LEAD_TIME_DAYS,
    service_level: float = SERVICE_LEVEL,
) -> dict:
    """
    Simulates a daily-review, order-up-to-reorder-point policy over one
    family's backtest window, using that model's own forecasts to set
    the reorder point, then testing it against what actually happened.

    Policy: each day, receive any order placed `lead_time` days ago,
    fill actual demand from on-hand stock (recording any shortfall as a
    stockout), then if inventory position (on-hand + on-order) has
    fallen to or below the reorder point, order enough to bring it back
    up to the reorder point.
    """
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
    """
    Runs simulate_policy for every family+model, using that model's
    full set of backtest-window forecasts (all folds concatenated, in
    date order) as the demand signal driving the replenishment policy.
    """
    rows = []
    for (family, model), g in detailed.sort_values(["family", "model", "date"]).groupby(["family", "model"]):
        result = simulate_policy(
            g["date"].values, g["actual"].values, g["predicted"].values,
            lead_time=lead_time, service_level=service_level,
        )
        rows.append({"family": family, "model": model, **result})
    return pd.DataFrame(rows)
