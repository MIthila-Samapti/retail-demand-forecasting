import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inventory import safety_stock, reorder_point


def test_safety_stock_zero_when_no_error():
    assert safety_stock(sigma_daily=0, lead_time=7, service_level=0.95) == 0


def test_safety_stock_matches_known_z_score():
    # z for 95% service level is ~1.645
    ss = safety_stock(sigma_daily=10, lead_time=4, service_level=0.95)
    expected = 1.645 * 10 * (4 ** 0.5)
    assert abs(ss - expected) < 0.01


def test_safety_stock_increases_with_service_level():
    low = safety_stock(sigma_daily=10, lead_time=7, service_level=0.90)
    high = safety_stock(sigma_daily=10, lead_time=7, service_level=0.99)
    assert high > low


def test_reorder_point_includes_lead_time_demand_and_safety_stock():
    rop = reorder_point(avg_daily_forecast=50, ss=20, lead_time=7)
    assert rop == 50 * 7 + 20
