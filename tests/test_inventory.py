"""Inventory policy tests.

The safety-stock formula encodes a business rule, so its properties are worth
asserting directly: more variability must always mean more buffer, in both
demand and lead time.
"""
import numpy as np
import pandas as pd
import pytest

from retail_intel.pipelines.serving.nodes import (
    build_inventory_position,
    build_reorder_alerts,
)

PARAMS = {
    "simulated": True,
    "seed": 42,
    "opening_cover_weeks": 6.0,
    "service_level": 0.95,
    "review_period_weeks": 1.0,
    "default_lead_time_days": 12.0,
}


def _inputs(demand_std: float, lead_std: float) -> tuple:
    weeks = pd.date_range("2017-01-02", periods=60, freq="W-MON")
    rng = np.random.default_rng(1)
    demand = pd.DataFrame(
        {
            "category": "toys",
            "week": weeks,
            "units": np.clip(rng.normal(50, demand_std, 60), 0, None),
            "revenue": 1000.0,
            "orders": 10,
        }
    )
    forecast = pd.DataFrame(
        {
            "category": "toys",
            "horizon": range(1, 9),
            "forecast_units": 50.0,
            "forecast_hi_95": 70.0,
        }
    )
    lead = pd.DataFrame(
        {
            "category": ["toys"],
            "seller_state": ["SP"],
            "shipments": [500],
            "lead_time_mean_days": [14.0],
            "lead_time_std_days": [lead_std],
            "lead_time_p50_days": [13.0],
            "lead_time_p95_days": [20.0],
            "exception_rate": [0.05],
        }
    )
    return forecast, demand, lead


def test_safety_stock_rises_with_demand_variability():
    low = build_inventory_position(*_inputs(demand_std=2, lead_std=1), PARAMS)
    high = build_inventory_position(*_inputs(demand_std=20, lead_std=1), PARAMS)
    assert high["safety_stock_units"].iat[0] > low["safety_stock_units"].iat[0]


def test_safety_stock_rises_with_lead_time_variability():
    """The term most policies omit -- an erratic supplier needs more buffer."""
    steady = build_inventory_position(*_inputs(demand_std=5, lead_std=0.5), PARAMS)
    erratic = build_inventory_position(*_inputs(demand_std=5, lead_std=9.0), PARAMS)
    assert erratic["safety_stock_units"].iat[0] > steady["safety_stock_units"].iat[0]


def test_reorder_point_exceeds_safety_stock():
    position = build_inventory_position(*_inputs(5, 2), PARAMS)
    assert (position["reorder_point_units"] > position["safety_stock_units"]).all()


def test_higher_service_level_demands_more_stock():
    low = build_inventory_position(*_inputs(5, 2), {**PARAMS, "service_level": 0.80})
    high = build_inventory_position(*_inputs(5, 2), {**PARAMS, "service_level": 0.99})
    assert high["safety_stock_units"].iat[0] > low["safety_stock_units"].iat[0]


def test_alerts_only_contain_actionable_categories():
    position = build_inventory_position(*_inputs(5, 2), PARAMS)
    alerts = build_reorder_alerts(position)
    assert set(alerts["stock_status"]).issubset({"STOCKOUT", "CRITICAL", "REORDER"})
    # ranked so the most urgent row is first
    assert alerts["urgency_rank"].is_monotonic_decreasing


def test_simulated_flag_is_carried_through():
    """Every artefact built on assumed stock must say so."""
    position = build_inventory_position(*_inputs(5, 2), PARAMS)
    assert position["simulated_stock"].all()
    assert "simulated_stock" in build_reorder_alerts(position).columns
