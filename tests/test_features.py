"""Feature-engineering tests, focused on temporal leakage.

Leakage is the defect that does not announce itself: the pipeline runs, the
metrics improve, and the model is worthless in production. These tests make
the absence of leakage an asserted property rather than a hope.
"""
import numpy as np
import pandas as pd

from retail_intel.pipelines.features.nodes import (
    build_demand_features,
    build_weekly_demand,
)

PARAMS = {
    "lags": [1, 2, 4],
    "rolling_windows": [4],
    "min_weekly_units": 3.0,
    "trim_incomplete_tail_weeks": 0,
    "frequency": "W-MON",
    "horizon_weeks": 4,
}


def test_lag_features_only_look_backwards(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    one = features[features["category"] == "toys"].sort_values("week").reset_index(drop=True)

    # units_lag_1 at row i must equal units at row i-1, exactly
    expected = one["units"].shift(1)
    pd.testing.assert_series_equal(
        one["units_lag_1"], expected, check_names=False, check_dtype=False
    )


def test_rolling_features_exclude_the_current_week(weekly_demand):
    """A spike in week t must not appear in week t's own rolling mean."""
    spiked = weekly_demand.copy()
    spike_position = spiked.index[spiked["category"] == "toys"][80]
    spiked.loc[spike_position, "units"] = 100_000

    features = build_demand_features(spiked, PARAMS)
    toys = features[features["category"] == "toys"].reset_index(drop=True)
    row = toys.index[toys["units"] == 100_000][0]

    # the spike's own row cannot see it
    assert toys.loc[row, "units_roll_mean_4"] < 1_000
    # the following row must -- otherwise the feature is simply not working
    assert toys.loc[row + 1, "units_roll_mean_4"] > 1_000


def test_first_row_of_each_category_has_no_history(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    first_rows = features.groupby("category").head(1)
    assert first_rows["units_lag_1"].isna().all()
    assert first_rows["units_roll_mean_4"].isna().all()


def test_weekly_demand_panel_is_dense(weekly_demand):
    """Missing weeks must become explicit zeros, or every lag is wrong."""
    sparse = weekly_demand.drop(
        weekly_demand.index[(weekly_demand["category"] == "toys")][10:20]
    )
    dense = build_weekly_demand(
        sparse.assign(
            order_week=sparse["week"],
            order_item_id=1,
            order_id="o",
            line_revenue=sparse["revenue"],
            price=sparse["avg_price"],
            freight_value=sparse["avg_freight"],
        ),
        PARAMS,
    )
    toys = dense[dense["category"] == "toys"].sort_values("week")
    gaps = toys["week"].diff().dropna().dt.days.unique()
    assert set(gaps) == {7}, f"panel has gaps: {gaps}"


def test_incomplete_tail_weeks_are_trimmed(weekly_demand):
    params = {**PARAMS, "trim_incomplete_tail_weeks": 3}
    source = weekly_demand.assign(
        order_week=weekly_demand["week"],
        order_item_id=1,
        order_id="o",
        line_revenue=weekly_demand["revenue"],
        price=weekly_demand["avg_price"],
        freight_value=weekly_demand["avg_freight"],
    )
    trimmed = build_weekly_demand(source, params)
    assert trimmed["week"].max() < weekly_demand["week"].max()


def test_densifying_preserves_total_demand(weekly_demand):
    """A densifying join must add zero-weeks, never destroy the measure.

    Regression test for a silent defect: `to_period("W-MON").start_time` yields
    Tuesdays while `date_range(freq="W-MON")` yields Mondays. Mixing them
    produced a panel with the right shape, the right categories, the right
    number of rows -- and every single unit value zeroed, with no error raised.
    The demand model then trained happily on an empty series.
    """
    source = weekly_demand.assign(
        order_week=weekly_demand["week"],
        order_item_id=1,
        order_id="o",
        line_revenue=weekly_demand["revenue"],
        price=weekly_demand["avg_price"],
        freight_value=weekly_demand["avg_freight"],
    )
    dense = build_weekly_demand(source, PARAMS)

    # `units` is a count of order lines, so the correct reference is the number
    # of source rows -- not the fixture's own units column, which the
    # aggregation replaces.
    assert dense["units"].sum() == len(source)
    assert (dense["units"] > 0).any()
    # and densifying must only ever add rows
    assert len(dense) >= source[["category", "order_week"]].drop_duplicates().shape[0]
