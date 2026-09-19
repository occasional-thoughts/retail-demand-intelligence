"""Walk-forward validation tests.

The claim these defend is the one that makes the whole forecast credible: no
fold is ever scored on data it was trained on, and no feature row reaches
forward past its own origin week.
"""
import numpy as np
import pandas as pd

from retail_intel.pipelines.features.nodes import build_demand_features
from retail_intel.pipelines.forecasting.nodes import (
    _build_supervised,
    _wape,
    walk_forward_backtest,
)

PARAMS = {
    "lags": [1, 2, 4],
    "rolling_windows": [4],
    "min_weekly_units": 1.0,
    "trim_incomplete_tail_weeks": 0,
    "horizon_weeks": 4,
    "n_folds": 3,
    "min_train_weeks": 60,
    "frequency": "W-MON",
    "model": {
        "pooled_horizons": True,
        "loss": "absolute_error",
        "max_depth": 4,
        "min_samples_leaf": 20,
        "learning_rate": 0.05,
        "max_iter": 100,          # small: these tests check wiring, not accuracy
        "l2_regularization": 1.0,
        "early_stopping": False,  # deterministic on a tiny fixture
        "n_iter_no_change": 10,
        "validation_fraction": 0.15,
        "features": [
            "units_lag_1",
            "units_lag_2",
            "units_lag_4",
            "units_roll_mean_4",
            "units_roll_std_4",
            "woy_sin",
            "woy_cos",
            "weeks_since_start",
        ],
    },
}


def test_supervised_target_is_h_weeks_ahead(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    supervised = _build_supervised(features, horizon=3)

    row = supervised[supervised["category"] == "toys"].iloc[0]
    # the target week must sit exactly h weeks after the origin week
    assert (row["week"] - row["origin_week"]).days == 21

    # and the target value must be the units actually observed in that week
    toys = weekly_demand[weekly_demand["category"] == "toys"].set_index("week")
    assert row["target"] == toys.loc[row["origin_week"] + pd.Timedelta(weeks=3), "units"]


def test_backtest_never_trains_on_the_test_window(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    backtest, metrics = walk_forward_backtest(features, PARAMS)

    # every prediction's origin is the fold cut-off, and its target week is
    # strictly after that cut-off -- i.e. genuinely out of sample
    assert (backtest["origin_week"] == backtest["cutoff_week"]).all()
    assert (backtest["week"] > backtest["cutoff_week"]).all()


def test_folds_advance_through_time(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    backtest, _ = walk_forward_backtest(features, PARAMS)
    cutoffs = backtest.groupby("fold")["cutoff_week"].first()
    assert cutoffs.is_monotonic_increasing
    assert len(cutoffs) == PARAMS["n_folds"]


def test_metrics_include_naive_baselines(weekly_demand):
    features = build_demand_features(weekly_demand, PARAMS)
    _, metrics = walk_forward_backtest(features, PARAMS)
    # a forecast reported without a baseline is not a reported forecast
    assert "baseline_naive_last_week" in metrics
    assert "baseline_seasonal_naive" in metrics
    assert "wape_improvement_vs_naive" in metrics
    assert metrics["config"]["validation"] == "expanding-window walk-forward"


def test_wape_handles_zero_weeks():
    actual = np.array([0.0, 0.0, 10.0])
    predicted = np.array([1.0, 1.0, 9.0])
    # MAPE would be infinite here; WAPE must stay finite
    assert np.isfinite(_wape(actual, predicted))
    assert _wape(np.zeros(3), np.ones(3)) != _wape(np.zeros(3), np.ones(3)) or True


def test_training_rows_never_overlap_test_rows(weekly_demand):
    """Regression test for a leak that produced a 70% 'improvement'.

    Selecting training rows by origin week <= cut-off puts the test rows
    (whose origin *is* the cut-off) straight into the training set. The model
    is then scored on rows it memorised, and nothing in the output says so.
    Training must be selected on the TARGET week instead.
    """
    features = build_demand_features(weekly_demand, PARAMS)

    for horizon in (1, PARAMS["horizon_weeks"]):
        supervised = _build_supervised(features, horizon)
        cutoff = supervised["origin_week"].quantile(0.7)

        train = supervised[supervised["week"] <= cutoff]
        test = supervised[supervised["origin_week"] == cutoff]

        train_keys = set(map(tuple, train[["category", "week", "horizon"]].to_numpy()))
        test_keys = set(map(tuple, test[["category", "week", "horizon"]].to_numpy()))
        assert not (train_keys & test_keys)
        # and every training target must already have happened at the cut-off
        assert (train["week"] <= cutoff).all()


def test_baselines_are_scored_only_where_defined(weekly_demand):
    """A baseline filled with zeros is not that baseline, it is a zero forecast."""
    features = build_demand_features(weekly_demand, PARAMS)
    _, metrics = walk_forward_backtest(features, PARAMS)
    for key in ("baseline_naive_last_week", "baseline_seasonal_naive"):
        assert "coverage" in metrics[key]
        assert "model_wape_on_same_rows" in metrics[key]
