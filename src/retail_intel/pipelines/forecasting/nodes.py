"""Weekly demand forecasting with walk-forward validation.

Validation design
-----------------
A random train/test split is meaningless for a forecast. It lets the model
train on July and be scored on June, so it "predicts" the past using the
future. Metrics from that setup look excellent and mean nothing.

This module uses expanding-window walk-forward validation. Each fold trains on
everything whose outcome was already known at a cut-off week, and is scored on
the `horizon_weeks` that follow it; the cut-off then advances. That is the
situation the model faces in production: know the past, predict forward, find
out later whether you were right.

Crucially, training rows are selected on their **target** week, not their
origin week. Selecting on origin puts the test rows themselves into the
training set -- their origin *is* the cut-off -- and hands the model the
answers it is about to be scored on. That mistake produced a WAPE of 0.10 in an
earlier revision of this file, against a true value of 0.40.

Multi-step design
-----------------
One model covers every horizon, with `horizon` as a feature, rather than one
model per horizon step. With roughly 100 weeks of history across 37 categories,
per-horizon models see too few rows and overfit hard enough to lose to a naive
baseline. Pooling multiplies the training set by the number of horizons.

Either way, the target is demand at week t+h and every feature is computed at
week t, so nothing reaches into the window being predicted.

Baselines
---------
Metrics are always reported against two naive baselines, each scored only on
the rows where that baseline is defined. A gradient-boosted model that cannot
beat "assume next week equals last week" is not a forecast, and without the
baseline in the table there is no way to tell. This model beats it by a modest
margin -- the margin is reported as it is.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

log = logging.getLogger(__name__)

TARGET = "units"


def _category_dtype(frame: pd.DataFrame) -> pd.CategoricalDtype:
    """A fixed category dtype, so train and predict share one encoding.

    Letting pandas infer categories separately per split would silently
    renumber them, and the model would read a different category than the row
    actually belongs to.
    """
    return pd.CategoricalDtype(categories=sorted(frame["category"].unique()))


def _feature_columns(frame: pd.DataFrame, params: dict) -> list[str]:
    """The configured feature set, checked against what the frame actually has."""
    configured = params["model"]["features"]
    missing = [column for column in configured if column not in frame.columns]
    if missing:
        raise KeyError(
            f"configured features absent from the feature store: {missing}. "
            f"Check `forecasting.model.features` in parameters.yml against the "
            f"lags and rolling windows the feature pipeline actually builds."
        )
    return list(configured)


def _design_matrix(
    frame: pd.DataFrame, features: list[str], dtype: pd.CategoricalDtype
) -> pd.DataFrame:
    """Numeric features, the horizon, and the category identity.

    Without the category column the model sees only lags and rolling means and
    cannot learn that one category peaks in a different week to another. The
    lags carry the level; the category carries the shape.
    """
    matrix = frame[features + ["horizon"]].copy()
    matrix["category"] = frame["category"].astype(dtype)
    return matrix


def _make_model(params: dict) -> HistGradientBoostingRegressor:
    """One estimator definition, shared by the backtest and the production refit.

    If these drifted apart, the validated error would describe a model that
    never ships.
    """
    config = params["model"]
    return HistGradientBoostingRegressor(
        loss=config["loss"],
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2_regularization"],
        early_stopping=config["early_stopping"],
        n_iter_no_change=config["n_iter_no_change"],
        validation_fraction=config["validation_fraction"],
        categorical_features="from_dtype",
        random_state=42,
    )


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error.

    Used in place of MAPE because weekly category demand contains genuine
    zeros, and MAPE divides by the actual value -- one zero week makes it
    infinite. WAPE divides by the total, so it stays finite and stays readable
    as "error as a share of volume".
    """
    denominator = np.abs(actual).sum()
    return float(np.abs(actual - predicted).sum() / denominator) if denominator else np.nan


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    return {
        "mae": round(float(np.abs(error).mean()), 4),
        "rmse": round(float(np.sqrt((error**2).mean())), 4),
        "bias": round(float(error.mean()), 4),
        "wape": round(_wape(actual, predicted), 4),
        "n": int(len(actual)),
    }


def _build_supervised(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach the week t+h target to the feature row observed at week t."""
    work = frame.sort_values(["category", "week"]).copy()
    work["target"] = work.groupby("category")[TARGET].shift(-horizon)
    work["origin_week"] = work["week"]
    work["week"] = work["week"] + pd.to_timedelta(horizon, unit="W")
    work["horizon"] = horizon
    return work.dropna(subset=["target"])


def _stack_horizons(data: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    """One long frame containing every (origin, horizon) pair."""
    return pd.concat(
        [_build_supervised(data, step) for step in range(1, horizon_weeks + 1)],
        ignore_index=True,
    )


def walk_forward_backtest(
    gold_demand_features: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, dict]:
    """Run expanding-window walk-forward validation."""
    data = gold_demand_features[gold_demand_features["is_forecastable"]].copy()
    features = _feature_columns(data, params)
    category_dtype = _category_dtype(data)
    horizon = params["horizon_weeks"]
    n_folds = params["n_folds"]

    weeks = np.sort(data["week"].unique())
    min_train = params["min_train_weeks"]
    if len(weeks) < min_train + horizon + n_folds:
        raise ValueError(
            f"only {len(weeks)} weeks available; need at least "
            f"{min_train + horizon + n_folds} for {n_folds} folds at a "
            f"{horizon}-week horizon"
        )

    supervised = _stack_horizons(data, horizon)
    cutoff_positions = np.linspace(
        min_train, len(weeks) - horizon, n_folds + 1
    )[:-1].astype(int)

    predictions: list[pd.DataFrame] = []

    for fold, position in enumerate(cutoff_positions, start=1):
        cutoff_week = weeks[position]

        # Training rows are selected on their TARGET week: a row may be trained
        # on only if the week it predicts has already happened by the cut-off.
        train = supervised[supervised["week"] <= cutoff_week]
        test = supervised[supervised["origin_week"] == cutoff_week]
        if train.empty or test.empty:
            continue

        overlap = set(map(tuple, train[["category", "week", "horizon"]].to_numpy())) & set(
            map(tuple, test[["category", "week", "horizon"]].to_numpy())
        )
        if overlap:
            raise RuntimeError(
                f"fold {fold}: {len(overlap)} rows appear in both train and test. "
                f"Refusing to report a leaked metric."
            )

        model = _make_model(params)
        model.fit(_design_matrix(train, features, category_dtype), train["target"])
        predicted = np.clip(
            np.nan_to_num(
                model.predict(_design_matrix(test, features, category_dtype))
            ),
            0,
            None,
        )

        fold_frame = test[
            ["category", "origin_week", "week", "horizon", "target"]
        ].copy()
        fold_frame["prediction"] = predicted
        fold_frame["fold"] = fold
        fold_frame["cutoff_week"] = cutoff_week
        fold_frame["naive_last_week"] = test["units_lag_1"].to_numpy()
        fold_frame["naive_seasonal"] = (
            test["units_lag_52"].to_numpy()
            if "units_lag_52" in test.columns
            else np.nan
        )
        predictions.append(fold_frame)

        log.info(
            "fold %d/%d: cut-off %s | train %d rows | test %d rows",
            fold, n_folds, pd.Timestamp(cutoff_week).date(), len(train), len(test),
        )

    backtest = pd.concat(predictions, ignore_index=True)
    actual = backtest["target"].to_numpy()

    def _baseline(column: str) -> dict:
        """Score a baseline only on rows where that baseline is defined.

        The seasonal baseline needs a 52-week lag, which does not exist for the
        first year of a two-year dataset. Filling those with zero would not
        measure the seasonal baseline -- it would measure a constant-zero
        forecast, and flatter the model for a reason unrelated to the model.
        """
        available = backtest[column].notna()
        scored = _metrics(
            backtest.loc[available, "target"].to_numpy(),
            backtest.loc[available, column].to_numpy(),
        )
        scored["coverage"] = round(float(available.mean()), 4)
        scored["model_wape_on_same_rows"] = round(
            _wape(
                backtest.loc[available, "target"].to_numpy(),
                backtest.loc[available, "prediction"].to_numpy(),
            ),
            4,
        )
        return scored

    metrics = {
        "config": {
            "horizon_weeks": horizon,
            "n_folds": n_folds,
            "min_train_weeks": min_train,
            "validation": "expanding-window walk-forward",
            "train_selection": "target week <= fold cut-off (test rows excluded)",
            "pooled_horizons": True,
            "categories_modelled": int(data["category"].nunique()),
            "features_used": len(features) + 2,   # + horizon, category
            "loss": params["model"]["loss"],
        },
        "model": _metrics(actual, backtest["prediction"].to_numpy()),
        "baseline_naive_last_week": _baseline("naive_last_week"),
        "baseline_seasonal_naive": _baseline("naive_seasonal"),
    }

    naive = metrics["baseline_naive_last_week"]
    metrics["wape_improvement_vs_naive"] = round(
        1 - naive["model_wape_on_same_rows"] / naive["wape"], 4
    )
    seasonal = metrics["baseline_seasonal_naive"]
    metrics["wape_improvement_vs_seasonal"] = (
        round(1 - seasonal["model_wape_on_same_rows"] / seasonal["wape"], 4)
        if seasonal["wape"] and not np.isnan(seasonal["wape"])
        else None
    )

    metrics["by_horizon"] = {
        int(step): {
            **_metrics(group["target"].to_numpy(), group["prediction"].to_numpy()),
            "naive_wape": round(
                _wape(
                    group["target"].to_numpy(),
                    np.nan_to_num(group["naive_last_week"].to_numpy()),
                ),
                4,
            ),
        }
        for step, group in backtest.groupby("horizon")
    }
    metrics["by_fold"] = {
        int(fold): _metrics(group["target"].to_numpy(), group["prediction"].to_numpy())
        for fold, group in backtest.groupby("fold")
    }

    # Stated plainly, because it qualifies every number above: the model
    # configuration was chosen by comparing candidates on these same folds.
    # The reported improvement is therefore mildly optimistic. A fully clean
    # estimate needs a nested scheme or a period held back from selection
    # entirely.
    metrics["caveat"] = (
        "Model hyperparameters were selected by comparing candidate "
        "configurations on these same walk-forward folds. The reported "
        "improvement over baseline is therefore optimistic; a nested "
        "cross-validation or a fully held-out period would be required for an "
        "unbiased estimate."
    )

    log.info(
        "backtest: model WAPE %.4f | naive %.4f (coverage %.0f%%) | seasonal %.4f "
        "(coverage %.0f%%) | %.1f%% better than naive",
        metrics["model"]["wape"],
        naive["wape"],
        100 * naive["coverage"],
        seasonal["wape"],
        100 * seasonal["coverage"],
        100 * metrics["wape_improvement_vs_naive"],
    )
    if metrics["wape_improvement_vs_naive"] <= 0:
        log.warning(
            "the model does NOT beat the naive baseline. Reported as-is; a "
            "forecast that loses to 'same as last week' should not be shipped."
        )
    return backtest, metrics


def train_final_model(gold_demand_features: pd.DataFrame, params: dict) -> dict:
    """Refit on the full history, with the configuration the backtest validated."""
    data = gold_demand_features[gold_demand_features["is_forecastable"]].copy()
    features = _feature_columns(data, params)
    category_dtype = _category_dtype(data)

    supervised = _stack_horizons(data, params["horizon_weeks"])
    model = _make_model(params)
    model.fit(
        _design_matrix(supervised, features, category_dtype), supervised["target"]
    )

    log.info(
        "trained pooled demand model on %d rows across %d horizons",
        len(supervised),
        params["horizon_weeks"],
    )
    return {
        "model": model,
        "features": features,
        "category_dtype": category_dtype,
        "params": params,
    }


def generate_forecast(
    demand_model: dict, gold_demand_features: pd.DataFrame, demand_backtest: pd.DataFrame
) -> pd.DataFrame:
    """Forecast the next `horizon_weeks` for every forecastable category.

    Prediction intervals come from the backtest's own residuals, per horizon.
    Uncertainty that widens with horizon is the entire basis of the
    safety-stock calculation downstream -- a point forecast cannot tell an
    inventory manager how much buffer to hold.
    """
    data = gold_demand_features[gold_demand_features["is_forecastable"]].copy()
    features = demand_model["features"]
    category_dtype = demand_model["category_dtype"]
    model = demand_model["model"]
    horizon_weeks = demand_model["params"]["horizon_weeks"]

    # forecast origin: the last fully-observed week of each category
    latest = data.sort_values("week").groupby("category").tail(1).copy()

    residual_std = (
        demand_backtest.assign(residual=lambda f: f["target"] - f["prediction"])
        .groupby("horizon")["residual"]
        .std()
        .to_dict()
    )

    rows = []
    for step in range(1, horizon_weeks + 1):
        window = latest.assign(horizon=step)
        predicted = np.clip(
            np.nan_to_num(
                model.predict(_design_matrix(window, features, category_dtype))
            ),
            0,
            None,
        )
        sigma = residual_std.get(step, np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "category": window["category"].to_numpy(),
                    "origin_week": window["week"].to_numpy(),
                    "forecast_week": window["week"].to_numpy()
                    + pd.to_timedelta(step, unit="W"),
                    "horizon": step,
                    "forecast_units": predicted,
                    "forecast_lo_80": np.clip(predicted - 1.2816 * sigma, 0, None),
                    "forecast_hi_80": predicted + 1.2816 * sigma,
                    "forecast_lo_95": np.clip(predicted - 1.9600 * sigma, 0, None),
                    "forecast_hi_95": predicted + 1.9600 * sigma,
                    "residual_std": sigma,
                }
            )
        )

    forecast = pd.concat(rows, ignore_index=True).sort_values(["category", "horizon"])
    log.info(
        "forecast: %d categories x %d weeks, %.0f total units forecast",
        forecast["category"].nunique(),
        forecast["horizon"].nunique(),
        forecast["forecast_units"].sum(),
    )
    return forecast.reset_index(drop=True)
