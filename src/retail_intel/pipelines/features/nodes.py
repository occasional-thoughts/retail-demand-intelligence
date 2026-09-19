"""Silver and gold feature construction.

Two feature stores are built here from one conformed order-line table:

  * a weekly demand series per category, with lag and rolling features, for
    the forecasting model;
  * a customer-level table combining RFM, delivery experience, browsing
    behaviour and review sentiment, for the churn model.

The rule both obey: no feature may be computed from data that would not have
been available at the moment the prediction is made. That constraint is what
separates a backtest from a fantasy, and it is enforced structurally here --
the churn features are cut off at an observation date, and the demand lags
are built with `shift` before any rolling window touches them.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from retail_intel.sentiment import (
    score_frame,
    score_with_vader,
    validate_against_rating,
)

log = logging.getLogger(__name__)

# Statuses that represent a real, revenue-bearing sale. Cancelled and
# unavailable orders are demand signals of a sort, but they are not shipments,
# and counting them would inflate every forecast and every revenue figure.
FULFILLED_STATUSES = {"delivered", "shipped", "invoiced", "processing", "approved"}


def build_order_lines(
    bronze_orders: pd.DataFrame,
    bronze_order_items: pd.DataFrame,
    pg_payments: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Conform the relational extracts into one order-line grain table."""
    frame = bronze_order_items.merge(bronze_orders, on="order_id", how="inner")

    before = len(frame)
    frame = frame[frame["order_status"].isin(FULFILLED_STATUSES)].copy()
    log.info(
        "silver.order_lines: dropped %d lines on non-fulfilled statuses (%.2f%%)",
        before - len(frame),
        100 * (before - len(frame)) / before,
    )

    frame["line_revenue"] = frame["price"] + frame["freight_value"].fillna(0)
    # Weekly anchor comes from the configured frequency, so this and the dense
    # panel in build_weekly_demand can never drift apart.
    frame["order_week"] = (
        frame["order_purchase_timestamp"]
        .dt.to_period(params["frequency"])
        .dt.start_time
    )
    frame["order_month"] = (
        frame["order_purchase_timestamp"].dt.to_period("M").dt.start_time
    )

    # delivery performance, which feeds both the churn model and the ops view
    frame["delivery_days"] = (
        frame["order_delivered_customer_date"] - frame["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    frame["promised_days"] = (
        frame["order_estimated_delivery_date"] - frame["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    frame["days_late"] = (
        frame["order_delivered_customer_date"] - frame["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    frame["delivered_late"] = frame["days_late"] > 0

    payments = pg_payments.copy()
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce")
    frame = frame.merge(payments, on="order_id", how="left")

    log.info(
        "silver.order_lines: %d lines, %d orders, %d categories, R$%.1fM revenue",
        len(frame),
        frame["order_id"].nunique(),
        frame["category"].nunique(),
        frame["line_revenue"].sum() / 1e6,
    )
    return frame


# ---------------------------------------------------------------------
# demand
# ---------------------------------------------------------------------
def build_weekly_demand(silver_order_lines: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aggregate order lines into a weekly demand series per category.

    Two corrections applied here matter more than the aggregation itself:

    1. The panel is made *dense*. A category with no sales in a week must
       appear as a zero, not be absent -- otherwise `shift(1)` silently
       reaches back to whenever that category last sold, and every lag
       feature becomes wrong by an unknown number of weeks.

    2. The final weeks are trimmed. Olist's extract stops mid-flight, so the
       last few weeks are partial. Left in, they look like a demand collapse,
       and the model dutifully learns to forecast one.
    """
    weekly = (
        silver_order_lines.groupby(["category", "order_week"], as_index=False)
        .agg(
            units=("order_item_id", "count"),
            revenue=("line_revenue", "sum"),
            orders=("order_id", "nunique"),
            avg_price=("price", "mean"),
            avg_freight=("freight_value", "mean"),
        )
        .rename(columns={"order_week": "week"})
    )

    cutoff = weekly["week"].max() - pd.Timedelta(weeks=params["trim_incomplete_tail_weeks"])
    trimmed = weekly[weekly["week"] <= cutoff].copy()
    log.info(
        "gold.weekly_demand: trimmed %d partial rows after %s",
        len(weekly) - len(trimmed),
        cutoff.date(),
    )

    # Dense panel: every category x every week in its own active span.
    #
    # The index is built with `period_range(...).start_time`, matching exactly
    # how `order_week` was derived in build_order_lines. This is not
    # interchangeable with `date_range(freq="W-MON")`: a W-MON *period* ends on
    # Monday and therefore starts on Tuesday, while a W-MON *date_range* yields
    # Mondays. Mixing the two produces an index that matches no observation, a
    # left join that fills every measure with zero, and a panel that is
    # structurally perfect and entirely empty -- with no error raised anywhere.
    spans = trimmed.groupby("category")["week"].agg(["min", "max"])
    index_frames = [
        pd.DataFrame(
            {
                "category": category,
                "week": pd.period_range(
                    row["min"], row["max"], freq=params["frequency"]
                ).start_time,
            }
        )
        for category, row in spans.iterrows()
    ]
    dense = pd.concat(index_frames, ignore_index=True).merge(
        trimmed, on=["category", "week"], how="left"
    )

    # Guard against exactly the failure described above. A densifying join must
    # add rows, never destroy the measure it is densifying around.
    observed_units = trimmed["units"].sum()
    joined_units = dense["units"].sum()
    if observed_units > 0 and joined_units < observed_units * 0.999:
        raise ValueError(
            f"densifying the weekly panel lost demand: {observed_units:,.0f} units "
            f"observed but only {joined_units:,.0f} survived the join. The generated "
            f"week index does not align with the observed week anchors "
            f"(observed weekdays: "
            f"{sorted(trimmed['week'].dt.day_name().unique())}, generated: "
            f"{sorted(pd.Series(dense['week'].unique()).dt.day_name().unique())})."
        )

    for column in ("units", "revenue", "orders"):
        dense[column] = dense[column].fillna(0)
    dense[["avg_price", "avg_freight"]] = dense.groupby("category")[
        ["avg_price", "avg_freight"]
    ].ffill()

    added = len(dense) - len(trimmed)
    log.info(
        "gold.weekly_demand: %d rows across %d categories (%d zero-weeks made explicit)",
        len(dense),
        dense["category"].nunique(),
        added,
    )
    return dense.sort_values(["category", "week"]).reset_index(drop=True)


def build_demand_features(gold_weekly_demand: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Lag, rolling and calendar features for the demand model.

    Every rolling statistic is computed on an already-shifted series. Rolling
    first and shifting afterwards would include the current week in its own
    predictor -- the single most common way a time-series model ends up with
    an impressive backtest and no predictive power.
    """
    frame = gold_weekly_demand.sort_values(["category", "week"]).copy()
    grouped = frame.groupby("category")["units"]

    for lag in params["lags"]:
        frame[f"units_lag_{lag}"] = grouped.shift(lag)

    # Rolling statistics are computed on the already-shifted series AND
    # re-grouped by category. Both halves matter:
    #   * shifting first keeps a week out of its own predictor;
    #   * re-grouping keeps one category's opening weeks from rolling over the
    #     previous category's closing weeks. Without it the frame is just one
    #     long series and every category boundary leaks across.
    shifted = grouped.shift(1)          # last fully-observed week, per category
    by_category = shifted.groupby(frame["category"])
    for window in params["rolling_windows"]:
        rolling = by_category.rolling(window, min_periods=2)
        frame[f"units_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        frame[f"units_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
        frame[f"units_roll_max_{window}"] = rolling.max().reset_index(level=0, drop=True)

    # short-window trend, using the two shortest configured lags
    short_lag, long_lag = sorted(params["lags"])[0], sorted(params["lags"])[-1]
    frame[f"units_trend_{short_lag}_{long_lag}"] = (
        frame[f"units_lag_{short_lag}"] - frame[f"units_lag_{long_lag}"]
    )

    # momentum: shortest rolling mean against the longest. Derived from the
    # configured windows rather than hard-coded, so changing `rolling_windows`
    # in parameters.yml cannot silently break this feature.
    windows = sorted(params["rolling_windows"])
    if len(windows) >= 2:
        fast, slow = windows[0], windows[-1]
        frame[f"units_ratio_{fast}_{slow}"] = frame[
            f"units_roll_mean_{fast}"
        ] / frame[f"units_roll_mean_{slow}"].replace(0, np.nan)

    frame["week_of_year"] = frame["week"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["week"].dt.month
    frame["quarter"] = frame["week"].dt.quarter
    frame["year"] = frame["week"].dt.year
    # cyclical encoding so week 52 sits next to week 1 rather than 51 units away
    frame["woy_sin"] = np.sin(2 * np.pi * frame["week_of_year"] / 52)
    frame["woy_cos"] = np.cos(2 * np.pi * frame["week_of_year"] / 52)

    # Brazilian retail peaks: Black Friday (Nov) and Mother's Day (May) are the
    # two events that visibly move this dataset.
    frame["is_black_friday_window"] = frame["week_of_year"].between(46, 48).astype(int)
    frame["is_mothers_day_window"] = frame["week_of_year"].between(17, 19).astype(int)

    frame["weeks_since_start"] = frame.groupby("category").cumcount()

    active = frame.groupby("category")["units"].transform("mean") >= params["min_weekly_units"]
    frame["is_forecastable"] = active

    log.info(
        "gold.demand_features: %d rows, %d features, %d of %d categories forecastable",
        len(frame),
        frame.shape[1],
        frame.loc[active, "category"].nunique(),
        frame["category"].nunique(),
    )
    return frame


# ---------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------
def build_review_sentiment(silver_reviews: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Score review text, and validate the scorer before trusting it.

    The validation block is not decoration. Olist reviews are Portuguese, and
    the obvious tool (VADER) is English; scoring both and comparing them
    against the star rating is what establishes which one is actually reading
    the text. The result is logged so it appears in every run's output.
    """
    frame = silver_reviews.copy()
    frame["message"] = frame["message"].astype("object").where(frame["message"].notna(), None)

    has_text = frame["message"].map(
        lambda value: isinstance(value, str) and len(value.strip()) >= params["min_chars"]
    )
    frame["has_text"] = has_text

    frame["sentiment"] = 0.0
    frame.loc[has_text, "sentiment"] = score_frame(frame.loc[has_text])

    frame["sentiment_vader"] = 0.0
    frame.loc[has_text, "sentiment_vader"] = score_with_vader(frame.loc[has_text])

    validation = validate_against_rating(
        frame.loc[has_text], ["sentiment", "sentiment_vader"]
    )
    log.info("sentiment validation vs star rating: %s", validation)

    chosen = validation["sentiment"]
    baseline = validation["sentiment_vader"]
    if chosen["spearman_vs_rating"] <= baseline["spearman_vs_rating"]:
        log.warning(
            "the Portuguese scorer did not beat the English baseline "
            "(%.3f vs %.3f) -- the lexicon needs work before this feature is used",
            chosen["spearman_vs_rating"],
            baseline["spearman_vs_rating"],
        )

    frame["sentiment_label"] = pd.cut(
        frame["sentiment"],
        bins=[-1.01, -0.2, 0.2, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    frame.attrs["validation"] = validation

    log.info(
        "silver.review_sentiment: %d reviews, %d with text, mean sentiment %.3f",
        len(frame),
        int(has_text.sum()),
        frame.loc[has_text, "sentiment"].mean(),
    )
    return frame


# ---------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------
def build_customer_features(
    silver_order_lines: pd.DataFrame,
    silver_sessions: pd.DataFrame,
    silver_review_sentiment: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Customer-level feature store, cut off at a fixed observation date.

    The observation cut-off is what makes the churn label honest. Everything
    to the left of it becomes a feature; whether the customer returns to the
    right of it becomes the label. Nothing from the outcome window is allowed
    to leak backwards into a predictor.

    Note on this dataset: Olist customers are overwhelmingly one-time buyers,
    so the returning class is small. That is reported rather than engineered
    away -- resampling the target to look balanced would only hide the fact
    that the base rate is what makes this problem hard.
    """
    lines = silver_order_lines.copy()
    max_date = lines["order_purchase_timestamp"].max()
    cutoff = max_date - pd.Timedelta(days=params["outcome_window_days"])
    log.info(
        "customer features: history <= %s, outcome window (%s, %s]",
        cutoff.date(),
        cutoff.date(),
        max_date.date(),
    )

    history = lines[lines["order_purchase_timestamp"] <= cutoff]
    outcome = lines[lines["order_purchase_timestamp"] > cutoff]

    # ---- RFM, from history only
    orders = history.groupby(
        ["customer_unique_id", "order_id"], as_index=False
    ).agg(
        order_value=("line_revenue", "sum"),
        order_ts=("order_purchase_timestamp", "first"),
        items=("order_item_id", "count"),
        state=("customer_state", "first"),
        late=("delivered_late", "max"),
        delivery_days=("delivery_days", "mean"),
        days_late=("days_late", "mean"),
    )

    features = orders.groupby("customer_unique_id", as_index=False).agg(
        recency_days=("order_ts", lambda s: (cutoff - s.max()).days),
        tenure_days=("order_ts", lambda s: (cutoff - s.min()).days),
        frequency=("order_id", "count"),
        monetary=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        max_order_value=("order_value", "max"),
        total_items=("items", "sum"),
        state=("state", "first"),
        any_late_delivery=("late", "max"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_days_late=("days_late", "mean"),
    )

    # ---- category breadth
    breadth = history.groupby("customer_unique_id", as_index=False).agg(
        distinct_categories=("category", "nunique"),
        top_category=("category", lambda s: s.mode().iat[0] if len(s) else "unknown"),
        avg_freight=("freight_value", "mean"),
    )
    features = features.merge(breadth, on="customer_unique_id", how="left")

    # ---- review sentiment, joined through the order key
    order_customer = history[["order_id", "customer_unique_id"]].drop_duplicates()
    reviews = silver_review_sentiment.merge(order_customer, on="order_id", how="inner")
    review_features = reviews.groupby("customer_unique_id", as_index=False).agg(
        n_reviews=("review_id", "count"),
        avg_review_score=("review_score", "mean"),
        min_review_score=("review_score", "min"),
        avg_sentiment=("sentiment", "mean"),
        min_sentiment=("sentiment", "min"),
        n_text_reviews=("has_text", "sum"),
    )
    features = features.merge(review_features, on="customer_unique_id", how="left")

    # ---- browsing behaviour from the event store
    session_customer = history[["customer_id", "customer_unique_id"]].drop_duplicates()
    sessions = silver_sessions.merge(session_customer, on="customer_id", how="inner")
    session_columns = {
        "sessions": "sum",
        "converted_sessions": "sum",
        "total_events": "sum",
    }
    for optional in ("n_add_to_cart", "n_remove_from_cart", "n_product_view", "n_begin_checkout"):
        if optional in sessions.columns:
            session_columns[optional] = "sum"
    session_features = (
        sessions.groupby("customer_unique_id").agg(session_columns).reset_index()
    )
    features = features.merge(session_features, on="customer_unique_id", how="left")

    if "n_add_to_cart" in features.columns and "n_product_view" in features.columns:
        features["cart_conversion"] = features["n_add_to_cart"] / features[
            "n_product_view"
        ].replace(0, np.nan)
    if "sessions" in features.columns:
        features["events_per_session"] = features["total_events"] / features[
            "sessions"
        ].replace(0, np.nan)

    # ---- the label: did they come back inside the outcome window?
    returners = set(outcome["customer_unique_id"])
    features["returned"] = features["customer_unique_id"].isin(returners).astype(int)
    features["churned"] = 1 - features["returned"]

    features["observation_cutoff"] = cutoff

    log.info(
        "gold.customer_features: %d customers, %d features, return rate %.2f%% "
        "(%d returners) -- heavily imbalanced by design, not by accident",
        len(features),
        features.shape[1],
        100 * features["returned"].mean(),
        int(features["returned"].sum()),
    )
    return features


def build_rfm_segments(gold_customer_features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Score customers on R, F and M quintiles and name the resulting segments.

    Frequency is ranked before binning because this dataset's frequency
    distribution is extremely lopsided -- most customers sit at exactly 1 --
    and `qcut` on tied values collapses into a single bin. Ranking first
    spreads the ties so the quintiles remain usable.
    """
    frame = gold_customer_features.copy()
    bins = params["n_bins"]

    def _quintile(series: pd.Series, ascending: bool) -> pd.Series:
        ranked = series.rank(method="first", ascending=ascending)
        return pd.qcut(ranked, bins, labels=range(1, bins + 1)).astype(int)

    # recency: fewer days since last order is better, so rank descending
    frame["r_score"] = _quintile(frame["recency_days"], ascending=False)
    frame["f_score"] = _quintile(frame["frequency"], ascending=True)
    frame["m_score"] = _quintile(frame["monetary"], ascending=True)
    frame["rfm_score"] = frame["r_score"] + frame["f_score"] + frame["m_score"]
    frame["rfm_cell"] = (
        frame["r_score"].astype(str)
        + frame["f_score"].astype(str)
        + frame["m_score"].astype(str)
    )

    def _segment(row) -> str:
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and m >= 4:
            return "Loyal High-Value"
        if r >= 4 and m <= 2:
            return "Recent Low-Value"
        if r >= 3 and f >= 3:
            return "Promising"
        if r <= 2 and m >= 4:
            return "At Risk High-Value"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r <= 2 and m <= 2:
            return "Hibernating"
        return "Needs Attention"

    frame["segment"] = frame.apply(_segment, axis=1)

    summary = frame.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        revenue=("monetary", "sum"),
        return_rate=("returned", "mean"),
    )
    log.info("gold.rfm_segments:\n%s", summary.to_string())
    return frame
