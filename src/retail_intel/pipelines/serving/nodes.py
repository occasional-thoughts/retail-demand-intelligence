"""Serving layer: the extracts the two dashboards read.

These are deliberately two different data products, not one table filtered two
ways. The operational extracts are at SKU-category grain, forward-looking over
eight weeks, and exist to answer "what must I do today". The executive
extracts are at company, segment and region grain, backward-looking over
months, and exist to answer "where is performance and risk moving".

On simulated inventory
----------------------
Olist publishes no stock ledger, no reorder policy and no supplier contracts.
The inventory position below is therefore *simulated* from a documented,
deterministic policy. It is built on real forecast demand and real lead times
derived from real delivery timestamps, so the mechanics and the relative
rankings are meaningful -- but the absolute stock figures are an assumption,
and every artefact that carries them says so.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)


def build_inventory_position(
    demand_forecast: pd.DataFrame,
    gold_weekly_demand: pd.DataFrame,
    silver_lead_times: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Compute stock cover, safety stock and reorder point per category.

    Safety stock uses the standard formula that accounts for variability in
    *both* demand and lead time:

        SS = z * sqrt( LT * sigma_demand^2  +  demand^2 * sigma_LT^2 )

    The second term is the one usually dropped, and dropping it is why so many
    reorder policies fail on exactly the suppliers that are erratic rather than
    slow. A supplier averaging 10 days with a 9-day spread needs far more
    buffer than one averaging 14 days with a 1-day spread, and only this form
    of the formula says so.
    """
    z = norm.ppf(params["service_level"])

    # forward demand over the full forecast horizon
    forward = demand_forecast.groupby("category", as_index=False).agg(
        forecast_units_horizon=("forecast_units", "sum"),
        forecast_units_p95_horizon=("forecast_hi_95", "sum"),
        horizon_weeks=("horizon", "max"),
    )
    forward["forecast_weekly_mean"] = (
        forward["forecast_units_horizon"] / forward["horizon_weeks"]
    )

    # historical demand variability, from observed weeks
    history = gold_weekly_demand.groupby("category", as_index=False).agg(
        hist_weekly_mean=("units", "mean"),
        hist_weekly_std=("units", "std"),
        weeks_observed=("units", "count"),
        last_week_units=("units", "last"),
    )
    history["hist_weekly_std"] = history["hist_weekly_std"].fillna(0)

    # lead time, aggregated up from the Spark scan feed
    lead = silver_lead_times.groupby("category", as_index=False).apply(
        lambda group: pd.Series(
            {
                "lead_time_mean_days": np.average(
                    group["lead_time_mean_days"], weights=group["shipments"]
                ),
                "lead_time_std_days": np.average(
                    group["lead_time_std_days"], weights=group["shipments"]
                ),
                "lead_time_p95_days": group["lead_time_p95_days"].max(),
                "shipments": group["shipments"].sum(),
                "exception_rate": np.average(
                    group["exception_rate"], weights=group["shipments"]
                ),
            }
        ),
        include_groups=False,
    )

    frame = forward.merge(history, on="category", how="left").merge(
        lead, on="category", how="left"
    )

    frame["lead_time_mean_days"] = frame["lead_time_mean_days"].fillna(
        params["default_lead_time_days"]
    )
    frame["lead_time_std_days"] = frame["lead_time_std_days"].fillna(0)
    frame["lead_time_weeks"] = frame["lead_time_mean_days"] / 7
    frame["lead_time_std_weeks"] = frame["lead_time_std_days"] / 7

    demand = frame["forecast_weekly_mean"]
    sigma_demand = frame["hist_weekly_std"]
    lead_weeks = frame["lead_time_weeks"]
    sigma_lead = frame["lead_time_std_weeks"]

    frame["safety_stock_units"] = z * np.sqrt(
        lead_weeks * sigma_demand**2 + (demand**2) * (sigma_lead**2)
    )
    frame["cycle_stock_units"] = demand * (lead_weeks + params["review_period_weeks"])
    frame["reorder_point_units"] = frame["cycle_stock_units"] + frame["safety_stock_units"]

    # ---- simulated opening stock ---------------------------------------
    # Deterministic: a fixed number of weeks of cover, perturbed by a hash of
    # the category name so the portfolio is not uniformly healthy. Seeded and
    # reproducible; clearly labelled as an assumption everywhere it surfaces.
    rng = np.random.default_rng(params["seed"])
    variation = rng.uniform(0.35, 1.9, len(frame))
    frame["on_hand_units"] = np.round(
        demand * params["opening_cover_weeks"] * variation
    )
    frame["simulated_stock"] = params["simulated"]

    frame["weeks_of_cover"] = frame["on_hand_units"] / demand.replace(0, np.nan)
    frame["cover_vs_leadtime_ratio"] = frame["weeks_of_cover"] / lead_weeks.replace(0, np.nan)
    frame["below_reorder_point"] = frame["on_hand_units"] < frame["reorder_point_units"]
    frame["projected_shortfall_units"] = np.clip(
        frame["forecast_units_horizon"] - frame["on_hand_units"], 0, None
    )
    frame["recommended_order_units"] = np.round(
        np.clip(frame["reorder_point_units"] - frame["on_hand_units"], 0, None)
        + demand * params["review_period_weeks"]
    )

    def _status(row) -> str:
        if row["on_hand_units"] <= 0:
            return "STOCKOUT"
        if row["weeks_of_cover"] < row["lead_time_weeks"]:
            return "CRITICAL"       # cannot be replenished before running out
        if row["below_reorder_point"]:
            return "REORDER"
        if row["weeks_of_cover"] > 16:
            return "OVERSTOCK"
        return "HEALTHY"

    frame["stock_status"] = frame.apply(_status, axis=1)

    log.info(
        "ops.inventory_position: %d categories -- %s",
        len(frame),
        frame["stock_status"].value_counts().to_dict(),
    )
    return frame.sort_values("projected_shortfall_units", ascending=False).reset_index(
        drop=True
    )


def build_reorder_alerts(ops_inventory_position: pd.DataFrame) -> pd.DataFrame:
    """The action list: only categories that need a decision this week.

    An operational dashboard that shows all 71 categories every morning is a
    dashboard nobody reads. This extract is filtered to the exceptions and
    ranked by consequence, so the top of the list is the first thing to do.
    """
    alerts = ops_inventory_position[
        ops_inventory_position["stock_status"].isin(
            ["STOCKOUT", "CRITICAL", "REORDER"]
        )
    ].copy()

    urgency = {"STOCKOUT": 3, "CRITICAL": 2, "REORDER": 1}
    alerts["urgency_rank"] = alerts["stock_status"].map(urgency)

    # days until cover runs out, at the forecast rate
    alerts["days_to_stockout"] = np.round(
        alerts["weeks_of_cover"].fillna(0) * 7
    ).clip(lower=0)
    # order-by date, working back from the lead time
    alerts["order_by_days"] = np.round(
        alerts["days_to_stockout"] - alerts["lead_time_mean_days"]
    )

    alerts["action"] = np.where(
        alerts["order_by_days"] <= 0,
        "ORDER NOW - already inside lead time",
        "Order within " + alerts["order_by_days"].astype(int).astype(str) + " days",
    )

    columns = [
        "category", "stock_status", "urgency_rank", "on_hand_units",
        "reorder_point_units", "safety_stock_units", "recommended_order_units",
        "weeks_of_cover", "lead_time_mean_days", "lead_time_p95_days",
        "days_to_stockout", "order_by_days", "action",
        "forecast_units_horizon", "projected_shortfall_units", "simulated_stock",
    ]
    alerts = alerts[columns].sort_values(
        ["urgency_rank", "projected_shortfall_units"], ascending=[False, False]
    )

    log.info(
        "ops.reorder_alerts: %d categories need action (%d already inside lead time)",
        len(alerts),
        int((alerts["order_by_days"] <= 0).sum()),
    )
    return alerts.reset_index(drop=True)


def build_supplier_risk(
    silver_lead_times: pd.DataFrame, silver_order_lines: pd.DataFrame
) -> pd.DataFrame:
    """Lead-time risk by category and seller state.

    Risk here is not slowness, it is unpredictability. The coefficient of
    variation is the ranking metric because a supplier you cannot plan around
    costs more in buffer stock than one that is merely slow but consistent.
    """
    frame = silver_lead_times.copy()

    frame["lead_time_cv"] = frame["lead_time_std_days"] / frame[
        "lead_time_mean_days"
    ].replace(0, np.nan)
    frame["tail_ratio_p95_p50"] = frame["lead_time_p95_days"] / frame[
        "lead_time_p50_days"
    ].replace(0, np.nan)

    late = (
        silver_order_lines.groupby(["category", "seller_state"], as_index=False)
        .agg(
            late_delivery_rate=("delivered_late", "mean"),
            avg_days_late=("days_late", "mean"),
            revenue=("line_revenue", "sum"),
        )
    )
    frame = frame.merge(late, on=["category", "seller_state"], how="left")

    def _flag(row) -> str:
        if row["lead_time_cv"] >= 0.6 or row["tail_ratio_p95_p50"] >= 2.5:
            return "HIGH"
        if row["lead_time_cv"] >= 0.35 or (row.get("late_delivery_rate") or 0) > 0.15:
            return "MEDIUM"
        return "LOW"

    frame["risk_flag"] = frame.apply(_flag, axis=1)
    frame["revenue_exposed"] = frame["revenue"].fillna(0)

    log.info(
        "ops.supplier_risk: %d category/state lanes -- %s",
        len(frame),
        frame["risk_flag"].value_counts().to_dict(),
    )
    return frame.sort_values(
        ["risk_flag", "revenue_exposed"], ascending=[True, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# executive extracts
# ---------------------------------------------------------------------
def build_revenue_trend(silver_order_lines: pd.DataFrame) -> pd.DataFrame:
    """Company-level monthly performance with growth rates."""
    monthly = (
        silver_order_lines.groupby("order_month", as_index=False)
        .agg(
            revenue=("line_revenue", "sum"),
            orders=("order_id", "nunique"),
            items=("order_item_id", "count"),
            customers=("customer_unique_id", "nunique"),
            avg_freight=("freight_value", "mean"),
            late_delivery_rate=("delivered_late", "mean"),
        )
        .rename(columns={"order_month": "month"})
        .sort_values("month")
    )

    # Drop trailing partial months. The extract stops mid-month, so the final
    # bucket holds a handful of orders and renders as a ~100% collapse. Left in,
    # it is the first thing a leader sees and the only thing they ask about.
    if len(monthly) > 1:
        median_orders = monthly["orders"].median()
        while len(monthly) > 1 and monthly["orders"].iloc[-1] < 0.25 * median_orders:
            dropped = monthly["month"].iloc[-1]
            monthly = monthly.iloc[:-1]
            log.info(
                "exec.revenue_trend: dropped partial month %s (%d orders vs "
                "median %d)",
                pd.Timestamp(dropped).date(),
                monthly["orders"].iloc[-1] if len(monthly) else 0,
                median_orders,
            )

    monthly["avg_order_value"] = monthly["revenue"] / monthly["orders"]
    monthly["items_per_order"] = monthly["items"] / monthly["orders"]
    monthly["revenue_mom_pct"] = monthly["revenue"].pct_change() * 100
    monthly["revenue_yoy_pct"] = monthly["revenue"].pct_change(12) * 100
    monthly["revenue_3m_ma"] = monthly["revenue"].rolling(3, min_periods=1).mean()
    monthly["cumulative_revenue"] = monthly["revenue"].cumsum()

    log.info(
        "exec.revenue_trend: %d months, R$%.1fM total",
        len(monthly),
        monthly["revenue"].sum() / 1e6,
    )
    return monthly


def build_segment_value(
    gold_rfm_segments: pd.DataFrame, churn_scores: pd.DataFrame
) -> pd.DataFrame:
    """Customer value distribution by RFM segment, with churn exposure."""
    merged = gold_rfm_segments.merge(
        churn_scores[
            [
                "customer_unique_id",
                "churn_risk",
                "relative_churn_risk",
                "revenue_at_risk",
                "expected_retained_revenue",
            ]
        ],
        on="customer_unique_id",
        how="left",
    )

    summary = (
        merged.groupby("segment", as_index=False)
        .agg(
            customers=("customer_unique_id", "count"),
            revenue=("monetary", "sum"),
            avg_order_value=("avg_order_value", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_review_score=("avg_review_score", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            return_rate=("returned", "mean"),
            avg_churn_risk=("churn_risk", "mean"),
            relative_churn_risk=("relative_churn_risk", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            expected_retained_revenue=("expected_retained_revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )

    summary["revenue_share_pct"] = 100 * summary["revenue"] / summary["revenue"].sum()
    summary["customer_share_pct"] = (
        100 * summary["customers"] / summary["customers"].sum()
    )
    summary["revenue_per_customer"] = summary["revenue"] / summary["customers"]
    # value concentration: the number that justifies segment-level investment
    summary["value_index"] = (
        summary["revenue_share_pct"] / summary["customer_share_pct"]
    )

    log.info("exec.segment_value: %d segments", len(summary))
    return summary


def build_churn_exposure(churn_scores: pd.DataFrame) -> pd.DataFrame:
    """Revenue at risk by state, for regional investment decisions."""
    by_state = (
        churn_scores.groupby("state", as_index=False)
        .agg(
            customers=("customer_unique_id", "count"),
            revenue=("monetary", "sum"),
            avg_churn_risk=("churn_risk", "mean"),
            relative_churn_risk=("relative_churn_risk", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            expected_retained_revenue=("expected_retained_revenue", "sum"),
            avg_review_score=("avg_review_score", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            late_delivery_rate=("any_late_delivery", "mean"),
            actual_return_rate=("returned", "mean"),
        )
        .sort_values("revenue_at_risk", ascending=False)
    )

    by_state["revenue_at_risk_pct"] = (
        100 * by_state["revenue_at_risk"] / by_state["revenue"]
    )
    by_state["share_of_total_exposure"] = (
        100 * by_state["revenue_at_risk"] / by_state["revenue_at_risk"].sum()
    )

    log.info(
        "exec.churn_exposure: %d states, R$%.2fM total revenue at risk",
        len(by_state),
        by_state["revenue_at_risk"].sum() / 1e6,
    )
    return by_state


def build_sentiment_trend(
    silver_review_sentiment: pd.DataFrame, silver_order_lines: pd.DataFrame
) -> pd.DataFrame:
    """Monthly sentiment and rating trend, overall and by category.

    Joined to order lines so sentiment can be read against the category mix
    and the delivery performance of the same period -- sentiment moving on its
    own is an observation; sentiment moving with late deliveries is a cause.
    """
    lines = silver_order_lines[
        ["order_id", "category", "order_month", "delivered_late", "days_late"]
    ].drop_duplicates("order_id")

    merged = silver_review_sentiment.merge(lines, on="order_id", how="inner")

    monthly = (
        merged.groupby(["order_month", "category"], as_index=False)
        .agg(
            reviews=("review_id", "count"),
            reviews_with_text=("has_text", "sum"),
            avg_sentiment=("sentiment", "mean"),
            avg_review_score=("review_score", "mean"),
            pct_negative=("sentiment", lambda s: float((s < -0.2).mean())),
            late_delivery_rate=("delivered_late", "mean"),
            avg_days_late=("days_late", "mean"),
        )
        .rename(columns={"order_month": "month"})
        .sort_values(["month", "category"])
    )

    log.info(
        "exec.sentiment_trend: %d month/category rows across %d months",
        len(monthly),
        monthly["month"].nunique(),
    )
    return monthly
