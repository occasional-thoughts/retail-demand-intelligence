"""Assemble the data payloads the published dashboards embed.

The Tableau workbooks read the extracts live from disk. A published web page
cannot, so the figures are embedded at build time. Same principle either way:
the dashboards are generated from the pipeline's output, so they cannot drift
from it.

    python scripts/build_web_dashboards.py

Writes dashboards/*.json, which the HTML templates inline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("dashboards")

ROOT = Path(__file__).resolve().parents[1]
REPORTING = ROOT / "data" / "08_reporting"
GOLD = ROOT / "data" / "03_gold"
OUT = ROOT / "dashboards"


def _records(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> list[dict]:
    """Round aggressively -- 12 decimal places of a forecast is noise, and it
    triples the size of the embedded payload."""
    trimmed = frame[columns].copy()
    for column in trimmed.columns:
        if pd.api.types.is_float_dtype(trimmed[column]):
            trimmed[column] = trimmed[column].round(decimals)
    return trimmed.to_dict("records")


def build_operations() -> dict:
    alerts = pd.read_csv(REPORTING / "ops_reorder_alerts.csv")
    position = pd.read_csv(REPORTING / "ops_inventory_position.csv")
    supplier = pd.read_csv(REPORTING / "ops_supplier_risk.csv")
    metrics = json.loads((GOLD / "demand_metrics.json").read_text())

    high_risk = supplier[supplier["risk_flag"] == "HIGH"].nlargest(12, "revenue_exposed")

    return {
        "generated": pd.Timestamp.now().strftime("%d %b %Y"),
        "kpi": {
            "critical": int((position["stock_status"] == "CRITICAL").sum()),
            "reorder": int((position["stock_status"] == "REORDER").sum()),
            "healthy": int((position["stock_status"] == "HEALTHY").sum()),
            "categories": int(len(position)),
            "order_units": round(float(alerts["recommended_order_units"].sum())),
            "shortfall_units": round(float(position["projected_shortfall_units"].sum())),
            "inside_lead_time": int((alerts["order_by_days"] <= 0).sum()),
            "horizon_weeks": int(position["horizon_weeks"].iloc[0]),
        },
        "forecast": {
            "wape": metrics["model"]["wape"],
            "naive_wape": metrics["baseline_naive_last_week"]["wape"],
            "improvement": metrics["wape_improvement_vs_naive"],
        },
        "alerts": _records(alerts, [
            "category", "stock_status", "on_hand_units", "reorder_point_units",
            "recommended_order_units", "weeks_of_cover", "lead_time_mean_days",
            "days_to_stockout", "order_by_days", "projected_shortfall_units",
            "forecast_units_horizon", "action",
        ], 1),
        "position": _records(position, [
            "category", "stock_status", "weeks_of_cover", "lead_time_mean_days",
            "forecast_units_horizon", "on_hand_units", "projected_shortfall_units",
            "safety_stock_units", "reorder_point_units",
        ], 2),
        "supplier": _records(high_risk, [
            "category", "seller_state", "lead_time_mean_days", "lead_time_p50_days",
            "lead_time_p95_days", "lead_time_cv", "late_delivery_rate",
            "revenue_exposed", "shipments",
        ], 3),
        "supplier_summary": {
            "high": int((supplier["risk_flag"] == "HIGH").sum()),
            "medium": int((supplier["risk_flag"] == "MEDIUM").sum()),
            "low": int((supplier["risk_flag"] == "LOW").sum()),
            "high_revenue": float(
                supplier.loc[supplier["risk_flag"] == "HIGH", "revenue_exposed"].sum()
            ),
            "high_cv": round(float(
                supplier.loc[supplier["risk_flag"] == "HIGH", "lead_time_cv"].mean()), 3),
            "low_cv": round(float(
                supplier.loc[supplier["risk_flag"] == "LOW", "lead_time_cv"].mean()), 3),
            "high_p95": round(float(
                supplier.loc[supplier["risk_flag"] == "HIGH", "lead_time_p95_days"].mean()), 1),
            "low_p95": round(float(
                supplier.loc[supplier["risk_flag"] == "LOW", "lead_time_p95_days"].mean()), 1),
        },
    }


def build_executive() -> dict:
    revenue = pd.read_csv(REPORTING / "exec_revenue_trend.csv")
    segments = pd.read_csv(REPORTING / "exec_segment_value.csv")
    churn = pd.read_csv(REPORTING / "exec_churn_exposure.csv")
    sentiment = pd.read_csv(REPORTING / "exec_sentiment_trend.csv")
    ablation = json.loads((GOLD / "churn_ablation.json").read_text())
    scores = pd.read_parquet(GOLD / "churn_scores.parquet")

    # sentiment is one row per month per category; weight by review volume so a
    # category with four reviews cannot swing the company-level line
    monthly = (
        sentiment.assign(w=lambda f: f["avg_sentiment"] * f["reviews"],
                         wl=lambda f: f["late_delivery_rate"] * f["reviews"])
        .groupby("month", as_index=False)
        .agg(reviews=("reviews", "sum"), w=("w", "sum"), wl=("wl", "sum"))
    )
    monthly["avg_sentiment"] = (monthly["w"] / monthly["reviews"]).round(4)
    monthly["late_delivery_rate"] = (monthly["wl"] / monthly["reviews"]).round(4)
    monthly = monthly[monthly["reviews"] >= 50]

    by_category = (
        sentiment.assign(w=lambda f: f["avg_sentiment"] * f["reviews"])
        .groupby("category", as_index=False)
        .agg(reviews=("reviews", "sum"), w=("w", "sum"))
    )
    by_category["avg_sentiment"] = (by_category["w"] / by_category["reviews"]).round(4)
    by_category = by_category[by_category["reviews"] >= 300].nsmallest(12, "avg_sentiment")

    bands = (
        scores.groupby("risk_band", observed=True)
        .agg(customers=("customer_unique_id", "count"),
             returned=("returned", "mean"),
             revenue=("monetary", "sum"))
        .reset_index()
    )
    bands["lift"] = (bands["returned"] / scores["returned"].mean()).round(2)

    return {
        "generated": pd.Timestamp.now().strftime("%d %b %Y"),
        "kpi": {
            "revenue": float(revenue["revenue"].sum()),
            "orders": int(revenue["orders"].sum()),
            "months": int(len(revenue)),
            "customers": int(churn["customers"].sum()),
            "return_rate": round(float(scores["returned"].mean()), 5),
            "revenue_at_risk": float(churn["revenue_at_risk"].sum()),
        },
        "revenue": _records(revenue, [
            "month", "revenue", "orders", "avg_order_value",
            "late_delivery_rate", "revenue_3m_ma",
        ], 2),
        "segments": _records(segments, [
            "segment", "customers", "revenue", "revenue_share_pct",
            "customer_share_pct", "value_index", "return_rate",
            "avg_order_value", "avg_review_score",
        ], 3),
        "states": _records(churn.nlargest(12, "revenue_at_risk"), [
            "state", "customers", "revenue", "revenue_at_risk",
            "share_of_total_exposure", "late_delivery_rate", "avg_sentiment",
        ], 3),
        "sentiment_monthly": _records(
            monthly, ["month", "avg_sentiment", "late_delivery_rate", "reviews"], 4),
        "sentiment_categories": _records(
            by_category, ["category", "avg_sentiment", "reviews"], 4),
        "risk_bands": _records(
            bands, ["risk_band", "customers", "returned", "revenue", "lift"], 4),
        "ablation": {
            "question": ablation["question"],
            "with": ablation["results"]["with_sentiment"],
            "without": ablation["results"]["without_sentiment"],
            "delta": ablation["delta"]["roc_auc"],
            "ci": ablation["bootstrap_auc_delta"]["ci_95"],
            "verdict": ablation["verdict"],
        },
    }


def inject(name: str, payload: dict) -> Path:
    """Inline the payload into the page template.

    A published page cannot read a local CSV, so the figures travel with the
    HTML. The template holds a single __DATA__ token rather than a hand-pasted
    blob, so re-running the pipeline and re-running this script is the whole
    update path -- the numbers on the page can never be stale relative to the
    numbers in the repo.
    """
    template = (OUT / f"{name}.html").read_text()
    if "__DATA__" not in template:
        raise ValueError(
            f"{name}.html has no __DATA__ token. It may already contain a built "
            f"payload -- rebuild from the template rather than editing in place."
        )

    # </script> inside the JSON would close the host script tag early
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    built = OUT / "build" / f"{name}.html"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_text(template.replace("__DATA__", blob))
    return built


def main() -> int:
    OUT.mkdir(exist_ok=True)
    profile = ROOT / "quality" / "source_profile.json"
    pages = [("operations", build_operations()), ("executive", build_executive())]
    if profile.exists():
        pages.append(("quality", json.loads(profile.read_text())))
    else:
        log.warning("skipping quality page: run scripts/profile_sources.py first")

    for name, payload in pages:
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        built = inject(name, payload)
        log.info("%-10s payload %3.0f KB -> %s (%.0f KB)",
                 name, path.stat().st_size / 1024,
                 built.relative_to(ROOT), built.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
