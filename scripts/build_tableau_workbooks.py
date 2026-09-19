"""Generate the two Tableau workbooks from the pipeline's serving extracts.

Two workbooks, because they are two data products for two audiences -- not one
report filtered twice. The split is enforced structurally: the operational
workbook may only read `ops_*.csv` and the executive workbook only `exec_*.csv`,
and that rule is asserted at the bottom of this file.

    python scripts/build_tableau_workbooks.py

Outputs tableau/*.twb, which open directly in Tableau Public Desktop.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tableau._twb import Dashboard, DataSource, Worksheet, Zone, write_workbook

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("tableau")

ROOT = Path(__file__).resolve().parents[1]
REPORTING = ROOT / "data" / "08_reporting"
OUT = ROOT / "tableau"

SIMULATED_NOTE = (
    "Stock positions are simulated from a documented policy (6 weeks opening "
    "cover, 95% service level). Forecast demand and supplier lead times are "
    "derived from actual order and delivery data."
)


def build_operations() -> None:
    """Inventory Operations Command Centre -- forward-looking, SKU-grain, daily."""
    alerts = DataSource((REPORTING / "ops_reorder_alerts.csv").resolve(), "Reorder Alerts")
    position = DataSource((REPORTING / "ops_inventory_position.csv").resolve(), "Inventory Position")
    supplier = DataSource((REPORTING / "ops_supplier_risk.csv").resolve(), "Supplier Risk")

    sheets = [
        # The single most important object: a to-do list, ranked so the top row
        # is the first thing to do this morning.
        Worksheet(
            name="Action Queue",
            source=alerts,
            rows=[alerts.dim("category")],
            cols=[alerts.measure("recommended_order_units")],
            mark="Bar",
            encodings={"color": alerts.dim("stock_status")},
        ),
        # How long before each one actually runs out -- the urgency behind the queue.
        Worksheet(
            name="Days To Stockout",
            source=alerts,
            rows=[alerts.dim("category")],
            cols=[alerts.measure("days_to_stockout")],
            mark="Bar",
            encodings={"color": alerts.dim("stock_status")},
        ),
        # Anything whose cover is below its lead time cannot be replenished in
        # time. That relationship is the insight; the points are context.
        Worksheet(
            name="Cover vs Lead Time",
            source=position,
            cols=[position.measure("lead_time_mean_days", "avg")],
            rows=[position.measure("weeks_of_cover", "avg")],
            mark="Circle",
            encodings={
                "color": position.dim("stock_status"),
                "size": position.measure("forecast_units_horizon"),
                "text": position.dim("category"),
            },
        ),
        # The size of the gap, ranked.
        Worksheet(
            name="Projected Shortfall",
            source=position,
            rows=[position.dim("category")],
            cols=[position.measure("projected_shortfall_units")],
            mark="Bar",
            encodings={"color": position.dim("stock_status")},
        ),
        # Risk is unpredictability, not slowness: colour by coefficient of
        # variation rather than by mean lead time.
        Worksheet(
            name="Supplier Lane Risk",
            source=supplier,
            rows=[supplier.dim("category")],
            cols=[supplier.dim("seller_state")],
            mark="Square",
            encodings={"color": supplier.measure("lead_time_cv", "avg")},
            filters=[("risk_flag", ["HIGH", "MEDIUM"])],
        ),
    ]

    dashboard = Dashboard(
        name="Operations Command Centre",
        width=1600,
        height=900,
        zones=[
            Zone(None, 0, 0, 100000, 6000,
                 text="INVENTORY OPERATIONS COMMAND CENTRE - action required this week"),
            Zone("Action Queue", 0, 6000, 50000, 44000),
            Zone("Days To Stockout", 0, 50000, 50000, 44000),
            Zone("Cover vs Lead Time", 50000, 6000, 50000, 44000),
            Zone("Projected Shortfall", 50000, 50000, 50000, 38000),
            Zone(None, 50000, 88000, 50000, 6000, text=SIMULATED_NOTE),
        ],
    )

    path = OUT / "operations_command_centre.twb"
    write_workbook(path, [alerts, position, supplier], sheets, [dashboard])
    log.info("wrote %s (%d sheets)", path.relative_to(ROOT), len(sheets))


def build_executive() -> None:
    """Retail Performance & Customer Intelligence -- backward-looking, monthly."""
    revenue = DataSource((REPORTING / "exec_revenue_trend.csv").resolve(), "Revenue Trend")
    segments = DataSource((REPORTING / "exec_segment_value.csv").resolve(), "Segment Value")
    churn = DataSource((REPORTING / "exec_churn_exposure.csv").resolve(), "Churn Exposure")
    sentiment = DataSource((REPORTING / "exec_sentiment_trend.csv").resolve(), "Sentiment Trend")

    sheets = [
        # One chart, two years, no filters. Readable standing up.
        Worksheet(
            name="Revenue Trajectory",
            source=revenue,
            cols=[revenue.date("month")],
            rows=[revenue.measure("revenue")],
            mark="Line",
        ),
        Worksheet(
            name="Average Order Value",
            source=revenue,
            cols=[revenue.date("month")],
            rows=[revenue.measure("avg_order_value", "avg")],
            mark="Line",
        ),
        # Distance from the 45-degree line is the argument for where retention
        # budget should go: segments earning more than their headcount.
        Worksheet(
            name="Where The Value Is",
            source=segments,
            cols=[segments.measure("customer_share_pct")],
            rows=[segments.measure("revenue_share_pct")],
            mark="Circle",
            encodings={
                "color": segments.measure("relative_churn_risk", "avg"),
                "size": segments.measure("revenue"),
                "text": segments.dim("segment"),
            },
        ),
        Worksheet(
            name="Churn Exposure By State",
            source=churn,
            rows=[churn.dim("state")],
            cols=[churn.measure("revenue_at_risk")],
            mark="Bar",
            encodings={"color": churn.measure("late_delivery_rate", "avg")},
        ),
        # If these two move together, sentiment is a logistics problem rather
        # than a product problem -- a different remediation budget entirely.
        Worksheet(
            name="Sentiment vs Late Delivery",
            source=sentiment,
            cols=[sentiment.measure("late_delivery_rate", "avg")],
            rows=[sentiment.measure("avg_sentiment", "avg")],
            mark="Circle",
            encodings={"color": sentiment.date("month")},
        ),
        Worksheet(
            name="Category Sentiment Risk",
            source=sentiment,
            rows=[sentiment.dim("category")],
            cols=[sentiment.measure("avg_sentiment", "avg")],
            mark="Bar",
            encodings={"color": sentiment.measure("avg_sentiment", "avg")},
        ),
    ]

    dashboard = Dashboard(
        name="Executive Review",
        width=1200,
        height=1600,
        zones=[
            Zone(None, 0, 0, 100000, 4000,
                 text="RETAIL PERFORMANCE & CUSTOMER INTELLIGENCE - monthly leadership review"),
            Zone("Revenue Trajectory", 0, 4000, 100000, 24000),
            Zone("Average Order Value", 0, 28000, 100000, 16000),
            Zone("Where The Value Is", 0, 44000, 50000, 28000),
            Zone("Churn Exposure By State", 50000, 44000, 50000, 28000),
            Zone("Sentiment vs Late Delivery", 0, 72000, 50000, 28000),
            Zone("Category Sentiment Risk", 50000, 72000, 50000, 28000),
        ],
    )

    path = OUT / "executive_review.twb"
    write_workbook(path, [revenue, segments, churn, sentiment], sheets, [dashboard])
    log.info("wrote %s (%d sheets)", path.relative_to(ROOT), len(sheets))


def assert_audiences_are_separate() -> None:
    """The two workbooks must not share a data source.

    The brief's hard requirement is that these are designed for different
    audiences rather than being the same charts in two places. Sharing an
    extract is the first step toward the second thing, so it is checked rather
    than trusted.
    """
    operations = (OUT / "operations_command_centre.twb").read_text()
    executive = (OUT / "executive_review.twb").read_text()

    if "exec_" in operations:
        raise AssertionError("the operational workbook references an executive extract")
    if "ops_" in executive:
        raise AssertionError("the executive workbook references an operational extract")
    if SIMULATED_NOTE[:40] not in operations:
        raise AssertionError("the operational workbook is missing the simulated-data notice")
    log.info("audience separation verified: no shared extracts")


if __name__ == "__main__":
    missing = [
        name for name in (
            "ops_reorder_alerts", "ops_inventory_position", "ops_supplier_risk",
            "exec_revenue_trend", "exec_segment_value", "exec_churn_exposure",
            "exec_sentiment_trend",
        )
        if not (REPORTING / f"{name}.csv").exists()
    ]
    if missing:
        raise SystemExit(
            f"missing extracts: {missing}. Run `make pipeline` first."
        )

    build_operations()
    build_executive()
    assert_audiences_are_separate()
    log.info("done -- open the .twb files in Tableau Public Desktop")
