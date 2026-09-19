"""Serving pipeline: operational and executive dashboard extracts."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    operational = pipeline(
        [
            node(
                func=nodes.build_inventory_position,
                inputs=[
                    "demand_forecast",
                    "gold_weekly_demand",
                    "silver_lead_times",
                    "params:inventory",
                ],
                outputs="ops_inventory_position",
                name="build_inventory_position",
                tags=["serving", "operational"],
            ),
            node(
                func=nodes.build_reorder_alerts,
                inputs="ops_inventory_position",
                outputs="ops_reorder_alerts",
                name="build_reorder_alerts",
                tags=["serving", "operational"],
            ),
            node(
                func=nodes.build_supplier_risk,
                inputs=["silver_lead_times", "silver_order_lines"],
                outputs="ops_supplier_risk",
                name="build_supplier_risk",
                tags=["serving", "operational"],
            ),
        ]
    )

    executive = pipeline(
        [
            node(
                func=nodes.build_revenue_trend,
                inputs="silver_order_lines",
                outputs="exec_revenue_trend",
                name="build_revenue_trend",
                tags=["serving", "executive"],
            ),
            node(
                func=nodes.build_segment_value,
                inputs=["gold_rfm_segments", "churn_scores"],
                outputs="exec_segment_value",
                name="build_segment_value",
                tags=["serving", "executive"],
            ),
            node(
                func=nodes.build_churn_exposure,
                inputs="churn_scores",
                outputs="exec_churn_exposure",
                name="build_churn_exposure",
                tags=["serving", "executive"],
            ),
            node(
                func=nodes.build_sentiment_trend,
                inputs=["silver_review_sentiment", "silver_order_lines"],
                outputs="exec_sentiment_trend",
                name="build_sentiment_trend",
                tags=["serving", "executive"],
            ),
        ]
    )

    return operational + executive
