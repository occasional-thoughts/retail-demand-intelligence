"""Feature pipeline: silver conformance and the two gold feature stores."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.build_order_lines,
                inputs=["bronze_orders", "bronze_order_items", "pg_payments", "params:forecasting"],
                outputs="silver_order_lines",
                name="conform_order_lines",
                tags=["silver"],
            ),
            node(
                func=nodes.build_weekly_demand,
                inputs=["silver_order_lines", "params:forecasting"],
                outputs="gold_weekly_demand",
                name="aggregate_weekly_demand",
                tags=["gold", "demand"],
            ),
            node(
                func=nodes.build_demand_features,
                inputs=["gold_weekly_demand", "params:forecasting"],
                outputs="gold_demand_features",
                name="engineer_demand_features",
                tags=["gold", "demand"],
            ),
            node(
                func=nodes.build_review_sentiment,
                inputs=["silver_reviews", "params:sentiment"],
                outputs="silver_review_sentiment",
                name="score_review_sentiment",
                tags=["silver", "nlp"],
            ),
            node(
                func=nodes.build_customer_features,
                inputs=[
                    "silver_order_lines",
                    "silver_sessions",
                    "silver_review_sentiment",
                    "params:churn",
                ],
                outputs="gold_customer_features",
                name="engineer_customer_features",
                tags=["gold", "customer"],
            ),
            node(
                func=nodes.build_rfm_segments,
                inputs=["gold_customer_features", "params:rfm"],
                outputs="gold_rfm_segments",
                name="build_rfm_segments",
                tags=["gold", "customer"],
            ),
        ]
    )
