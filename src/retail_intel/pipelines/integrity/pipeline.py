"""Integrity pipeline: cross-system referential checks and quarantine."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.check_review_integrity,
                inputs=["bronze_reviews", "pg_valid_order_ids", "params:integrity"],
                outputs=["silver_reviews", "quarantined_reviews"],
                name="check_review_referential_integrity",
                tags=["silver", "quality"],
            ),
            node(
                func=nodes.check_session_integrity,
                inputs=["bronze_sessions", "pg_valid_customer_ids", "params:integrity"],
                outputs=["silver_sessions", "quarantined_sessions"],
                name="check_session_referential_integrity",
                tags=["silver", "quality"],
            ),
            node(
                func=nodes.combine_quarantine,
                inputs=[
                    "quarantined_reviews",
                    "quarantined_sessions",
                    "bronze_reviews",
                    "bronze_sessions",
                    "params:integrity",
                ],
                outputs="integrity_report",
                name="build_integrity_report",
                tags=["quality", "reporting"],
            ),
        ]
    )
