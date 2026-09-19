"""Ingestion pipeline: three source systems -> the bronze layer."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.land_orders,
                inputs="pg_orders",
                outputs="bronze_orders",
                name="land_orders_from_postgres",
                tags=["bronze", "sql"],
            ),
            node(
                func=nodes.land_order_items,
                inputs="pg_order_items",
                outputs="bronze_order_items",
                name="land_order_items_from_postgres",
                tags=["bronze", "sql"],
            ),
            node(
                func=nodes.land_reviews,
                inputs="mongo_reviews",
                outputs="bronze_reviews",
                name="land_reviews_from_mongo",
                tags=["bronze", "nosql"],
            ),
            node(
                func=nodes.land_sessions,
                inputs=["mongo_session_features", "mongo_cart_behaviour"],
                outputs="bronze_sessions",
                name="land_sessions_from_mongo",
                tags=["bronze", "nosql"],
            ),
        ]
    )
