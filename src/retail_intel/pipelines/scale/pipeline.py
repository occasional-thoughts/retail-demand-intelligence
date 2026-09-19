"""Scale pipeline: build and aggregate the carrier scan feed with Spark."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.build_scan_events,
                inputs=["bronze_orders", "bronze_order_items", "params:scale"],
                outputs="scan_events_bronze",
                name="build_carrier_scan_feed",
                tags=["bronze", "spark"],
            ),
            node(
                func=nodes.summarise_scan_feed,
                inputs="scan_events_bronze",
                outputs="scan_feed_summary",
                name="summarise_carrier_scan_feed",
                tags=["spark", "quality"],
            ),
            node(
                func=nodes.aggregate_lead_times,
                inputs="scan_events_bronze",
                outputs="silver_lead_times",
                name="aggregate_supplier_lead_times",
                tags=["silver", "spark"],
            ),
        ]
    )
