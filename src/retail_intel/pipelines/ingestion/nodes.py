"""Bronze layer: land each source system's extract, unmodified in substance.

Bronze nodes do the minimum: coerce types that the connectors get wrong, and
persist a snapshot. No business logic, no joins, no filtering. Keeping this
layer dumb is what makes it possible to re-run everything downstream without
touching the source systems again.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

ORDER_TIMESTAMPS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def land_orders(pg_orders: pd.DataFrame) -> pd.DataFrame:
    """Snapshot the order header extract."""
    frame = pg_orders.copy()
    for column in ORDER_TIMESTAMPS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    log.info(
        "bronze.orders: %d rows, %s -> %s",
        len(frame),
        frame["order_purchase_timestamp"].min().date(),
        frame["order_purchase_timestamp"].max().date(),
    )
    return frame


def land_order_items(pg_order_items: pd.DataFrame) -> pd.DataFrame:
    """Snapshot the order line extract."""
    frame = pg_order_items.copy()
    frame["shipping_limit_date"] = pd.to_datetime(
        frame["shipping_limit_date"], errors="coerce"
    )
    for column in ("price", "freight_value", "product_weight_g"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    log.info(
        "bronze.order_items: %d rows, %d distinct products, %d categories",
        len(frame),
        frame["product_id"].nunique(),
        frame["category"].nunique(),
    )
    return frame


def land_reviews(mongo_reviews: pd.DataFrame) -> pd.DataFrame:
    """Snapshot review documents.

    Documents have optional fields, so a column may be absent entirely when no
    document in the collection carries it. Missing columns are materialised
    here so that every downstream node can rely on a stable frame shape.
    """
    frame = mongo_reviews.copy()
    for column in ("title", "message"):
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame["has_text"] = frame["message"].notna() & (
        frame["message"].astype("string").str.strip().str.len() > 0
    )
    log.info(
        "bronze.reviews: %d documents, %d carry free text (%.1f%%)",
        len(frame),
        int(frame["has_text"].sum()),
        100 * frame["has_text"].mean(),
    )
    return frame


def land_sessions(
    mongo_session_features: pd.DataFrame, mongo_cart_behaviour: pd.DataFrame
) -> pd.DataFrame:
    """Combine the two server-side session aggregations into one frame.

    The cart-behaviour aggregation arrives long (one row per customer per
    event type) and is pivoted to one row per customer here.
    """
    sessions = mongo_session_features.copy()
    if "customer_id" not in sessions.columns:
        raise KeyError(
            "mongo_session_features has no customer_id column -- the aggregation "
            "pipeline in the catalog must promote the group key with a $set stage, "
            f"got columns: {list(sessions.columns)}"
        )

    counts = (
        mongo_cart_behaviour.pivot_table(
            index="customer_id", columns="event_type", values="n", fill_value=0
        )
        .add_prefix("n_")
        .reset_index()
    )
    dwell = (
        mongo_cart_behaviour.pivot_table(
            index="customer_id", columns="event_type", values="mean_dwell_ms"
        )
        .add_prefix("dwell_")
        .reset_index()
    )

    frame = sessions.merge(counts, on="customer_id", how="left").merge(
        dwell, on="customer_id", how="left"
    )
    for column in ("first_seen", "last_seen"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    log.info("bronze.sessions: %d customers with browsing history", len(frame))
    return frame
