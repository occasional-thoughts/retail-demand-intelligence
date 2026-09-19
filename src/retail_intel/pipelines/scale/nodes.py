"""Scale layer: the carrier scan feed, processed with Spark.

Why this dataset exists
-----------------------
The operational dashboard needs supplier lead times and lead-time variability
per category. Olist publishes no such feed -- it publishes four timestamps per
order (purchased, approved, handed to carrier, delivered).

Real logistics data does not arrive as four tidy columns. It arrives as a
parcel-scan event log: one row every time a parcel is handled, tens of
millions of rows a month, dumped as flat files into object storage. So that is
what this layer reconstructs and then processes.

Honesty about what is real here:
  * The *intervals* are real. Every scan is interpolated between genuine Olist
    timestamps, so the lead times aggregated back out are the dataset's own
    lead times, not invented ones.
  * The *event granularity* is synthetic. Individual scan rows are a
    reconstruction, used to create a file large enough that single-machine
    processing is genuinely the wrong tool.

Why Spark rather than pandas
----------------------------
Exploding ~112k order lines into ~2.8M scan rows, partitioning the result by
month, and aggregating percentiles per category is a shuffle-heavy job on a
dataset several times larger than the source. It is the smallest honest
example of work that belongs on a distributed engine: pandas would hold every
intermediate in one process's memory, and the partitioned Parquet output
could not be produced at all without a writer that understands partitions.
"""
from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

log = logging.getLogger(__name__)

# The handling sequence a parcel moves through. Position in this list maps to
# position in the parcel's journey, so scan_index -> scan_type is monotonic.
SCAN_STAGES = [
    "order_received",
    "picked",
    "packed",
    "label_printed",
    "handed_to_carrier",
    "origin_hub_inbound",
    "origin_hub_outbound",
    "line_haul_departed",
    "line_haul_arrived",
    "transit_scan",
    "destination_hub_inbound",
    "destination_hub_sorted",
    "out_for_delivery",
    "delivery_attempted",
    "delivered",
]


def build_scan_events(
    bronze_orders: pd.DataFrame,
    bronze_order_items: pd.DataFrame,
    params: dict,
) -> SparkDataFrame:
    """Explode delivered order lines into a parcel-scan event feed.

    Returns a Spark DataFrame; Kedro writes it to partitioned Parquet via the
    ``scan_events_bronze`` catalog entry.
    """
    spark = SparkSession.builder.getOrCreate()

    # Only delivered orders have a complete journey to interpolate across.
    orders = bronze_orders.loc[
        bronze_orders["order_delivered_customer_date"].notna()
        & bronze_orders["order_approved_at"].notna(),
        [
            "order_id",
            "customer_state",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    ]

    seed = bronze_order_items.merge(orders, on="order_id", how="inner")[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "seller_state",
            "customer_state",
            "category",
            "price",
            "freight_value",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ]
    log.info("scale: seeding from %d delivered order lines", len(seed))

    sdf = spark.createDataFrame(seed)

    lo, hi = params["min_scans_per_line"], params["max_scans_per_line"]
    span = hi - lo

    # A deterministic per-line scan count. Hashing the natural key rather than
    # drawing at random keeps the feed reproducible across runs without having
    # to persist a seed table.
    sdf = sdf.withColumn(
        "n_scans",
        F.lit(lo) + F.pmod(F.hash(F.col("order_id"), F.col("order_item_id")), F.lit(span)),
    )

    # One row per scan. sequence() + explode() is the distributed equivalent of
    # a nested Python loop, and is where the row count multiplies ~25x.
    sdf = sdf.withColumn(
        "scan_index", F.explode(F.sequence(F.lit(0), F.col("n_scans") - 1))
    )

    approved = F.col("order_approved_at").cast("double")
    delivered = F.col("order_delivered_customer_date").cast("double")
    progress = F.col("scan_index") / F.greatest(F.col("n_scans") - 1, F.lit(1))

    # Jitter keeps scans from landing on a suspiciously perfect grid, bounded
    # so it can never reorder two scans or push one outside the real interval.
    jitter = (
        F.pmod(F.hash(F.col("order_id"), F.col("scan_index")), F.lit(1000)) / 1000.0 - 0.5
    ) * (delivered - approved) / F.greatest(F.col("n_scans") * 4, F.lit(1))

    sdf = sdf.withColumn(
        "scan_ts",
        F.to_timestamp(
            F.least(
                F.greatest(approved + (delivered - approved) * progress + jitter, approved),
                delivered,
            )
        ),
    )

    stage_expr = F.element_at(
        F.array([F.lit(stage) for stage in SCAN_STAGES]),
        F.least(
            F.floor(progress * F.lit(len(SCAN_STAGES))) + 1, F.lit(len(SCAN_STAGES))
        ).cast("int"),
    )

    sdf = (
        sdf.withColumn("scan_type", stage_expr)
        .withColumn(
            "facility",
            F.concat_ws(
                "-",
                F.when(progress < 0.35, F.col("seller_state"))
                .when(progress < 0.7, F.lit("BR"))
                .otherwise(F.col("customer_state")),
                F.lit("HUB"),
                F.lpad(
                    (F.pmod(F.hash(F.col("order_id")), F.lit(24)) + 1).cast("string"), 2, "0"
                ),
            ),
        )
        .withColumn("scan_year", F.year("scan_ts"))
        .withColumn("scan_month", F.month("scan_ts"))
        .withColumn(
            "is_exception",
            (F.col("scan_type") == "delivery_attempted")
            & (F.pmod(F.hash(F.col("order_id"), F.col("scan_index")), F.lit(100)) < 18),
        )
        .drop("n_scans")
    )

    log.info("scale: scan feed built (lazy); partitioned by scan_year/scan_month on write")
    return sdf


def summarise_scan_feed(scan_events_bronze: SparkDataFrame) -> dict:
    """Report the size and span of the feed actually written.

    Counting is a wide operation, so it is done once, here, and the result is
    persisted -- rather than being recomputed by every node that wants to know
    how big the feed is.
    """
    summary = scan_events_bronze.agg(
        F.count("*").alias("scan_rows"),
        F.countDistinct("order_id").alias("orders"),
        F.min("scan_ts").alias("first_scan"),
        F.max("scan_ts").alias("last_scan"),
        F.sum(F.col("is_exception").cast("int")).alias("exception_scans"),
    ).collect()[0]

    report = {
        "scan_rows": int(summary["scan_rows"]),
        "orders_covered": int(summary["orders"]),
        "first_scan": str(summary["first_scan"]),
        "last_scan": str(summary["last_scan"]),
        "exception_scans": int(summary["exception_scans"]),
    }
    log.info(
        "scale: %s scan rows across %s orders",
        f"{report['scan_rows']:,}",
        f"{report['orders_covered']:,}",
    )
    return report


def aggregate_lead_times(scan_events_bronze: SparkDataFrame) -> pd.DataFrame:
    """Derive supplier lead time and its variability, per category and seller state.

    Lead time is measured from the seller accepting the order to the parcel
    reaching the customer. The dashboard needs the *spread* as much as the
    average: safety stock is a function of demand variability and lead-time
    variability, so a supplier with a 10-day mean and a 9-day standard
    deviation is a materially worse risk than one with a 14-day mean and a
    1-day standard deviation.

    The aggregation runs in Spark over the full scan feed and returns a small
    frame -- a few thousand rows -- which is the right point to hand back to
    pandas.
    """
    per_line = scan_events_bronze.groupBy(
        "order_id", "order_item_id", "category", "seller_state", "customer_state"
    ).agg(
        F.min("scan_ts").alias("first_scan"),
        F.max("scan_ts").alias("last_scan"),
        F.count("*").alias("scans"),
        F.sum(F.col("is_exception").cast("int")).alias("exceptions"),
    )

    per_line = per_line.withColumn(
        "lead_time_days",
        (F.col("last_scan").cast("double") - F.col("first_scan").cast("double")) / 86400.0,
    ).filter(F.col("lead_time_days") >= 0)

    aggregated = (
        per_line.groupBy("category", "seller_state")
        .agg(
            F.count("*").alias("shipments"),
            F.avg("lead_time_days").alias("lead_time_mean_days"),
            F.stddev_samp("lead_time_days").alias("lead_time_std_days"),
            F.expr("percentile_approx(lead_time_days, 0.5)").alias("lead_time_p50_days"),
            F.expr("percentile_approx(lead_time_days, 0.95)").alias("lead_time_p95_days"),
            # Spark will not average a boolean; the cast makes the intent
            # explicit rather than relying on an implicit promotion.
            F.avg((F.col("exceptions") > 0).cast("int")).alias("exception_rate"),
        )
        .filter(F.col("shipments") >= 5)
    )

    frame = aggregated.toPandas()
    # A single-shipment group has no meaningful spread; fall back to 0 rather
    # than propagating nulls into the safety-stock formula.
    frame["lead_time_std_days"] = frame["lead_time_std_days"].fillna(0.0)
    log.info(
        "scale: lead times for %d category/state pairs (median %.1f days)",
        len(frame),
        frame["lead_time_p50_days"].median(),
    )
    return frame
