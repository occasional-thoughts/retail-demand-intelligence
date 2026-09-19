"""Build source system 1: the relational order-management database.

Loads the nine Olist CSVs into a normalised, constrained Postgres schema.

Two things make this different from `df.to_sql(...)`:

1. The schema is declared first, with real types, primary keys and foreign
   keys (see scripts/sql/01_schema.sql). The database enforces the data
   contract, rather than hoping the loader got it right.
2. Tables are loaded parent-first. Any referential problem in the source
   data surfaces here as a loud failure instead of silently becoming a
   null join key three layers downstream.

Usage:
    python scripts/load_postgres.py
"""
from __future__ import annotations

import io
import logging
import sys

import pandas as pd
import psycopg2

from _common import PG, PROJECT_ROOT, RAW_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
log = logging.getLogger("load_postgres")

# Load order is the topological order of the foreign-key graph: every table
# appears after the tables it points at.
LOAD_PLAN: list[dict] = [
    {
        "table": "category_translation",
        "csv": "product_category_name_translation.csv",
        "columns": ["product_category_name", "product_category_name_english"],
    },
    {
        "table": "customers",
        "int_columns": ["customer_zip_code_prefix"],
        "csv": "olist_customers_dataset.csv",
        "columns": [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ],
    },
    {
        "table": "sellers",
        "int_columns": ["seller_zip_code_prefix"],
        "csv": "olist_sellers_dataset.csv",
        "columns": [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ],
    },
    {
        "table": "products",
        "int_columns": ["product_name_length", "product_description_length", "product_photos_qty"],
        "csv": "olist_products_dataset.csv",
        # the source file misspells "length" in two column names; the schema
        # fixes the spelling, so the loader maps across the rename.
        "rename": {
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        },
        "columns": [
            "product_id",
            "product_category_name",
            "product_name_length",
            "product_description_length",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    },
    {
        "table": "geolocation",
        "int_columns": ["geolocation_zip_code_prefix"],
        "csv": "olist_geolocation_dataset.csv",
        "columns": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
    },
    {
        "table": "orders",
        "csv": "olist_orders_dataset.csv",
        "columns": [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    },
    {
        "table": "order_items",
        "int_columns": ["order_item_id"],
        "csv": "olist_order_items_dataset.csv",
        "columns": [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ],
    },
    {
        "table": "order_payments",
        "int_columns": ["payment_sequential", "payment_installments"],
        "csv": "olist_order_payments_dataset.csv",
        "columns": [
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
            "payment_value",
        ],
    },
    {
        "table": "order_reviews",
        "int_columns": ["review_score"],
        "csv": "olist_order_reviews_dataset.csv",
        # the free-text columns are intentionally excluded -- they are the
        # payload of source system 2 (MongoDB), not of the relational core.
        "columns": [
            "review_id",
            "order_id",
            "review_score",
            "review_creation_date",
            "review_answer_timestamp",
        ],
    },
]


def apply_schema(conn) -> None:
    ddl = (PROJECT_ROOT / "scripts" / "sql" / "01_schema.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log.info("schema applied (tables dropped and recreated)")


def copy_frame(conn, table: str, frame: pd.DataFrame) -> int:
    """Stream a DataFrame into Postgres with COPY.

    COPY is used rather than row-by-row INSERT because geolocation alone is
    ~1M rows; INSERT would take minutes where COPY takes seconds.
    """
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="\\N")
    # pandas writes the nullable-Int64 sentinel as <NA>, which COPY does not
    # recognise; normalise it to the NULL marker COPY was told to expect.
    payload = buffer.getvalue().replace(",<NA>", ",\\N")
    buffer = io.StringIO(payload)

    columns = ", ".join(frame.columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
            buffer,
        )
    conn.commit()
    return len(frame)


def load_table(conn, spec: dict) -> int:
    path = RAW_DIR / spec["csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Olist dataset into data/raw/ first."
        )

    # utf-8-sig strips the byte-order mark that the translation file carries.
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename(columns=spec.get("rename", {}))

    missing = set(spec["columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{spec['csv']} is missing expected columns: {sorted(missing)}")

    frame = frame[spec["columns"]]

    # A pandas integer column containing any null is promoted to float64, and
    # `40.0` is not valid input for a Postgres INTEGER. The nullable Int64
    # dtype keeps the nulls and drops the decimal point, so the schema's types
    # stay honest instead of being widened to accommodate the loader.
    for column in spec.get("int_columns", []):
        frame[column] = frame[column].astype("Int64")

    rows = copy_frame(conn, spec["table"], frame)
    log.info("%-22s %8d rows loaded", spec["table"], rows)
    return rows


def main() -> int:
    log.info("connecting to postgres at %s:%s/%s", PG["host"], PG["port"], PG["dbname"])
    with psycopg2.connect(**PG) as conn:
        # Olist carries Portuguese text in city names, category names and
        # review bodies. Pinning the client encoding means an accented
        # character is either stored correctly or fails loudly -- never
        # silently mangled.
        conn.set_client_encoding("UTF8")
        apply_schema(conn)
        total = sum(load_table(conn, spec) for spec in LOAD_PLAN)

        # A load that "succeeded" but wrote nothing is a silent failure; assert
        # against the known shape of the dataset so that never passes unnoticed.
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            order_count = cur.fetchone()[0]
        if order_count < 90_000:
            raise RuntimeError(
                f"expected ~99k orders, found {order_count}: load is incomplete"
            )

    log.info("done: %d rows across %d tables", total, len(LOAD_PLAN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
