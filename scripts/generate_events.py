"""Build source system 2: the web platform's event store (MongoDB).

Why this data lives in a document store rather than in Postgres:

* Review documents have genuinely optional fields -- most reviews carry no
  title, many carry no body at all, and the moderation block only exists on
  reviews that were flagged. Modelling that relationally means a wide table
  of nulls or a pile of sparse side-tables.
* A browsing session is naturally a *nested* object: one session document
  contains a variable-length array of events, each with its own shape
  depending on the event type. Flattening that into rows loses the grouping
  that every downstream query actually wants.

This script also injects the orphaned records the pipeline is required to
detect. Roughly 5% of the documents are written with order/customer
identifiers that do not exist in Postgres -- the realistic failure mode when
two systems are maintained independently and no foreign key spans them.

Every injected orphan id is written to data/quarantine/expected_orphans.json
so the integrity checks can be tested against ground truth rather than
against themselves.

Usage:
    python scripts/generate_events.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
import uuid
from datetime import timedelta

import pandas as pd
import psycopg2
from pymongo import MongoClient, ASCENDING

from _common import MONGO_DB, MONGO_URI, PG, PROJECT_ROOT, RAW_DIR, SEED

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
log = logging.getLogger("generate_events")

ORPHAN_RATE = 0.05          # share of documents given a dangling reference
SESSIONS_PER_CUSTOMER = (1, 4)
EVENTS_PER_SESSION = (2, 14)
BATCH = 5_000

DEVICES = ["mobile-ios", "mobile-android", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.31, 0.38, 0.26, 0.05]
CHANNELS = ["organic_search", "paid_search", "direct", "email", "social", "affiliate"]
CHANNEL_WEIGHTS = [0.34, 0.19, 0.21, 0.12, 0.10, 0.04]
EVENT_TYPES = ["page_view", "search", "product_view", "add_to_cart", "remove_from_cart",
               "begin_checkout", "purchase"]


# ----------------------------------------------------------------------
# reference data pulled from the relational system
# ----------------------------------------------------------------------
def load_reference() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Read the keys the event store must line up with."""
    with psycopg2.connect(**PG) as conn:
        orders = pd.read_sql(
            """
            SELECT o.order_id,
                   o.customer_id,
                   c.customer_unique_id,
                   o.order_purchase_timestamp
            FROM orders o
            JOIN customers c ON c.customer_id = o.customer_id
            """,
            conn,
        )
        products = pd.read_sql(
            "SELECT product_id, product_category_name FROM products "
            "WHERE product_category_name IS NOT NULL",
            conn,
        )
        item_categories = pd.read_sql(
            """
            SELECT DISTINCT oi.order_id, p.product_category_name AS category
            FROM order_items oi
            JOIN products p ON p.product_id = oi.product_id
            WHERE p.product_category_name IS NOT NULL
            """,
            conn,
        )
    log.info("reference: %d orders, %d products", len(orders), len(products))
    return orders, products["product_id"].tolist(), item_categories


def fake_id() -> str:
    """A well-formed but non-existent 32-char identifier, matching Olist's format."""
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
# review documents
# ----------------------------------------------------------------------
def build_reviews(orders: pd.DataFrame, rng: random.Random) -> tuple[list[dict], list[str]]:
    """Review documents, carrying the free text the relational schema omits."""
    raw = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv", encoding="utf-8-sig")
    valid_orders = set(orders["order_id"])
    order_to_customer = dict(zip(orders["order_id"], orders["customer_id"]))

    documents: list[dict] = []
    orphans: list[str] = []

    for row in raw.itertuples(index=False):
        is_orphan = rng.random() < ORPHAN_RATE
        order_id = fake_id() if is_orphan else row.order_id

        if is_orphan:
            orphans.append(order_id)
        elif row.order_id not in valid_orders:
            # already dangling in the source file; keep it, the checks will catch it
            pass

        doc = {
            "review_id": row.review_id,
            "order_id": order_id,
            "customer_id": order_to_customer.get(row.order_id),
            "review_score": int(row.review_score),
            "created_at": pd.to_datetime(row.review_creation_date).to_pydatetime(),
            "language": "pt-BR",
            "source": rng.choices(["web", "app", "email_survey"], [0.5, 0.35, 0.15])[0],
        }

        # optional fields -- present only when the source actually has them.
        # this sparsity is the reason these documents are not relational rows.
        if isinstance(row.review_comment_title, str) and row.review_comment_title.strip():
            doc["title"] = row.review_comment_title.strip()
        if isinstance(row.review_comment_message, str) and row.review_comment_message.strip():
            doc["message"] = row.review_comment_message.strip()
        if pd.notna(row.review_answer_timestamp):
            doc["answered_at"] = pd.to_datetime(row.review_answer_timestamp).to_pydatetime()

        # a nested block that exists on a minority of documents only
        if doc.get("message") and rng.random() < 0.08:
            doc["moderation"] = {
                "flagged": True,
                "reason": rng.choice(["profanity", "pii_suspected", "spam_suspected"]),
                "reviewed_by": f"agent_{rng.randint(1, 40):03d}",
            }

        documents.append(doc)

    log.info("reviews: %d documents (%d orphaned)", len(documents), len(orphans))
    return documents, orphans


# ----------------------------------------------------------------------
# browsing sessions
# ----------------------------------------------------------------------
def build_sessions(
    orders: pd.DataFrame,
    product_ids: list[str],
    item_categories: pd.DataFrame,
    rng: random.Random,
):
    """Synthetic browsing sessions, yielded in batches to bound memory use.

    Sessions are anchored to real purchase timestamps so that the derived
    behavioural features (session recency, browse-to-buy ratio, cart
    abandonment) carry signal rather than noise. A converted session sits
    just before its order; the rest scatter around it.
    """
    category_by_order = (
        item_categories.groupby("order_id")["category"].apply(list).to_dict()
    )
    customer_ids = set(orders["customer_id"])

    batch: list[dict] = []
    orphan_customers: list[str] = []
    total = 0

    for row in orders.itertuples(index=False):
        purchase = row.order_purchase_timestamp
        categories = category_by_order.get(row.order_id, [])
        n_sessions = rng.randint(*SESSIONS_PER_CUSTOMER)

        for index in range(n_sessions):
            converted = index == n_sessions - 1     # last session is the one that bought

            is_orphan = rng.random() < ORPHAN_RATE
            customer_id = fake_id() if is_orphan else row.customer_id
            if is_orphan:
                orphan_customers.append(customer_id)

            # converted session lands minutes before the order; others days before
            if converted:
                start = purchase - timedelta(minutes=rng.randint(4, 90))
            else:
                start = purchase - timedelta(
                    days=rng.randint(1, 75), minutes=rng.randint(0, 1440)
                )

            n_events = rng.randint(*EVENTS_PER_SESSION)
            events = []
            cursor = start
            for _ in range(n_events):
                cursor += timedelta(seconds=rng.randint(3, 420))
                event_type = rng.choices(
                    EVENT_TYPES, [0.30, 0.12, 0.31, 0.13, 0.04, 0.07, 0.03]
                )[0]
                event = {
                    "event_type": event_type,
                    "ts": cursor,
                    "dwell_ms": rng.randint(400, 180_000),
                }
                # event payloads differ by type -- the second reason these are documents
                if event_type in {"product_view", "add_to_cart", "remove_from_cart"}:
                    event["product_id"] = (
                        rng.choice(product_ids) if not categories or rng.random() < 0.4
                        else rng.choice(product_ids)
                    )
                    event["category"] = (
                        rng.choice(categories) if categories else None
                    )
                elif event_type == "search":
                    event["query_terms"] = rng.randint(1, 5)
                    event["results_returned"] = rng.randint(0, 240)
                events.append(event)

            if converted:
                events.append(
                    {"event_type": "purchase", "ts": purchase, "order_id": row.order_id}
                )

            document = {
                "session_id": uuid.uuid4().hex,
                "customer_id": customer_id,
                "customer_unique_id": None if is_orphan else row.customer_unique_id,
                "started_at": start,
                "ended_at": events[-1]["ts"],
                "device": rng.choices(DEVICES, DEVICE_WEIGHTS)[0],
                "channel": rng.choices(CHANNELS, CHANNEL_WEIGHTS)[0],
                "converted": converted,
                "event_count": len(events),
                "events": events,
            }
            if converted:
                document["order_id"] = row.order_id

            batch.append(document)
            total += 1

            if len(batch) >= BATCH:
                yield batch, orphan_customers
                batch, orphan_customers = [], []

    if batch:
        yield batch, orphan_customers
    log.info("sessions: %d documents generated", total)


# ----------------------------------------------------------------------
def main() -> int:
    rng = random.Random(SEED)

    orders, product_ids, item_categories = load_reference()

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    db.drop_collection("product_reviews")
    db.drop_collection("web_sessions")

    reviews, orphan_orders = build_reviews(orders, rng)
    for start in range(0, len(reviews), BATCH):
        db.product_reviews.insert_many(reviews[start : start + BATCH], ordered=False)
    db.product_reviews.create_index([("order_id", ASCENDING)])
    db.product_reviews.create_index([("review_id", ASCENDING)])

    orphan_customers: list[str] = []
    session_total = 0
    for batch, batch_orphans in build_sessions(orders, product_ids, item_categories, rng):
        db.web_sessions.insert_many(batch, ordered=False)
        orphan_customers.extend(batch_orphans)
        session_total += len(batch)
        if session_total % 50_000 == 0:
            log.info("  ... %d sessions inserted", session_total)

    db.web_sessions.create_index([("customer_id", ASCENDING)])
    db.web_sessions.create_index([("started_at", ASCENDING)])

    # ground truth for the integrity tests
    manifest = {
        "seed": SEED,
        "orphan_rate_target": ORPHAN_RATE,
        "review_orphan_order_ids": sorted(orphan_orders),
        "session_orphan_customer_ids": sorted(orphan_customers),
        "counts": {
            "reviews": len(reviews),
            "review_orphans": len(orphan_orders),
            "sessions": session_total,
            "session_orphans": len(orphan_customers),
        },
    }
    out = PROJECT_ROOT / "data" / "quarantine" / "expected_orphans.json"
    out.write_text(json.dumps(manifest, indent=2))

    log.info(
        "done: %d reviews (%.1f%% orphaned), %d sessions (%.1f%% orphaned)",
        len(reviews),
        100 * len(orphan_orders) / len(reviews),
        session_total,
        100 * len(orphan_customers) / session_total,
    )
    log.info("ground-truth manifest -> %s", out)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
