"""Profile the raw source files for the data-quality layer.

This is the data behind the Source Data Quality explorer -- the project's
low-code layer, aimed at a data steward rather than an engineer.

It is deliberately NOT part of the Kedro pipeline. The pipeline validates what
it loads; this profiles what arrived, before anyone has decided what to do
about it. Different question, different reader, different moment. Wiring it
into the DAG would collapse it into a duplicate of work the pipeline already
does.

Two kinds of check come out of here:

* **fixed** -- nulls, duplicates, key uniqueness. There is nothing to tune:
  a null is a null.
* **tunable** -- outlier rules, whose answer depends on where you draw the
  line. Rather than hard-coding a threshold, the full value distribution is
  exported so the explorer can recompute the count at any threshold the
  steward picks. That is the part that makes the layer genuinely low-code:
  changing the rule needs no edit to this file.

    python scripts/profile_sources.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("profile")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "quality" / "source_profile.json"

# Every finding names the pipeline decision it justified. A profile that
# changes nothing downstream is an observation, not a deliverable.
REMEDIATION = {
    "primary key uniqueness": ("composite PRIMARY KEY on (order_id, order_item_id)",
                               "scripts/sql/01_schema.sql"),
    "non-positive price": ("CHECK (price >= 0) in the schema",
                           "scripts/sql/01_schema.sql"),
    "negative freight": ("CHECK (freight_value >= 0) in the schema",
                         "scripts/sql/01_schema.sql"),
    "missing price": ("NOT NULL in the schema", "scripts/sql/01_schema.sql"),
    "missing freight": ("fillna(0) in build_order_lines, row retained",
                        "pipelines/features/nodes.py"),
    "missing shipping limit": ("nullable TIMESTAMP in the schema",
                               "scripts/sql/01_schema.sql"),
    "price outliers": ("WAPE rather than MAPE as the forecast metric, so a handful of "
                       "extremes cannot dominate the error",
                       "pipelines/forecasting/nodes.py"),
    "freight outliers": ("freight enters as a line cost, never as a model target",
                         "pipelines/features/nodes.py"),
    "missing product category": ("COALESCE(..., 'unknown') rather than dropping the rows",
                                 "conf/base/catalog.yml"),
    "untranslated categories": ("no FK declared on that column -- an FK there would "
                                "reject the load; the gap is handled in the silver layer",
                                "scripts/sql/01_schema.sql"),
    "duplicate review ids": ("surrogate review_key primary key, so duplicates are "
                             "preserved rather than rejected",
                             "scripts/sql/01_schema.sql"),
    "reviews with no body": ("review text lives in MongoDB, not in a relational column "
                             "of mostly nulls", "scripts/generate_events.py"),
    "reviews with no title": ("same -- optional fields belong in a document store",
                              "scripts/generate_events.py"),
    "orders never delivered": ("lead times computed only from delivered orders; the rest "
                               "are excluded explicitly rather than silently producing "
                               "null intervals", "pipelines/scale/nodes.py"),
}


def _fixed(name: str, column: str, rows: int, failed: int, severity: str) -> dict:
    action, where = REMEDIATION.get(name, ("", ""))
    return {
        "kind": "fixed",
        "check_name": name,
        "column_checked": column,
        "rows_examined": int(rows),
        "rows_failed": int(failed),
        "severity": severity,
        "recommendation": action,
        "handled_in": where,
    }


def _tunable(name: str, column: str, rows: int, severity: str) -> dict:
    """A check whose count the explorer computes from the distribution."""
    action, where = REMEDIATION.get(name, ("", ""))
    return {
        "kind": "tunable",
        "check_name": name,
        "column_checked": column,
        "rows_examined": int(rows),
        "severity": severity,
        "recommendation": action,
        "handled_in": where,
    }


def _distribution(series: pd.Series) -> dict:
    """Value/count pairs plus quartiles, enough to recompute any fence."""
    rounded = series.dropna().round(2)
    counts = rounded.value_counts().sort_index()
    q1, q3 = float(rounded.quantile(0.25)), float(rounded.quantile(0.75))
    return {
        "values": [round(float(v), 2) for v in counts.index],
        "counts": [int(c) for c in counts.values],
        # Quartiles keep full precision: rounding them to currency precision
        # shifts the computed fence just enough to change the row count by a
        # handful, and a quality tool that disagrees with the source by five
        # rows is a quality tool nobody trusts.
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": round(float(rounded.min()), 2),
        "max": round(float(rounded.max()), 2),
        "median": round(float(rounded.median()), 2),
        "n": int(len(rounded)),
    }


def profile_order_items() -> tuple[dict, dict]:
    frame = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    rows = len(frame)
    price = pd.to_numeric(frame["price"], errors="coerce")
    freight = pd.to_numeric(frame["freight_value"], errors="coerce")

    profile = {
        "file": "olist_order_items_dataset.csv",
        "label": "Order items",
        "rows": rows,
        "checks": [
            _fixed("primary key uniqueness", "order_id + order_item_id", rows,
                   rows - len(frame.drop_duplicates(["order_id", "order_item_id"])), "FAIL"),
            _fixed("non-positive price", "price", rows, (price <= 0).sum(), "FAIL"),
            _fixed("negative freight", "freight_value", rows, (freight < 0).sum(), "FAIL"),
            _fixed("missing price", "price", rows, price.isna().sum(), "FAIL"),
            _fixed("missing freight", "freight_value", rows, freight.isna().sum(), "WARN"),
            _fixed("missing shipping limit", "shipping_limit_date", rows,
                   frame["shipping_limit_date"].isna().sum(), "WARN"),
            _tunable("price outliers", "price", rows, "WARN"),
            _tunable("freight outliers", "freight_value", rows, "WARN"),
        ],
        "facts": {
            "distinct orders": f"{frame['order_id'].nunique():,}",
            "distinct products": f"{frame['product_id'].nunique():,}",
            "distinct sellers": f"{frame['seller_id'].nunique():,}",
            "median price": f"R${price.median():.2f}",
        },
    }
    distributions = {"price": _distribution(price),
                     "freight_value": _distribution(freight)}
    return profile, distributions


def profile_products() -> dict:
    frame = pd.read_csv(RAW / "olist_products_dataset.csv")
    translation = pd.read_csv(RAW / "product_category_name_translation.csv",
                              encoding="utf-8-sig")
    rows = len(frame)
    categories = set(frame["product_category_name"].dropna())
    untranslated = sorted(categories - set(translation["product_category_name"]))
    return {
        "file": "olist_products_dataset.csv",
        "label": "Products",
        "rows": rows,
        "checks": [
            _fixed("missing product category", "product_category_name", rows,
                   frame["product_category_name"].isna().sum(), "WARN"),
            _fixed("untranslated categories", "product_category_name", len(categories),
                   len(untranslated), "WARN"),
        ],
        "facts": {
            "distinct categories": str(len(categories)),
            "untranslated": ", ".join(untranslated) or "none",
        },
    }


def profile_reviews() -> dict:
    frame = pd.read_csv(RAW / "olist_order_reviews_dataset.csv")
    rows = len(frame)
    return {
        "file": "olist_order_reviews_dataset.csv",
        "label": "Reviews",
        "rows": rows,
        "checks": [
            _fixed("duplicate review ids", "review_id", rows,
                   rows - frame["review_id"].nunique(), "FAIL"),
            # INFO, not WARN: this sparsity is expected and is the whole reason
            # review text sits in a document store. Scored as a defect it drags
            # the scorecard to ~53/100 and sends a steward chasing a
            # non-problem, which is how quality scorecards lose their audience.
            _fixed("reviews with no body", "review_comment_message", rows,
                   frame["review_comment_message"].isna().sum(), "INFO"),
            _fixed("reviews with no title", "review_comment_title", rows,
                   frame["review_comment_title"].isna().sum(), "INFO"),
        ],
        "facts": {"distinct review ids": f"{frame['review_id'].nunique():,}"},
    }


def profile_orders() -> dict:
    frame = pd.read_csv(RAW / "olist_orders_dataset.csv")
    rows = len(frame)
    return {
        "file": "olist_orders_dataset.csv",
        "label": "Orders",
        "rows": rows,
        "checks": [
            _fixed("orders never delivered", "order_delivered_customer_date", rows,
                   frame["order_delivered_customer_date"].isna().sum(), "WARN"),
        ],
        "facts": {"order statuses": str(frame["order_status"].nunique())},
    }


def main() -> int:
    required = ["olist_order_items_dataset.csv", "olist_products_dataset.csv",
                "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
                "product_category_name_translation.csv"]
    missing = [name for name in required if not (RAW / name).exists()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")

    order_items, distributions = profile_order_items()
    files = [order_items, profile_products(), profile_reviews(), profile_orders()]

    report = {
        "generated": pd.Timestamp.now().strftime("%d %b %Y"),
        "files": files,
        "distributions": distributions,
        # starting positions for the explorer's controls; the steward moves them
        "defaults": {
            "iqr_multiplier": 1.5,
            "fail_weight": 5.0,
            "warn_weight": 1.0,
            "score_info": False,
            "pass_mark": 90,
        },
        "scoring": ("Score = 100 minus the weighted share of rows that breached a check. "
                    "INFO rows are reported but excluded by default: they record expected "
                    "structure, not defects."),
        "totals": {
            "rows_examined": sum(f["rows"] for f in files),
            "checks": sum(len(f["checks"]) for f in files),
            "files": len(files),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, separators=(",", ":")))

    fixed = [c for f in files for c in f["checks"] if c["kind"] == "fixed"]
    log.info("%d files, %d checks (%d fixed, %d tunable), %s rows",
             len(files), report["totals"]["checks"], len(fixed),
             report["totals"]["checks"] - len(fixed),
             f"{report['totals']['rows_examined']:,}")
    for f in files:
        breaches = [c for c in f["checks"]
                    if c["kind"] == "fixed" and c["rows_failed"] > 0]
        log.info("  %-12s %6s rows  %d breaching checks", f["label"],
                 f"{f['rows']:,}", len(breaches))
    log.info("-> %s (%.0f KB)", OUT.relative_to(ROOT), OUT.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
