"""Reference data-quality figures for verifying the KNIME workflow.

This is a VERIFICATION FIXTURE, not part of the pipeline and not a substitute
for the KNIME workflow. It is deliberately excluded from the Kedro DAG.

The point of Phase 2 is that a business analyst profiles the raw file in a
no-code tool, independently of the engineered pipeline. That only works as a
check if there is something to check the answer against -- otherwise a
mis-configured Rule Engine produces a confident scorecard full of wrong
numbers and nobody notices.

So this script computes the same population statistics over the whole file.
Build the workflow in KNIME, then confirm its scorecard lands on these figures.

Note on sampling: the KNIME recipe profiles a 10% sample. KNIME's random
number generator is not Python's, so a sampled scorecard will not match these
numbers exactly -- it should approximate them. Rates should agree to within a
few tenths of a percent; counts will be roughly one tenth.

    python scripts/knime_reference_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "knime" / "expected_figures.json"

# Each finding names the pipeline decision it justified. A profile that changes
# nothing downstream is an observation, not a deliverable.
REMEDIATION = {
    "primary key uniqueness": ("composite PRIMARY KEY on (order_id, order_item_id)",
                               "scripts/sql/01_schema.sql"),
    "non-positive price": ("CHECK (price >= 0) in the schema", "scripts/sql/01_schema.sql"),
    "negative freight": ("CHECK (freight_value >= 0) in the schema", "scripts/sql/01_schema.sql"),
    "missing price": ("NOT NULL in the schema", "scripts/sql/01_schema.sql"),
    "missing freight": ("fillna(0) in build_order_lines, row retained",
                        "pipelines/features/nodes.py"),
    "price IQR outliers": ("WAPE rather than MAPE as the forecast metric, so a handful "
                           "of extremes cannot dominate the error",
                           "pipelines/forecasting/nodes.py"),
    "missing shipping limit": ("nullable TIMESTAMP in the schema", "scripts/sql/01_schema.sql"),
    "missing product category": ("COALESCE(..., 'unknown') rather than dropping the rows",
                                 "conf/base/catalog.yml"),
    "untranslated categories": ("no FK declared on that column -- an FK there would reject "
                                "the load; the gap is handled in the silver layer",
                                "scripts/sql/01_schema.sql"),
    "duplicate review ids": ("surrogate review_key primary key, so duplicates are preserved "
                             "rather than rejected", "scripts/sql/01_schema.sql"),
    "reviews with no body": ("review text lives in MongoDB, not in a relational column of "
                             "mostly nulls", "scripts/generate_events.py"),
    "reviews with no title": ("same -- optional fields belong in a document store",
                              "scripts/generate_events.py"),
    "orders never delivered": ("lead times computed only from delivered orders; the rest are "
                               "excluded explicitly rather than silently producing null "
                               "intervals", "pipelines/scale/nodes.py"),
}


def _check(name, column, rows, failed, severity):
    action, where = REMEDIATION.get(name, ("", ""))
    return {
        "check_name": name,
        "column_checked": column,
        "rows_examined": int(rows),
        "rows_failed": int(failed),
        "fail_rate_pct": round(100 * failed / rows, 4) if rows else 0.0,
        "severity": severity,
        "recommendation": action,
        "handled_in": where,
    }


def profile_order_items() -> dict:
    frame = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    rows = len(frame)
    price = pd.to_numeric(frame["price"], errors="coerce")
    freight = pd.to_numeric(frame["freight_value"], errors="coerce")

    q1, q3 = price.quantile(0.25), price.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    checks = [
        _check("primary key uniqueness", "order_id + order_item_id", rows,
               rows - len(frame.drop_duplicates(["order_id", "order_item_id"])), "FAIL"),
        _check("non-positive price", "price", rows, (price <= 0).sum(), "FAIL"),
        _check("negative freight", "freight_value", rows, (freight < 0).sum(), "FAIL"),
        _check("missing price", "price", rows, price.isna().sum(), "FAIL"),
        _check("missing freight", "freight_value", rows, freight.isna().sum(), "WARN"),
        _check("price IQR outliers", "price", rows,
               ((price < low) | (price > high)).sum(), "WARN"),
        _check("missing shipping limit", "shipping_limit_date", rows,
               frame["shipping_limit_date"].isna().sum(), "WARN"),
    ]
    return {
        "file": "olist_order_items_dataset.csv", "label": "Order items",
        "rows": rows, "checks": checks,
        "outlier_fence": {"q1": round(float(q1), 2), "q3": round(float(q3), 2),
                          "upper": round(float(high), 2)},
        "facts": {
            "distinct orders": int(frame["order_id"].nunique()),
            "distinct products": int(frame["product_id"].nunique()),
            "distinct sellers": int(frame["seller_id"].nunique()),
            "median price": round(float(price.median()), 2),
        },
    }


def profile_products() -> dict:
    frame = pd.read_csv(RAW / "olist_products_dataset.csv")
    translation = pd.read_csv(RAW / "product_category_name_translation.csv",
                              encoding="utf-8-sig")
    rows = len(frame)
    categories = set(frame["product_category_name"].dropna())
    untranslated = sorted(categories - set(translation["product_category_name"]))
    return {
        "file": "olist_products_dataset.csv", "label": "Products",
        "rows": rows,
        "checks": [
            _check("missing product category", "product_category_name", rows,
                   frame["product_category_name"].isna().sum(), "WARN"),
            _check("untranslated categories", "product_category_name", len(categories),
                   len(untranslated), "WARN"),
        ],
        "facts": {"distinct categories": len(categories),
                  "untranslated": ", ".join(untranslated) or "none"},
    }


def profile_reviews() -> dict:
    frame = pd.read_csv(RAW / "olist_order_reviews_dataset.csv")
    rows = len(frame)
    return {
        "file": "olist_order_reviews_dataset.csv", "label": "Reviews",
        "rows": rows,
        "checks": [
            _check("duplicate review ids", "review_id", rows,
                   rows - frame["review_id"].nunique(), "FAIL"),
            # INFO, not WARN: this sparsity is expected and is the entire reason
            # review text sits in a document store. Scoring it as a defect would
            # drag the scorecard to ~53/100 and send an analyst chasing a
            # non-problem, which is how quality scorecards lose their audience.
            _check("reviews with no body", "review_comment_message", rows,
                   frame["review_comment_message"].isna().sum(), "INFO"),
            _check("reviews with no title", "review_comment_title", rows,
                   frame["review_comment_title"].isna().sum(), "INFO"),
        ],
        "facts": {"distinct review ids": int(frame["review_id"].nunique())},
    }


def profile_orders() -> dict:
    frame = pd.read_csv(RAW / "olist_orders_dataset.csv")
    rows = len(frame)
    return {
        "file": "olist_orders_dataset.csv", "label": "Orders",
        "rows": rows,
        "checks": [
            _check("orders never delivered", "order_delivered_customer_date", rows,
                   frame["order_delivered_customer_date"].isna().sum(), "WARN"),
        ],
        "facts": {"statuses": int(frame["order_status"].nunique())},
    }


def main() -> int:
    missing = [f for f in ("olist_order_items_dataset.csv", "olist_products_dataset.csv",
                           "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv")
               if not (RAW / f).exists()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")

    files = [profile_order_items(), profile_products(),
             profile_reviews(), profile_orders()]

    # the weighting the KNIME recipe's scoring step applies
    total_rows = sum(f["rows"] for f in files)
    failed = sum(c["rows_failed"] for f in files for c in f["checks"]
                 if c["severity"] == "FAIL")
    warned = sum(c["rows_failed"] for f in files for c in f["checks"]
                 if c["severity"] == "WARN")
    noted = sum(c["rows_failed"] for f in files for c in f["checks"]
                if c["severity"] == "INFO")
    # INFO rows are reported but not scored -- see the reviews profile
    score = 100 - (failed * 5.0 + warned * 1.0) / total_rows * 100

    report = {
        "note": ("Reference figures for verifying the KNIME workflow output. Not part of "
                 "the Kedro pipeline. A 10% KNIME sample should approximate the rates "
                 "below, not match the counts."),
        "generated": pd.Timestamp.now().strftime("%d %b %Y"),
        "files": files,
        "totals": {"rows_examined": total_rows, "rows_failed": failed,
                   "rows_warned": warned, "rows_noted": noted,
                   "checks": sum(len(f["checks"]) for f in files)},
        "scoring": ("5 points per FAIL row and 1 per WARN row, as a share of all rows "
                    "examined. INFO rows are reported but not scored: they record "
                    "expected structure, not defects."),
        "quality_score": round(score, 2),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"{total_rows:,} rows across {len(files)} files | score {score:.2f}/100\n")
    for f in files:
        print(f"{f['label']} ({f['rows']:,} rows)")
        for c in f["checks"]:
            print(f"  {c['check_name']:28s} {c['rows_failed']:>8,} "
                  f"{c['fail_rate_pct']:>8.3f}%  {c['severity']}")
    print(f"\nwritten -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
