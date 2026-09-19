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
SOURCE = ROOT / "data" / "raw" / "olist_order_items_dataset.csv"
OUT = ROOT / "knime" / "expected_figures.json"


def profile(frame: pd.DataFrame) -> list[dict]:
    """The checks the KNIME recipe implements, computed over the full file."""
    rows = len(frame)

    def check(name: str, column: str, failed: int, severity: str, action: str) -> dict:
        return {
            "check_name": name,
            "column_checked": column,
            "rows_examined": rows,
            "rows_failed": int(failed),
            "fail_rate_pct": round(100 * failed / rows, 4),
            "severity": severity,
            "recommendation": action,
        }

    price = pd.to_numeric(frame["price"], errors="coerce")
    freight = pd.to_numeric(frame["freight_value"], errors="coerce")

    # IQR outlier bounds, matching the Numeric Outliers node's default 1.5x
    q1, q3 = price.quantile(0.25), price.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    checks = [
        check("primary key uniqueness", "order_id + order_item_id",
              rows - len(frame.drop_duplicates(["order_id", "order_item_id"])),
              "FAIL", "composite PRIMARY KEY on (order_id, order_item_id)"),
        check("non-positive price", "price", (price <= 0).sum(),
              "FAIL", "CHECK (price >= 0) in the schema"),
        check("negative freight", "freight_value", (freight < 0).sum(),
              "FAIL", "CHECK (freight_value >= 0) in the schema"),
        check("missing price", "price", price.isna().sum(),
              "FAIL", "NOT NULL in the schema"),
        check("missing freight", "freight_value", freight.isna().sum(),
              "WARN", "fillna(0) in build_order_lines, row retained"),
        check("price IQR outliers", "price",
              ((price < low) | (price > high)).sum(),
              "WARN", "WAPE not MAPE as the forecast metric, so extremes cannot dominate"),
        check("missing shipping limit", "shipping_limit_date",
              frame["shipping_limit_date"].isna().sum(),
              "WARN", "nullable TIMESTAMP in the schema"),
    ]
    return checks


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found -- download the Olist dataset first")

    frame = pd.read_csv(SOURCE)
    checks = profile(frame)

    price = pd.to_numeric(frame["price"], errors="coerce")
    total_failed = sum(c["rows_failed"] for c in checks if c["severity"] == "FAIL")
    total_warned = sum(c["rows_failed"] for c in checks if c["severity"] == "WARN")
    # the same weighting the recipe's Math Formula node applies
    score = 100 - (total_failed * 5.0 + total_warned * 1.0) / len(frame) * 100

    report = {
        "note": (
            "Reference figures for verifying the KNIME workflow output. Not part "
            "of the Kedro pipeline. A 10% KNIME sample should approximate the "
            "rates below, not match the counts."
        ),
        "source_file": SOURCE.name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "distinct_orders": int(frame["order_id"].nunique()),
        "distinct_products": int(frame["product_id"].nunique()),
        "distinct_sellers": int(frame["seller_id"].nunique()),
        "price_summary": {
            "min": round(float(price.min()), 2),
            "median": round(float(price.median()), 2),
            "mean": round(float(price.mean()), 2),
            "max": round(float(price.max()), 2),
        },
        "checks": checks,
        "quality_score": round(score, 2),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"{len(frame):,} rows | quality score {score:.2f}/100\n")
    print(f"{'check':28s} {'failed':>8s} {'rate %':>9s}  severity")
    for item in checks:
        print(f"{item['check_name']:28s} {item['rows_failed']:>8,} "
              f"{item['fail_rate_pct']:>9.4f}  {item['severity']}")
    print(f"\nwritten -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
