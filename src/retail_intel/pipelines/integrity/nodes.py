"""Referential integrity between the relational system and the event store.

No foreign key can span two databases. When the order-management system and
the web platform are maintained independently, references drift: the event
store accumulates documents pointing at orders and customers that the
relational system has never heard of.

The requirement here is specific, and it is the opposite of what a naive
pipeline does. A naive pipeline inner-joins the two sources, silently loses
the unmatched rows, and reports a clean run. This one:

  * separates matched from unmatched records explicitly,
  * writes the unmatched records to a quarantine table so a human can
    investigate them,
  * fails the run outright when the breakage rate crosses a threshold,
  * and fails the run when it finds *no* breakage at all, because a check
    that never fires is indistinguishable from a check that is broken.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


class ReferentialIntegrityError(RuntimeError):
    """Raised when cross-system breakage exceeds the tolerated threshold."""


def _split_on_key(
    frame: pd.DataFrame, key: str, valid_keys: set[str], label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a frame into rows whose key resolves and rows whose key does not."""
    if key not in frame.columns:
        raise KeyError(f"{label}: expected join key '{key}' is absent from the extract")

    resolves = frame[key].isin(valid_keys)
    matched = frame.loc[resolves].copy()
    orphaned = frame.loc[~resolves].copy()

    orphaned["quarantine_reason"] = f"{key} not present in relational source of truth"
    orphaned["quarantined_at"] = datetime.now(timezone.utc).isoformat()
    orphaned["source_dataset"] = label

    rate = len(orphaned) / len(frame) if len(frame) else 0.0
    log.info(
        "%s: %d rows, %d matched, %d orphaned (%.2f%%)",
        label,
        len(frame),
        len(matched),
        len(orphaned),
        100 * rate,
    )
    return matched, orphaned


def check_review_integrity(
    bronze_reviews: pd.DataFrame,
    pg_valid_order_ids: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every review document must name an order that exists in Postgres."""
    valid = set(pg_valid_order_ids["order_id"])
    matched, orphaned = _split_on_key(
        bronze_reviews, "order_id", valid, "reviews"
    )
    _enforce(len(orphaned), len(bronze_reviews), "reviews", params)
    return matched, orphaned


def check_session_integrity(
    bronze_sessions: pd.DataFrame,
    pg_valid_customer_ids: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every session document must name a customer that exists in Postgres."""
    valid = set(pg_valid_customer_ids["customer_id"])
    matched, orphaned = _split_on_key(
        bronze_sessions, "customer_id", valid, "sessions"
    )
    _enforce(len(orphaned), len(bronze_sessions), "sessions", params)
    return matched, orphaned


def _thresholds(label: str, params: dict) -> dict:
    """Resolve the thresholds for one check.

    Per-check rather than global: the two checks measure at different grains
    (see conf/base/parameters.yml), so one shared number would be wrong for at
    least one of them.
    """
    try:
        return params["checks"][label]
    except KeyError as error:
        raise KeyError(
            f"no integrity thresholds configured for check '{label}'. Add a "
            f"`{label}:` block under `integrity.checks` in parameters.yml -- "
            f"refusing to fall back to a default, because a silently-defaulted "
            f"threshold is how a check stops checking."
        ) from error


def _enforce(n_orphans: int, n_total: int, label: str, params: dict) -> None:
    """Apply the two failure conditions.

    Upper bound: breakage above `max_orphan_rate` means the two systems have
    diverged badly enough that downstream numbers would be wrong, so the run
    stops rather than producing a quietly incorrect forecast.

    Lower bound: this project deliberately injects orphans, so finding none
    means the check is not actually checking anything. Silent no-op tests are
    worse than no tests, because they buy false confidence.
    """
    rate = n_orphans / n_total if n_total else 0.0
    thresholds = _thresholds(label, params)
    ceiling = thresholds["max_orphan_rate"]
    floor = thresholds["min_orphan_rate"]

    if rate > ceiling:
        raise ReferentialIntegrityError(
            f"{label}: {n_orphans}/{n_total} records ({rate:.2%}) reference keys that "
            f"do not exist in the relational source of truth. This exceeds the "
            f"tolerated ceiling of {ceiling:.2%}. The run has been stopped rather "
            f"than producing figures derived from a partially-joined dataset. "
            f"Quarantined records are in data/quarantine/ for investigation."
        )

    if rate < floor:
        raise ReferentialIntegrityError(
            f"{label}: found only {rate:.2%} orphaned records, below the expected "
            f"floor of {floor:.2%}. This fixture injects a known share of orphans, "
            f"so a near-zero result means the integrity check itself is not working "
            f"(wrong join key, wrong extract, or an upstream filter that already "
            f"removed them). Failing loudly rather than reporting a clean run."
        )

    log.warning(
        "%s: %d records (%.2f%%) quarantined for unresolvable references "
        "-- within tolerance [%.2f%%, %.2f%%] but NOT ignored",
        label,
        n_orphans,
        100 * rate,
        100 * floor,
        100 * ceiling,
    )


def combine_quarantine(
    orphaned_reviews: pd.DataFrame,
    orphaned_sessions: pd.DataFrame,
    bronze_reviews: pd.DataFrame,
    bronze_sessions: pd.DataFrame,
    params: dict,
) -> dict:
    """Assemble the run's data-quality report.

    This artefact is what a data steward reads after a run. It is persisted as
    JSON so it can be diffed between runs and alerted on -- a quality signal
    that only lives in a log line is a quality signal nobody acts on.
    """
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": params["checks"],
        "checks": [
            {
                "check": "reviews.order_id -> orders.order_id",
                "records_examined": len(bronze_reviews),
                "records_passed": len(bronze_reviews) - len(orphaned_reviews),
                "records_quarantined": len(orphaned_reviews),
                "orphan_rate": round(len(orphaned_reviews) / max(len(bronze_reviews), 1), 5),
                "grain": "one row per review document",
                "thresholds": params["checks"]["reviews"],
                "status": "PASS_WITH_QUARANTINE" if len(orphaned_reviews) else "PASS",
            },
            {
                "check": "sessions.customer_id -> customers.customer_id",
                "records_examined": len(bronze_sessions),
                "records_passed": len(bronze_sessions) - len(orphaned_sessions),
                "records_quarantined": len(orphaned_sessions),
                "orphan_rate": round(len(orphaned_sessions) / max(len(bronze_sessions), 1), 5),
                "grain": "one row per customer, aggregated from session documents",
                "thresholds": params["checks"]["sessions"],
                "status": "PASS_WITH_QUARANTINE" if len(orphaned_sessions) else "PASS",
            },
        ],
        "totals": {
            "records_examined": len(bronze_reviews) + len(bronze_sessions),
            "records_quarantined": len(orphaned_reviews) + len(orphaned_sessions),
        },
    }
    log.info(
        "integrity report: %d of %d records quarantined",
        report["totals"]["records_quarantined"],
        report["totals"]["records_examined"],
    )
    return report
