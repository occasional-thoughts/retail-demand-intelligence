"""Referential integrity checks.

The behaviour under test is not "does it find orphans" but "does it refuse to
continue when it should" -- both when there is too much breakage and when
there is suspiciously little.
"""
import pandas as pd
import pytest

from retail_intel.pipelines.integrity.nodes import (
    ReferentialIntegrityError,
    check_review_integrity,
    combine_quarantine,
)


def test_orphans_are_separated_not_dropped(
    reviews_with_orphans, valid_order_ids, integrity_params
):
    clean, quarantined = check_review_integrity(
        reviews_with_orphans, valid_order_ids, integrity_params
    )
    # nothing may vanish: every input row ends up in exactly one output
    assert len(clean) + len(quarantined) == len(reviews_with_orphans)
    assert len(quarantined) == 5
    assert set(clean["order_id"]).issubset(set(valid_order_ids["order_id"]))


def test_quarantined_rows_carry_an_explanation(
    reviews_with_orphans, valid_order_ids, integrity_params
):
    _, quarantined = check_review_integrity(
        reviews_with_orphans, valid_order_ids, integrity_params
    )
    # a quarantine table nobody can interpret is a quarantine table nobody uses
    for column in ("quarantine_reason", "quarantined_at", "source_dataset"):
        assert column in quarantined.columns
    assert quarantined["quarantine_reason"].str.contains("order_id").all()


def test_run_fails_when_breakage_exceeds_ceiling(valid_order_ids, integrity_params):
    mostly_broken = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(100)],
            "order_id": [f"ghost_{i}" for i in range(50)]
            + [f"order_{i:03d}" for i in range(50)],
            "review_score": 5,
        }
    )
    with pytest.raises(ReferentialIntegrityError, match="exceeds the tolerated ceiling"):
        check_review_integrity(mostly_broken, valid_order_ids, integrity_params)


def test_run_fails_when_no_breakage_is_found(valid_order_ids, integrity_params):
    """A check that never fires is indistinguishable from a broken check."""
    perfectly_clean = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(100)],
            "order_id": [f"order_{i:03d}" for i in range(100)],
            "review_score": 5,
        }
    )
    with pytest.raises(ReferentialIntegrityError, match="below the expected floor"):
        check_review_integrity(perfectly_clean, valid_order_ids, integrity_params)


def test_report_totals_reconcile(reviews_with_orphans, valid_order_ids, integrity_params):
    clean, quarantined = check_review_integrity(
        reviews_with_orphans, valid_order_ids, integrity_params
    )
    sessions = pd.DataFrame({"customer_id": ["c1"], "sessions": [1]})
    report = combine_quarantine(
        quarantined, sessions.head(0), reviews_with_orphans, sessions, integrity_params
    )
    check = report["checks"][0]
    assert check["records_passed"] + check["records_quarantined"] == check["records_examined"]
    assert check["status"] == "PASS_WITH_QUARANTINE"
