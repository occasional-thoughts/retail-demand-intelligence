"""Post-run verification: did the integrity layer catch what was injected?

Every other test in this suite runs on synthetic fixtures and needs no
database. This one is different: it reads the artefacts of an actual pipeline
run and checks them against the ground-truth manifest written by
`scripts/generate_events.py`.

It exists because the unit tests can only prove the check behaves correctly on
data shaped the way the test author imagined. This proves it behaved correctly
on the data that actually went through -- which is the claim the README makes.

Skipped automatically when the pipeline has not been run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "quarantine" / "expected_orphans.json"
QUARANTINED_REVIEWS = ROOT / "data" / "quarantine" / "orphaned_reviews.parquet"
QUARANTINED_SESSIONS = ROOT / "data" / "quarantine" / "orphaned_sessions.parquet"
REPORT = ROOT / "data" / "quarantine" / "integrity_report.json"

# Skip unless a *complete* run is on disk. A partial run would make these
# tests fail for the wrong reason -- reporting a missing file as a data-quality
# defect is exactly the kind of misleading signal this suite exists to avoid.
REQUIRED = (MANIFEST, QUARANTINED_REVIEWS, QUARANTINED_SESSIONS, REPORT)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="no complete pipeline run to verify -- run `make sources && make pipeline`",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_every_injected_review_orphan_was_caught(manifest):
    """Recall must be exactly 1.0 -- a missed orphan is a corrupted join."""
    injected = set(manifest["review_orphan_order_ids"])
    caught = set(pd.read_parquet(QUARANTINED_REVIEWS)["order_id"])
    missed = injected - caught
    assert not missed, f"{len(missed)} injected orphans were not quarantined"


def test_no_valid_review_was_quarantined(manifest):
    """Precision matters too -- quarantining good rows silently loses data."""
    injected = set(manifest["review_orphan_order_ids"])
    caught = set(pd.read_parquet(QUARANTINED_REVIEWS)["order_id"])
    false_positives = caught - injected
    # the source file itself has no dangling order_ids, so anything beyond the
    # injected set is the check over-reaching
    assert not false_positives, f"{len(false_positives)} valid orders quarantined"


def test_every_injected_session_orphan_was_caught(manifest):
    injected = set(manifest["session_orphan_customer_ids"])
    caught = set(pd.read_parquet(QUARANTINED_SESSIONS)["customer_id"])
    assert not injected - caught


def test_each_check_stayed_within_its_own_thresholds(manifest):
    """Rates are checked against the per-check band, not one global number.

    The review check runs at document grain and sees the injected rate
    directly. The session check runs after a customer-level aggregation, where
    valid sessions collapse together and orphans do not, so the same breakage
    presents higher. Asserting one shared rate across both would be asserting
    something untrue about the second.
    """
    report = json.loads(REPORT.read_text())
    for check in report["checks"]:
        band = check["thresholds"]
        assert band["min_orphan_rate"] <= check["orphan_rate"] <= band["max_orphan_rate"], (
            f"{check['check']}: observed {check['orphan_rate']:.4f}, outside "
            f"[{band['min_orphan_rate']}, {band['max_orphan_rate']}]"
        )


def test_document_grain_orphan_rate_matches_injection(manifest):
    """At document grain the observed rate must equal what was injected."""
    report = json.loads(REPORT.read_text())
    review_check = next(c for c in report["checks"] if "reviews" in c["check"])
    assert abs(review_check["orphan_rate"] - manifest["orphan_rate_target"]) < 0.01


def test_nothing_was_lost(manifest):
    """Rows examined must equal rows passed plus rows quarantined, exactly."""
    report = json.loads(REPORT.read_text())
    for check in report["checks"]:
        assert (
            check["records_passed"] + check["records_quarantined"]
            == check["records_examined"]
        )


def test_quarantined_rows_are_investigable():
    frame = pd.read_parquet(QUARANTINED_REVIEWS)
    for column in ("quarantine_reason", "quarantined_at", "source_dataset"):
        assert column in frame.columns
        assert frame[column].notna().all()
