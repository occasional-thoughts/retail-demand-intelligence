"""Tests for the data-quality layer.

The explorer recomputes outlier counts in the browser from an exported
distribution, rather than reading a precomputed number. That is what makes the
rules tunable without a deploy -- and it means there are now two
implementations of the same arithmetic, in two languages, that must agree.

These tests re-implement the page's algorithm in Python and check it against a
direct pandas computation over the raw file. A five-row disagreement caused by
rounding the exported quartiles was caught exactly this way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "quality" / "source_profile.json"
ORDER_ITEMS = ROOT / "data" / "raw" / "olist_order_items_dataset.csv"

pytestmark = pytest.mark.skipif(
    not (PROFILE.exists() and ORDER_ITEMS.exists()),
    reason="no source profile -- run `make quality` first",
)


@pytest.fixture(scope="module")
def profile() -> dict:
    return json.loads(PROFILE.read_text())


@pytest.fixture(scope="module")
def order_items() -> pd.DataFrame:
    return pd.read_csv(ORDER_ITEMS)


def count_outside(distribution: dict, multiplier: float) -> int:
    """The page's algorithm: count rows beyond the fence, from value/count pairs.

    Mirrors countOutside() in dashboards/quality.html. If the two ever diverge,
    the page is lying to a data steward about their own data.
    """
    values, counts = distribution["values"], distribution["counts"]
    low = distribution["q1"] - multiplier * distribution["iqr"]
    high = distribution["q3"] + multiplier * distribution["iqr"]

    below = sum(c for v, c in zip(values, counts) if v < low)
    above = sum(c for v, c in zip(values, counts) if v > high)
    return below + above


@pytest.mark.parametrize("multiplier", [1.0, 1.5, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("column", ["price", "freight_value"])
def test_tunable_counts_match_the_raw_file(profile, order_items, column, multiplier):
    """The browser's count must equal what pandas says, at every fence."""
    series = pd.to_numeric(order_items[column], errors="coerce").round(2)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    expected = int(((series < q1 - multiplier * iqr) |
                    (series > q3 + multiplier * iqr)).sum())

    actual = count_outside(profile["distributions"][column], multiplier)
    assert actual == expected, (
        f"{column} at IQR x{multiplier}: page would show {actual}, "
        f"source says {expected}"
    )


def test_exported_quartiles_match_the_source_exactly(profile, order_items):
    """Rounding the quartiles shifts the fence and changes the row count.

    Regression test: exporting them at currency precision moved the IQR x3.0
    count by five rows, so the page disagreed with the file it was describing.
    """
    for column, distribution in profile["distributions"].items():
        series = pd.to_numeric(order_items[column], errors="coerce").round(2)
        assert distribution["q1"] == pytest.approx(series.quantile(0.25), abs=1e-9)
        assert distribution["q3"] == pytest.approx(series.quantile(0.75), abs=1e-9)
        assert distribution["iqr"] == pytest.approx(
            series.quantile(0.75) - series.quantile(0.25), abs=1e-9)


def test_distribution_totals_match_the_row_count(profile, order_items):
    for column, distribution in profile["distributions"].items():
        series = pd.to_numeric(order_items[column], errors="coerce")
        assert sum(distribution["counts"]) == int(series.notna().sum())
        assert distribution["n"] == int(series.notna().sum())


def test_values_are_sorted(profile):
    """The page walks the arrays assuming ascending order."""
    for column, distribution in profile["distributions"].items():
        values = distribution["values"]
        assert values == sorted(values), column


def test_every_breaching_check_names_its_remediation(profile):
    """A finding with nowhere to point is an observation, not a deliverable."""
    for source in profile["files"]:
        for check in source["checks"]:
            if check["kind"] == "fixed" and check["rows_failed"] == 0:
                continue
            assert check["recommendation"], f"{check['check_name']} has no remediation"
            assert check["handled_in"], f"{check['check_name']} names no file"


def test_remediation_targets_exist(profile):
    """Every file a finding points at must actually be in the repo."""
    for source in profile["files"]:
        for check in source["checks"]:
            where = check.get("handled_in")
            if not where:
                continue
            candidates = [ROOT / where, ROOT / "src" / "retail_intel" / where]
            assert any(path.exists() for path in candidates), (
                f"{check['check_name']} points at {where}, which does not exist"
            )


def test_order_items_is_clean_on_every_fixed_check(profile):
    """The headline claim the page makes about the transactional core."""
    items = next(f for f in profile["files"] if f["label"] == "Order items")
    breaches = [c for c in items["checks"]
                if c["kind"] == "fixed" and c["rows_failed"] > 0]
    assert not breaches, f"order_items is not clean: {[c['check_name'] for c in breaches]}"


def test_sparsity_checks_are_not_scored_as_defects(profile):
    """Expected sparsity must be INFO, or the score reads ~53 and misleads."""
    reviews = next(f for f in profile["files"] if f["label"] == "Reviews")
    for name in ("reviews with no body", "reviews with no title"):
        check = next(c for c in reviews["checks"] if c["check_name"] == name)
        assert check["severity"] == "INFO"
    assert profile["defaults"]["score_info"] is False
