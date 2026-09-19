"""Tests for the generated Tableau workbooks.

The workbooks are build artefacts, so they get the same treatment as the rest
of the pipeline: the properties that matter are asserted rather than eyeballed
once and assumed to hold.

The property that matters most is audience separation. The brief's hard
requirement is that the two dashboards are designed for different readers, not
the same charts rendered twice -- and the first way that erodes is one workbook
quietly reaching for the other's extract.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ROOT / "tableau" / "operations_command_centre.twb"
EXECUTIVE = ROOT / "tableau" / "executive_review.twb"

pytestmark = pytest.mark.skipif(
    not (OPERATIONS.exists() and EXECUTIVE.exists()),
    reason="workbooks not built -- run `make dashboards`",
)


@pytest.fixture(scope="module")
def operations() -> ET.Element:
    return ET.parse(OPERATIONS).getroot()


@pytest.fixture(scope="module")
def executive() -> ET.Element:
    return ET.parse(EXECUTIVE).getroot()


def test_both_workbooks_are_well_formed(operations, executive):
    for workbook in (operations, executive):
        assert workbook.tag == "workbook"
        assert workbook.find("datasources") is not None
        assert workbook.find("worksheets") is not None
        assert workbook.find("dashboards") is not None


def test_workbooks_do_not_share_extracts():
    """Operational and executive views must read different data."""
    assert "exec_" not in OPERATIONS.read_text()
    assert "ops_" not in EXECUTIVE.read_text()


def test_operational_workbook_declares_simulated_stock():
    """Any view built on assumed stock has to say so on the face of it."""
    assert "simulated" in OPERATIONS.read_text().lower()


def test_every_shelf_reference_is_declared(operations, executive):
    """A field on a shelf that has no column-instance resolves to nothing.

    Tableau does not error on this -- it renders an empty sheet. So the
    reference graph is checked here instead.
    """
    for workbook in (operations, executive):
        for worksheet in workbook.find("worksheets"):
            table = worksheet.find("table")
            declared = {
                instance.get("name")
                for instance in table.iter("column-instance")
            }
            used: set[str] = set()
            for shelf in ("rows", "cols"):
                text = (table.find(shelf).text or "").strip()
                for reference in text.split(" / "):
                    if reference:
                        used.add("[" + reference.split("].[")[1])
            for encoding in table.iter("encodings"):
                for channel in encoding:
                    used.add("[" + channel.get("column").split("].[")[1])

            missing = used - declared
            assert not missing, (
                f"{worksheet.get('name')}: shelf references with no "
                f"column-instance declaration: {sorted(missing)}"
            )


def test_each_worksheet_binds_exactly_one_datasource(operations, executive):
    for workbook in (operations, executive):
        for worksheet in workbook.find("worksheets"):
            sources = worksheet.find("table/view/datasources")
            assert len(sources) == 1, worksheet.get("name")


def test_dashboard_zones_reference_real_worksheets(operations, executive):
    """A zone naming a sheet that does not exist renders as a blank tile."""
    for workbook in (operations, executive):
        names = {sheet.get("name") for sheet in workbook.find("worksheets")}
        for dashboard in workbook.find("dashboards"):
            for zone in dashboard.iter("zone"):
                referenced = zone.get("name")
                if referenced and zone.get("type-v2") != "text":
                    assert referenced in names, (
                        f"{dashboard.get('name')}: zone references unknown sheet "
                        f"'{referenced}'"
                    )


def test_dashboard_zones_do_not_overflow(operations, executive):
    """Tableau's dashboard coordinate space runs 0-100000 in both axes."""
    for workbook in (operations, executive):
        for dashboard in workbook.find("dashboards"):
            for zone in dashboard.iter("zone"):
                x, y = int(zone.get("x")), int(zone.get("y"))
                w, h = int(zone.get("w")), int(zone.get("h"))
                assert x + w <= 100000, f"{zone.get('name')} overflows horizontally"
                assert y + h <= 100000, f"{zone.get('name')} overflows vertically"


def test_csv_paths_resolve(operations, executive):
    """Generated paths are absolute and must point at files that exist."""
    for workbook in (operations, executive):
        for connection in workbook.iter("connection"):
            if connection.get("class") == "textscan":
                path = Path(connection.get("directory")) / connection.get("filename")
                assert path.exists(), f"workbook points at a missing extract: {path}"
