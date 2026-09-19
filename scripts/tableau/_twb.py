"""Minimal Tableau workbook (.twb) writer.

A .twb is plain XML. Generating it rather than clicking it has three concrete
benefits here:

* the dashboards are rebuilt from the pipeline's own extracts by running a
  script, so they cannot drift from the data;
* the design is reviewable in a diff, like the rest of the repo;
* absolute CSV paths are injected at build time, so the workbooks open on any
  machine that has run the pipeline.

Only the subset of the format this project needs is implemented. It is not a
general-purpose Tableau library.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_DATATYPE = {
    "object": "string",
    "string": "string",
    "int64": "integer",
    "float64": "real",
    "bool": "boolean",
}
_ROLE = {"string": "dimension", "boolean": "dimension", "date": "dimension"}
_TYPE = {"string": "nominal", "boolean": "nominal", "date": "ordinal"}

DATE_COLUMNS = {"month", "week", "forecast_week", "origin_week"}


def _ident(text: str) -> str:
    """A stable short id, so regenerating a workbook produces the same names."""
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _title_case(name: str) -> str:
    """snake_case column -> the caption a reader should see on the shelf."""
    keep_upper = {"p95", "p50", "cv", "mom", "yoy", "ma", "pct", "aov", "rfm"}
    return " ".join(
        word.upper() if word.lower() in keep_upper else word.capitalize()
        for word in name.replace("_", " ").split()
    )


@dataclass
class Column:
    name: str
    datatype: str
    role: str
    coltype: str
    caption: str


@dataclass
class DataSource:
    """One CSV, connected through Tableau's text-file connector."""

    path: Path
    caption: str
    columns: list[Column] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = f"federated.{_ident(str(self.path))}"
        self.conn_id = f"textscan.{_ident(str(self.path) + 'c')}"
        frame = pd.read_csv(self.path, nrows=500)
        for name in frame.columns:
            datatype = (
                "date" if name in DATE_COLUMNS
                else _DATATYPE.get(str(frame[name].dtype), "string")
            )
            self.columns.append(
                Column(
                    name=name,
                    datatype=datatype,
                    role=_ROLE.get(datatype, "measure"),
                    coltype=_TYPE.get(datatype, "quantitative"),
                    caption=_title_case(name),
                )
            )

    def column(self, name: str) -> Column:
        for candidate in self.columns:
            if candidate.name == name:
                return candidate
        raise KeyError(
            f"{self.caption}: no column '{name}'. Available: "
            f"{[c.name for c in self.columns]}"
        )

    # -- shelf references -------------------------------------------------
    # Tableau names a field on a shelf by aggregation and data role, e.g.
    # [none:category:nk] for a discrete dimension and [sum:revenue:qk] for a
    # continuous measure. Every reference used on a shelf must also be
    # declared as a <column-instance> in that worksheet's dependencies.
    def dim(self, name: str) -> str:
        self.column(name)
        return f"[{self.id}].[none:{name}:nk]"

    def measure(self, name: str, agg: str = "sum") -> str:
        self.column(name)
        return f"[{self.id}].[{agg}:{name}:qk]"

    def date(self, name: str) -> str:
        self.column(name)
        return f"[{self.id}].[none:{name}:qk]"

    def to_xml(self) -> ET.Element:
        datasource = ET.Element(
            "datasource", caption=self.caption, inline="true",
            name=self.id, version="18.1",
        )
        connection = ET.SubElement(datasource, "connection", **{"class": "federated"})

        named = ET.SubElement(connection, "named-connections")
        named_connection = ET.SubElement(
            named, "named-connection", caption=self.path.stem, name=self.conn_id
        )
        ET.SubElement(
            named_connection, "connection",
            **{"class": "textscan", "directory": str(self.path.parent),
               "filename": self.path.name, "password": "", "server": ""},
        )

        relation = ET.SubElement(
            connection, "relation", connection=self.conn_id, name=self.path.name,
            table=f"[{self.path.stem}#csv]", type="table",
        )
        columns = ET.SubElement(
            relation, "columns",
            **{"character-set": "UTF-8", "header": "yes",
               "locale": "en_US", "separator": ","},
        )
        for ordinal, item in enumerate(self.columns):
            ET.SubElement(columns, "column", datatype=item.datatype,
                          name=item.name, ordinal=str(ordinal))

        for item in self.columns:
            ET.SubElement(
                datasource, "column", caption=item.caption, datatype=item.datatype,
                name=f"[{item.name}]", role=item.role, type=item.coltype,
            )
        return datasource


# Tableau spells the aggregation twice: once in the field reference
# ([sum:x:qk]) and once as the column-instance derivation ('Sum'). They must
# agree or the field resolves to nothing.
_DERIVATION = {
    "sum": "Sum", "avg": "Avg", "max": "Max", "min": "Min",
    "cnt": "Count", "none": "None",
}


def _parse_ref(reference: str) -> tuple[str, str, str]:
    """[federated.x].[sum:revenue:qk] -> (datasource id, 'sum', 'revenue')."""
    source, field_part = reference.split("].[")
    source = source.lstrip("[")
    field_part = field_part.rstrip("]")
    aggregation, name, _ = field_part.split(":")
    return source, aggregation, name


@dataclass
class Worksheet:
    """One Tableau sheet.

    Deliberately narrow: rows, columns, a mark type and a few encodings. Every
    sheet in this project is one of those shapes, and keeping the writer small
    keeps the generated XML close to what Tableau itself produces.
    """

    name: str
    source: DataSource
    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    mark: str = "Automatic"
    encodings: dict[str, str] = field(default_factory=dict)
    filters: list[tuple[str, list[str]]] = field(default_factory=list)
    title: str | None = None

    def _referenced(self) -> list[str]:
        seen: list[str] = []
        for reference in [*self.rows, *self.cols, *self.encodings.values()]:
            if reference not in seen:
                seen.append(reference)
        for column_name, _ in self.filters:
            reference = self.source.dim(column_name)
            if reference not in seen:
                seen.append(reference)
        return seen

    def to_xml(self) -> ET.Element:
        worksheet = ET.Element("worksheet", name=self.name)
        table = ET.SubElement(worksheet, "table")
        view = ET.SubElement(table, "view")

        datasources = ET.SubElement(view, "datasources")
        ET.SubElement(
            datasources, "datasource", caption=self.source.caption, name=self.source.id
        )

        dependencies = ET.SubElement(
            view, "datasource-dependencies", datasource=self.source.id
        )
        declared: set[str] = set()
        for reference in self._referenced():
            _, aggregation, column_name = _parse_ref(reference)
            column = self.source.column(column_name)
            if column_name not in declared:
                ET.SubElement(
                    dependencies, "column", caption=column.caption,
                    datatype=column.datatype, name=f"[{column_name}]",
                    role=column.role, type=column.coltype,
                )
                declared.add(column_name)
            instance_name = reference.split("].[")[1].rstrip("]")
            ET.SubElement(
                dependencies, "column-instance",
                column=f"[{column_name}]",
                derivation=_DERIVATION[aggregation],
                name=f"[{instance_name}]",
                pivot="key",
                type="quantitative" if instance_name.endswith("qk") else "nominal",
            )

        # Categorical filters, kept as explicit keep-lists so the sheet shows
        # exactly the rows the design calls for.
        for column_name, keep in self.filters:
            filter_element = ET.SubElement(
                view, "filter", **{"class": "categorical",
                                   "column": self.source.dim(column_name)}
            )
            members = ET.SubElement(filter_element, "groupfilter",
                                    function="union", user_ui_enabled="true")
            for value in keep:
                ET.SubElement(members, "groupfilter", function="member",
                              level=self.source.dim(column_name), member=f'"{value}"')

        ET.SubElement(view, "aggregation", value="true")

        ET.SubElement(table, "style")
        panes = ET.SubElement(table, "panes")
        pane = ET.SubElement(panes, "pane",
                             **{"selection-relaxation-option": "selection-relaxation-allow"})
        pane_view = ET.SubElement(pane, "view")
        ET.SubElement(pane_view, "breakdown", value="auto")
        ET.SubElement(pane, "mark", **{"class": self.mark})
        if self.encodings:
            encodings = ET.SubElement(pane, "encodings")
            for channel, reference in self.encodings.items():
                ET.SubElement(encodings, channel, column=reference)

        ET.SubElement(table, "rows").text = " / ".join(self.rows) if self.rows else ""
        ET.SubElement(table, "cols").text = " / ".join(self.cols) if self.cols else ""
        return worksheet


@dataclass
class Zone:
    """A rectangle on a dashboard, in Tableau's 0-100000 coordinate space."""

    sheet: str | None
    x: int
    y: int
    w: int
    h: int
    text: str | None = None


@dataclass
class Dashboard:
    name: str
    zones: list[Zone]
    width: int = 1600
    height: int = 900

    def to_xml(self, counter: list[int]) -> ET.Element:
        dashboard = ET.Element("dashboard", name=self.name)
        ET.SubElement(dashboard, "style")
        ET.SubElement(
            dashboard, "size",
            maxheight=str(self.height), maxwidth=str(self.width),
            minheight=str(self.height), minwidth=str(self.width),
        )
        zones = ET.SubElement(dashboard, "zones")
        counter[0] += 1
        root = ET.SubElement(
            zones, "zone", h="100000", id=str(counter[0]),
            **{"type-v2": "layout-basic"}, w="100000", x="0", y="0",
        )
        for zone in self.zones:
            counter[0] += 1
            attributes = {
                "h": str(zone.h), "id": str(counter[0]),
                "w": str(zone.w), "x": str(zone.x), "y": str(zone.y),
            }
            if zone.sheet:
                attributes["name"] = zone.sheet
                ET.SubElement(root, "zone", **attributes)
            else:
                # a text tile: used for dashboard titles and the simulated-data
                # caption that every operational view is required to carry
                attributes["type-v2"] = "text"
                text_zone = ET.SubElement(root, "zone", **attributes)
                formatted = ET.SubElement(text_zone, "formatted-text")
                ET.SubElement(formatted, "run").text = zone.text or ""
        return dashboard


def write_workbook(
    path: Path,
    sources: list[DataSource],
    worksheets: list[Worksheet],
    dashboards: list[Dashboard],
) -> None:
    workbook = ET.Element(
        "workbook",
        **{"source-build": "2024.1.0", "source-platform": "mac", "version": "18.1"},
    )
    ET.SubElement(workbook, "preferences")

    datasources = ET.SubElement(workbook, "datasources")
    for source in sources:
        datasources.append(source.to_xml())

    worksheet_root = ET.SubElement(workbook, "worksheets")
    for worksheet in worksheets:
        worksheet_root.append(worksheet.to_xml())

    if dashboards:
        dashboard_root = ET.SubElement(workbook, "dashboards")
        counter = [0]
        for dashboard in dashboards:
            dashboard_root.append(dashboard.to_xml(counter))

    windows = ET.SubElement(workbook, "windows", **{"source-height": "30"})
    for worksheet in worksheets:
        ET.SubElement(windows, "window", **{"class": "worksheet", "name": worksheet.name})
    for dashboard in dashboards:
        window = ET.SubElement(
            windows, "window",
            **{"class": "dashboard", "maximized": "true", "name": dashboard.name},
        )
        ET.SubElement(window, "viewpoints")

    ET.indent(workbook, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<?xml version='1.0' encoding='utf-8' ?>\n\n"
        + ET.tostring(workbook, encoding="unicode")
    )
