"""Build analysis-ready Excel workbooks for brain-strain results.

The workbook model in this module is deliberately independent of the viewer.
New result sections can be added as :class:`WorkbookSheet` instances without
changing the XLSX package writer or the user-interface code.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import TypeAlias
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import numpy.typing as npt

CellValue: TypeAlias = str | int | float | bool | None

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_RELATIONSHIP_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_CORE_PROPERTIES_NS = (
    "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
)
_DUBLIN_CORE_NS = "http://purl.org/dc/elements/1.1/"
_DUBLIN_TERMS_NS = "http://purl.org/dc/terms/"
_DUBLIN_TYPE_NS = "http://purl.org/dc/dcmitype/"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_EXTENDED_PROPERTIES_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
)
_VTYPES_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
_MAX_EXCEL_ROWS = 1_048_576
_MAX_EXCEL_COLUMNS = 16_384
_INVALID_SHEET_NAME_CHARACTERS = frozenset("[]:*?/\\")


def _qualified(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _normalise_cell_value(value: object) -> CellValue:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if isfinite(number) else None
    return str(value)


@dataclass(frozen=True, slots=True)
class WorkbookSheet:
    """One flat, filterable worksheet in a result workbook."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[CellValue, ...], ...]

    @classmethod
    def from_rows(
        cls,
        name: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[object]],
    ) -> WorkbookSheet:
        normalised_columns = tuple(str(column) for column in columns)
        normalised_rows = tuple(
            tuple(_normalise_cell_value(value) for value in row)
            for row in rows
        )
        return cls(name, normalised_columns, normalised_rows)

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 31:
            raise ValueError("worksheet names must contain 1 to 31 characters")
        if any(character in self.name for character in _INVALID_SHEET_NAME_CHARACTERS):
            raise ValueError(f"invalid worksheet name: {self.name!r}")
        if not self.columns:
            raise ValueError(f"worksheet {self.name!r} must have columns")
        if len(self.columns) > _MAX_EXCEL_COLUMNS:
            raise ValueError(f"worksheet {self.name!r} exceeds Excel's column limit")
        if len(self.rows) + 1 > _MAX_EXCEL_ROWS:
            raise ValueError(f"worksheet {self.name!r} exceeds Excel's row limit")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"worksheet {self.name!r} has duplicate columns")
        for index, row in enumerate(self.rows, start=2):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"worksheet {self.name!r} row {index} has {len(row)} "
                    f"values; expected {len(self.columns)}"
                )


@dataclass(slots=True)
class ResultWorkbook:
    """Extensible collection of worksheets that can be saved as XLSX."""

    sheets: list[WorkbookSheet] = field(default_factory=list)

    def add_sheet(
        self,
        name: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[object]],
    ) -> WorkbookSheet:
        if any(sheet.name.casefold() == name.casefold() for sheet in self.sheets):
            raise ValueError(f"duplicate worksheet name: {name!r}")
        sheet = WorkbookSheet.from_rows(name, columns, rows)
        self.sheets.append(sheet)
        return sheet

    def save(self, path: str | Path) -> Path:
        """Write the workbook atomically and return its absolute path."""
        if not self.sheets:
            raise ValueError("a result workbook must contain at least one sheet")
        destination = Path(path).expanduser()
        if destination.suffix.casefold() != ".xlsx":
            destination = destination.with_suffix(".xlsx")
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".xlsx",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            _write_xlsx_package(temporary_path, self.sheets)
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination


@dataclass(frozen=True, slots=True)
class ResultParameter:
    """One selected input or interface setting written to Parameters."""

    category: str
    name: str
    value: CellValue
    unit: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class ResultSeries:
    """One real or simulated time-by-cell scalar result series."""

    source_type: str
    source_name: str
    field_name: str
    times: npt.ArrayLike
    values: npt.ArrayLike
    unit: str = ""

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("result-series times must be a non-empty vector")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise ValueError("result-series times must be finite and increasing")
        if values.ndim != 2 or values.shape[0] != times.size:
            raise ValueError(
                "result-series values must have shape (n_times, n_cells)"
            )
        if values.shape[1] == 0 or not np.isfinite(values).any():
            raise ValueError("result-series values contain no finite cell data")
        # Keep views of the viewer-owned arrays: exporting a large mesh should
        # not duplicate the complete real and simulated data in memory.
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ResultExportData:
    """All inputs used to assemble the standard result workbook sections."""

    simulation_method: str
    active_source: str
    active_frame_index: int
    active_time: float
    parameters: tuple[ResultParameter, ...]
    series: tuple[ResultSeries, ...]
    selected_cell_ids: tuple[int, ...] = ()
    source_element_ids: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.simulation_method.strip():
            raise ValueError("simulation_method cannot be empty")
        if not self.series:
            raise ValueError("at least one result series is required")
        if self.active_frame_index < 0:
            raise ValueError("active_frame_index cannot be negative")
        if not isfinite(float(self.active_time)):
            raise ValueError("active_time must be finite")


def build_result_workbook(data: ResultExportData) -> ResultWorkbook:
    """Build the standard analysis workbook from a viewer result snapshot."""
    workbook = ResultWorkbook()
    workbook.add_sheet(
        "Results Summary",
        ("Metric", "Value", "Unit", "Source", "Field"),
        _summary_rows(data),
    )
    workbook.add_sheet(
        "Parameters",
        ("Category", "Parameter", "Value", "Unit", "Description"),
        (
            (
                parameter.category,
                parameter.name,
                parameter.value,
                parameter.unit,
                parameter.description,
            )
            for parameter in data.parameters
        ),
    )
    workbook.add_sheet(
        "Frame Results",
        (
            "Source Type",
            "Data Source",
            "Field",
            "Frame Index",
            "Time (s)",
            "Minimum Value",
            "Maximum Value",
            "Mean Value",
            "Standard Deviation",
            "Finite Cell Count",
            "Peak Cell Index",
            "Unit",
        ),
        _frame_result_rows(data.series),
    )
    workbook.add_sheet(
        "Selected Cell Results",
        (
            "Source Type",
            "Data Source",
            "Field",
            "Cell Index",
            "Source Element ID",
            "Frame Index Used",
            "Time (s)",
            "Value at Frame",
            "Minimum Value",
            "Maximum Value",
            "Mean Value",
            "Peak Frame Index",
            "Peak Time (s)",
            "Unit",
        ),
        _selected_cell_rows(data),
    )
    workbook.add_sheet(
        "Data Dictionary",
        ("Worksheet", "Column or Section", "Definition"),
        _data_dictionary_rows(),
    )
    return workbook


def _summary_rows(data: ResultExportData) -> Iterable[tuple[CellValue, ...]]:
    yield ("Export schema version", "1.0", "", "Workbook", "")
    yield (
        "Simulation method",
        data.simulation_method,
        "",
        "Simulation",
        "Maximum shear strain",
    )
    yield ("Active result source", data.active_source, "", "Interface", "")
    yield (
        "Active frame index",
        data.active_frame_index,
        "",
        "Interface",
        "",
    )
    yield ("Active time", data.active_time, "s", "Interface", "")

    for series in data.series:
        values = np.asarray(series.values, dtype=np.float64)
        finite = np.isfinite(values)
        comparable = np.where(finite, values, -np.inf)
        flat_index = int(np.argmax(comparable))
        frame_index, cell_index = np.unravel_index(flat_index, values.shape)
        maximum = float(values[frame_index, cell_index])
        metric = (
            "Actual real-data global maximum"
            if series.source_type.casefold() == "real"
            else "Calculated maximum strain"
        )
        yield (
            metric,
            maximum,
            series.unit,
            series.source_name,
            series.field_name,
        )
        yield (
            "Global maximum frame index",
            int(frame_index),
            "",
            series.source_name,
            series.field_name,
        )
        yield (
            "Global maximum time",
            float(series.times[frame_index]),
            "s",
            series.source_name,
            series.field_name,
        )
        yield (
            "Global maximum cell index",
            int(cell_index),
            "",
            series.source_name,
            series.field_name,
        )


def _frame_result_rows(
    series_collection: Sequence[ResultSeries],
) -> Iterable[tuple[CellValue, ...]]:
    for series in series_collection:
        for frame_index, (time_value, frame) in enumerate(
            zip(series.times, series.values, strict=True)
        ):
            finite = np.isfinite(frame)
            if finite.any():
                finite_values = frame[finite]
                comparable = np.where(finite, frame, -np.inf)
                minimum: CellValue = float(np.min(finite_values))
                maximum: CellValue = float(np.max(finite_values))
                mean: CellValue = float(np.mean(finite_values))
                standard_deviation: CellValue = float(np.std(finite_values))
                peak_cell: CellValue = int(np.argmax(comparable))
            else:
                minimum = maximum = mean = standard_deviation = peak_cell = None
            yield (
                series.source_type,
                series.source_name,
                series.field_name,
                frame_index,
                float(time_value),
                minimum,
                maximum,
                mean,
                standard_deviation,
                int(np.count_nonzero(finite)),
                peak_cell,
                series.unit,
            )


def _selected_cell_rows(
    data: ResultExportData,
) -> Iterable[tuple[CellValue, ...]]:
    for series in data.series:
        values = np.asarray(series.values, dtype=np.float64)
        frame_index = min(data.active_frame_index, values.shape[0] - 1)
        for cell_index in data.selected_cell_ids:
            if cell_index < 0 or cell_index >= values.shape[1]:
                continue
            history = values[:, cell_index]
            finite = np.isfinite(history)
            if finite.any():
                comparable = np.where(finite, history, -np.inf)
                peak_frame = int(np.argmax(comparable))
                minimum: CellValue = float(np.min(history[finite]))
                maximum: CellValue = float(np.max(history[finite]))
                mean: CellValue = float(np.mean(history[finite]))
                peak_time: CellValue = float(series.times[peak_frame])
            else:
                peak_frame = 0
                minimum = maximum = mean = peak_time = None
            value_at_frame = (
                float(history[frame_index])
                if np.isfinite(history[frame_index])
                else None
            )
            yield (
                series.source_type,
                series.source_name,
                series.field_name,
                cell_index,
                data.source_element_ids.get(cell_index),
                frame_index,
                float(series.times[frame_index]),
                value_at_frame,
                minimum,
                maximum,
                mean,
                peak_frame if finite.any() else None,
                peak_time,
                series.unit,
            )


def _data_dictionary_rows() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "Results Summary",
            "Actual real-data global maximum",
            "Largest finite value present in the retained real result series.",
        ),
        (
            "Results Summary",
            "Calculated maximum strain",
            "Largest finite generalized-Maxwell strain value in the calculated series.",
        ),
        (
            "Parameters",
            "All columns",
            "Chosen simulation inputs and interface state at export time; "
            "numeric values remain numeric.",
        ),
        (
            "Frame Results",
            "One row per source and frame",
            "Descriptive statistics across all finite mesh-cell values. "
            "Standard deviation uses the population definition.",
        ),
        (
            "Selected Cell Results",
            "One row per source and selected cell",
            "Actual or calculated value at the selected frame plus "
            "time-history statistics for that cell.",
        ),
        (
            "Workbook",
            "Scientific interpretation",
            "Real and simulated fields are exported side by side without "
            "assuming matching units, anatomy, or scientific equivalence.",
        ),
    )


def _write_xlsx_package(path: Path, sheets: Sequence[WorkbookSheet]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", _root_relationships_xml())
        archive.writestr("docProps/core.xml", _core_properties_xml())
        archive.writestr("docProps/app.xml", _app_properties_xml(sheets))
        archive.writestr("xl/workbook.xml", _workbook_xml(sheets))
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            _workbook_relationships_xml(len(sheets)),
        )
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(sheet),
            )


def _content_types_xml(sheet_count: int) -> bytes:
    root = ET.Element(_qualified(_CONTENT_TYPES_NS, "Types"))
    ET.SubElement(
        root,
        _qualified(_CONTENT_TYPES_NS, "Default"),
        Extension="rels",
        ContentType="application/vnd.openxmlformats-package.relationships+xml",
    )
    ET.SubElement(
        root,
        _qualified(_CONTENT_TYPES_NS, "Default"),
        Extension="xml",
        ContentType="application/xml",
    )
    overrides = (
        (
            "/xl/workbook.xml",
            "application/vnd.openxmlformats-officedocument.spreadsheetml."
            "sheet.main+xml",
        ),
        (
            "/xl/styles.xml",
            "application/vnd.openxmlformats-officedocument.spreadsheetml."
            "styles+xml",
        ),
        (
            "/docProps/core.xml",
            "application/vnd.openxmlformats-package.core-properties+xml",
        ),
        (
            "/docProps/app.xml",
            "application/vnd.openxmlformats-officedocument."
            "extended-properties+xml",
        ),
    )
    for part_name, content_type in overrides:
        ET.SubElement(
            root,
            _qualified(_CONTENT_TYPES_NS, "Override"),
            PartName=part_name,
            ContentType=content_type,
        )
    for index in range(1, sheet_count + 1):
        ET.SubElement(
            root,
            _qualified(_CONTENT_TYPES_NS, "Override"),
            PartName=f"/xl/worksheets/sheet{index}.xml",
            ContentType=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml."
                "worksheet+xml"
            ),
        )
    return _xml_bytes(root)


def _root_relationships_xml() -> bytes:
    root = ET.Element(_qualified(_PACKAGE_RELATIONSHIP_NS, "Relationships"))
    relationships = (
        (
            "rId1",
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/officeDocument",
            "xl/workbook.xml",
        ),
        (
            "rId2",
            "http://schemas.openxmlformats.org/package/2006/relationships/"
            "metadata/core-properties",
            "docProps/core.xml",
        ),
        (
            "rId3",
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/extended-properties",
            "docProps/app.xml",
        ),
    )
    for relationship_id, relationship_type, target in relationships:
        ET.SubElement(
            root,
            _qualified(_PACKAGE_RELATIONSHIP_NS, "Relationship"),
            Id=relationship_id,
            Type=relationship_type,
            Target=target,
        )
    return _xml_bytes(root)


def _core_properties_xml() -> bytes:
    ET.register_namespace("cp", _CORE_PROPERTIES_NS)
    ET.register_namespace("dc", _DUBLIN_CORE_NS)
    ET.register_namespace("dcterms", _DUBLIN_TERMS_NS)
    ET.register_namespace("dcmitype", _DUBLIN_TYPE_NS)
    ET.register_namespace("xsi", _XSI_NS)
    root = ET.Element(_qualified(_CORE_PROPERTIES_NS, "coreProperties"))
    ET.SubElement(root, _qualified(_DUBLIN_CORE_NS, "creator")).text = (
        "Brain Strain Visualisation"
    )
    ET.SubElement(root, _qualified(_CORE_PROPERTIES_NS, "lastModifiedBy")).text = (
        "Brain Strain Visualisation"
    )
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    created = ET.SubElement(root, _qualified(_DUBLIN_TERMS_NS, "created"))
    created.set(_qualified(_XSI_NS, "type"), "dcterms:W3CDTF")
    created.text = timestamp
    modified = ET.SubElement(root, _qualified(_DUBLIN_TERMS_NS, "modified"))
    modified.set(_qualified(_XSI_NS, "type"), "dcterms:W3CDTF")
    modified.text = timestamp
    return _xml_bytes(root)


def _app_properties_xml(sheets: Sequence[WorkbookSheet]) -> bytes:
    ET.register_namespace("vt", _VTYPES_NS)
    root = ET.Element(_qualified(_EXTENDED_PROPERTIES_NS, "Properties"))
    ET.SubElement(root, _qualified(_EXTENDED_PROPERTIES_NS, "Application")).text = (
        "Brain Strain Visualisation"
    )
    heading_pairs = ET.SubElement(
        root, _qualified(_EXTENDED_PROPERTIES_NS, "HeadingPairs")
    )
    heading_vector = ET.SubElement(
        heading_pairs,
        _qualified(_VTYPES_NS, "vector"),
        size="2",
        baseType="variant",
    )
    variant = ET.SubElement(heading_vector, _qualified(_VTYPES_NS, "variant"))
    ET.SubElement(variant, _qualified(_VTYPES_NS, "lpstr")).text = "Worksheets"
    variant = ET.SubElement(heading_vector, _qualified(_VTYPES_NS, "variant"))
    ET.SubElement(variant, _qualified(_VTYPES_NS, "i4")).text = str(len(sheets))
    titles = ET.SubElement(root, _qualified(_EXTENDED_PROPERTIES_NS, "TitlesOfParts"))
    titles_vector = ET.SubElement(
        titles,
        _qualified(_VTYPES_NS, "vector"),
        size=str(len(sheets)),
        baseType="lpstr",
    )
    for sheet in sheets:
        ET.SubElement(titles_vector, _qualified(_VTYPES_NS, "lpstr")).text = sheet.name
    return _xml_bytes(root)


def _workbook_xml(sheets: Sequence[WorkbookSheet]) -> bytes:
    ET.register_namespace("r", _RELATIONSHIP_NS)
    root = ET.Element(_qualified(_SPREADSHEET_NS, "workbook"))
    book_views = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "bookViews"))
    ET.SubElement(book_views, _qualified(_SPREADSHEET_NS, "workbookView"))
    sheet_nodes = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "sheets"))
    for index, sheet in enumerate(sheets, start=1):
        node = ET.SubElement(
            sheet_nodes,
            _qualified(_SPREADSHEET_NS, "sheet"),
            name=sheet.name,
            sheetId=str(index),
        )
        node.set(_qualified(_RELATIONSHIP_NS, "id"), f"rId{index}")
    ET.SubElement(
        root,
        _qualified(_SPREADSHEET_NS, "calcPr"),
        calcId="191029",
        fullCalcOnLoad="1",
        forceFullCalc="1",
    )
    return _xml_bytes(root)


def _workbook_relationships_xml(sheet_count: int) -> bytes:
    root = ET.Element(_qualified(_PACKAGE_RELATIONSHIP_NS, "Relationships"))
    for index in range(1, sheet_count + 1):
        ET.SubElement(
            root,
            _qualified(_PACKAGE_RELATIONSHIP_NS, "Relationship"),
            Id=f"rId{index}",
            Type=(
                "http://schemas.openxmlformats.org/officeDocument/2006/"
                "relationships/worksheet"
            ),
            Target=f"worksheets/sheet{index}.xml",
        )
    ET.SubElement(
        root,
        _qualified(_PACKAGE_RELATIONSHIP_NS, "Relationship"),
        Id=f"rId{sheet_count + 1}",
        Type=(
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/styles"
        ),
        Target="styles.xml",
    )
    return _xml_bytes(root)


def _styles_xml() -> bytes:
    root = ET.Element(_qualified(_SPREADSHEET_NS, "styleSheet"))
    number_formats = ET.SubElement(
        root, _qualified(_SPREADSHEET_NS, "numFmts"), count="1"
    )
    ET.SubElement(
        number_formats,
        _qualified(_SPREADSHEET_NS, "numFmt"),
        numFmtId="164",
        formatCode="0.000000",
    )
    fonts = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "fonts"), count="2")
    default_font = ET.SubElement(fonts, _qualified(_SPREADSHEET_NS, "font"))
    ET.SubElement(default_font, _qualified(_SPREADSHEET_NS, "sz"), val="11")
    ET.SubElement(default_font, _qualified(_SPREADSHEET_NS, "name"), val="Calibri")
    header_font = ET.SubElement(fonts, _qualified(_SPREADSHEET_NS, "font"))
    ET.SubElement(header_font, _qualified(_SPREADSHEET_NS, "b"))
    ET.SubElement(header_font, _qualified(_SPREADSHEET_NS, "color"), rgb="FFFFFFFF")
    ET.SubElement(header_font, _qualified(_SPREADSHEET_NS, "sz"), val="11")
    ET.SubElement(header_font, _qualified(_SPREADSHEET_NS, "name"), val="Calibri")
    fills = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "fills"), count="3")
    fill = ET.SubElement(fills, _qualified(_SPREADSHEET_NS, "fill"))
    ET.SubElement(fill, _qualified(_SPREADSHEET_NS, "patternFill"), patternType="none")
    fill = ET.SubElement(fills, _qualified(_SPREADSHEET_NS, "fill"))
    ET.SubElement(
        fill,
        _qualified(_SPREADSHEET_NS, "patternFill"),
        patternType="gray125",
    )
    fill = ET.SubElement(fills, _qualified(_SPREADSHEET_NS, "fill"))
    pattern = ET.SubElement(
        fill, _qualified(_SPREADSHEET_NS, "patternFill"), patternType="solid"
    )
    ET.SubElement(pattern, _qualified(_SPREADSHEET_NS, "fgColor"), rgb="FF0F766E")
    ET.SubElement(pattern, _qualified(_SPREADSHEET_NS, "bgColor"), indexed="64")
    borders = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "borders"), count="2")
    border = ET.SubElement(borders, _qualified(_SPREADSHEET_NS, "border"))
    for side in ("left", "right", "top", "bottom", "diagonal"):
        ET.SubElement(border, _qualified(_SPREADSHEET_NS, side))
    border = ET.SubElement(borders, _qualified(_SPREADSHEET_NS, "border"))
    for side in ("left", "right", "top"):
        ET.SubElement(border, _qualified(_SPREADSHEET_NS, side))
    bottom = ET.SubElement(
        border, _qualified(_SPREADSHEET_NS, "bottom"), style="thin"
    )
    ET.SubElement(bottom, _qualified(_SPREADSHEET_NS, "color"), rgb="FF0B5345")
    ET.SubElement(border, _qualified(_SPREADSHEET_NS, "diagonal"))
    cell_style_xfs = ET.SubElement(
        root, _qualified(_SPREADSHEET_NS, "cellStyleXfs"), count="1"
    )
    ET.SubElement(
        cell_style_xfs,
        _qualified(_SPREADSHEET_NS, "xf"),
        numFmtId="0",
        fontId="0",
        fillId="0",
        borderId="0",
    )
    cell_xfs = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "cellXfs"), count="3")
    ET.SubElement(
        cell_xfs,
        _qualified(_SPREADSHEET_NS, "xf"),
        numFmtId="0",
        fontId="0",
        fillId="0",
        borderId="0",
        xfId="0",
    )
    header_xf = ET.SubElement(
        cell_xfs,
        _qualified(_SPREADSHEET_NS, "xf"),
        numFmtId="0",
        fontId="1",
        fillId="2",
        borderId="1",
        xfId="0",
        applyFont="1",
        applyFill="1",
        applyBorder="1",
        applyAlignment="1",
    )
    ET.SubElement(
        header_xf,
        _qualified(_SPREADSHEET_NS, "alignment"),
        horizontal="left",
        vertical="center",
    )
    ET.SubElement(
        cell_xfs,
        _qualified(_SPREADSHEET_NS, "xf"),
        numFmtId="164",
        fontId="0",
        fillId="0",
        borderId="0",
        xfId="0",
        applyNumberFormat="1",
    )
    cell_styles = ET.SubElement(
        root, _qualified(_SPREADSHEET_NS, "cellStyles"), count="1"
    )
    ET.SubElement(
        cell_styles,
        _qualified(_SPREADSHEET_NS, "cellStyle"),
        name="Normal",
        xfId="0",
        builtinId="0",
    )
    return _xml_bytes(root)


def _worksheet_xml(sheet: WorkbookSheet) -> bytes:
    root = ET.Element(_qualified(_SPREADSHEET_NS, "worksheet"))
    last_column = _column_name(len(sheet.columns))
    last_row = len(sheet.rows) + 1
    ET.SubElement(
        root,
        _qualified(_SPREADSHEET_NS, "dimension"),
        ref=f"A1:{last_column}{last_row}",
    )
    sheet_views = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "sheetViews"))
    sheet_view = ET.SubElement(
        sheet_views,
        _qualified(_SPREADSHEET_NS, "sheetView"),
        workbookViewId="0",
        showGridLines="0",
    )
    ET.SubElement(
        sheet_view,
        _qualified(_SPREADSHEET_NS, "pane"),
        ySplit="1",
        topLeftCell="A2",
        activePane="bottomLeft",
        state="frozen",
    )
    ET.SubElement(
        sheet_view,
        _qualified(_SPREADSHEET_NS, "selection"),
        pane="bottomLeft",
        activeCell="A2",
        sqref="A2",
    )
    ET.SubElement(
        root,
        _qualified(_SPREADSHEET_NS, "sheetFormatPr"),
        defaultRowHeight="15",
    )
    columns = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "cols"))
    for index, width in enumerate(_column_widths(sheet), start=1):
        ET.SubElement(
            columns,
            _qualified(_SPREADSHEET_NS, "col"),
            min=str(index),
            max=str(index),
            width=f"{width:.1f}",
            customWidth="1",
        )
    sheet_data = ET.SubElement(root, _qualified(_SPREADSHEET_NS, "sheetData"))
    header_row = ET.SubElement(
        sheet_data,
        _qualified(_SPREADSHEET_NS, "row"),
        r="1",
        ht="22",
        customHeight="1",
    )
    for column_index, value in enumerate(sheet.columns, start=1):
        _append_cell(header_row, 1, column_index, value, header=True)
    for row_index, row_values in enumerate(sheet.rows, start=2):
        row = ET.SubElement(
            sheet_data, _qualified(_SPREADSHEET_NS, "row"), r=str(row_index)
        )
        for column_index, value in enumerate(row_values, start=1):
            _append_cell(row, row_index, column_index, value, header=False)
    ET.SubElement(
        root,
        _qualified(_SPREADSHEET_NS, "autoFilter"),
        ref=f"A1:{last_column}{last_row}",
    )
    ET.SubElement(
        root,
        _qualified(_SPREADSHEET_NS, "pageMargins"),
        left="0.7",
        right="0.7",
        top="0.75",
        bottom="0.75",
        header="0.3",
        footer="0.3",
    )
    return _xml_bytes(root)


def _append_cell(
    row: ET.Element,
    row_index: int,
    column_index: int,
    value: CellValue,
    *,
    header: bool,
) -> None:
    if value is None:
        return
    reference = f"{_column_name(column_index)}{row_index}"
    if isinstance(value, bool):
        cell = ET.SubElement(
            row,
            _qualified(_SPREADSHEET_NS, "c"),
            r=reference,
            t="b",
            s="1" if header else "0",
        )
        ET.SubElement(cell, _qualified(_SPREADSHEET_NS, "v")).text = (
            "1" if value else "0"
        )
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        cell = ET.SubElement(
            row,
            _qualified(_SPREADSHEET_NS, "c"),
            r=reference,
            s="1" if header else ("2" if isinstance(value, float) else "0"),
        )
        ET.SubElement(cell, _qualified(_SPREADSHEET_NS, "v")).text = repr(value)
        return
    cell = ET.SubElement(
        row,
        _qualified(_SPREADSHEET_NS, "c"),
        r=reference,
        t="inlineStr",
        s="1" if header else "0",
    )
    inline_string = ET.SubElement(cell, _qualified(_SPREADSHEET_NS, "is"))
    text_node = ET.SubElement(inline_string, _qualified(_SPREADSHEET_NS, "t"))
    text = str(value)
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text


def _column_widths(sheet: WorkbookSheet) -> tuple[float, ...]:
    widths: list[float] = []
    for index, header in enumerate(sheet.columns):
        maximum = len(header)
        for row in sheet.rows[:2000]:
            value = row[index]
            if value is not None:
                maximum = max(maximum, len(str(value)))
        widths.append(float(min(max(maximum + 2, 10), 48)))
    return tuple(widths)


def _column_name(index: int) -> str:
    if index < 1 or index > _MAX_EXCEL_COLUMNS:
        raise ValueError("column index is outside Excel's supported range")
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


__all__ = [
    "ResultExportData",
    "ResultParameter",
    "ResultSeries",
    "ResultWorkbook",
    "WorkbookSheet",
    "build_result_workbook",
]
