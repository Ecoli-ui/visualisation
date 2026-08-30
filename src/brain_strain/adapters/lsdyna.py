"""Build one analysis-ready VTU model from an LS-DYNA mesh and results.

The LS-DYNA keyword mesh is the source of geometry and connectivity.  Optional
keyword files add part, section, material, and set metadata, while an optional
tabular result file adds cell results by joining on ``element_id`` (and
``part_id`` when the result supplies it).  The final product is a single
:class:`pyvista.UnstructuredGrid` that can be saved as ``.vtu``.

Result files may be CSV, whitespace-delimited text, JSON, NPY, or NPZ.  A
result must contain an ``element_id``/``eid`` column.  CSV and text files with
no header are interpreted as ``element_id, strain`` when they have two
columns.  By default results are mapped to solid elements because LS-DYNA
shell and solid element IDs commonly overlap.
"""

from __future__ import annotations

import json
import re
import warnings
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
from lsdyna_mesh_reader import Deck
from numpy.typing import NDArray

from .mre134_labels import (
    mapping_document as mre134_recode_mapping_document,
)
from .mre134_labels import (
    unified_label_for_recode_part,
)

PathInput = str | PathLike[str]
ResultTarget = Literal["solid", "shell", "all"]

_ELEMENT_ID_KEYS = ("element_id", "element id", "elementid", "eid")
_PART_ID_KEYS = ("part_id", "part id", "partid", "pid")
_TIME_KEYS = ("time", "times", "t")

# Broad tissue groups used by the ReCoDE head model.  Detailed anatomical
# names still come from *PART cards in part_list_full.k when that file is
# supplied or discovered beside the mesh.
_TISSUE_PIDS: tuple[tuple[str, frozenset[int]], ...] = (
    ("Skin", frozenset({260})),
    ("Skull", frozenset({257})),
    ("CSF", frozenset({24, 256, 258, 259})),
    (
        "Grey matter",
        frozenset(
            {
                3,
                8,
                10,
                11,
                12,
                13,
                17,
                18,
                26,
                28,
                42,
                47,
                49,
                50,
                51,
                52,
                53,
                54,
                58,
                60,
            }
        ),
    ),
    (
        "White matter",
        frozenset({2, 7, 41, 46, 77, 85, 251, 252, 253, 254, 255}),
    ),
    ("Ventricles", frozenset({4, 5, 14, 15, 43, 44})),
    ("Brain stem", frozenset({16})),
    ("Meninges", frozenset({900, 901, 902, 903, 904, 905})),
)
_TISSUE_NAMES = tuple(name for name, _ in _TISSUE_PIDS) + ("Other",)

_FALLBACK_PART_NAMES = {
    16: "Brain-Stem",
    24: "CSF",
    256: "CSF",
    257: "Skull",
    258: "CSF-by-Falx",
    259: "CSF-by-Tentorium",
    260: "Skin",
    900: "Falx",
    901: "Tentorium",
    902: "Pia_Matter",
    903: "Dura_Matter",
}


def _normalise_key(value: str) -> str:
    """Return a comparison key insensitive to case and punctuation."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _find_mapping_key(
    values: Mapping[str, Any],
    requested: str | None,
    alternatives: Sequence[str],
) -> str | None:
    """Find a mapping key without requiring one exact spelling."""
    normalised = {_normalise_key(str(key)): str(key) for key in values}
    if requested is not None:
        return normalised.get(_normalise_key(requested))
    for candidate in alternatives:
        match = normalised.get(_normalise_key(candidate))
        if match is not None:
            return match
    return None


def _existing_file(path: PathInput, description: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {file_path}")
    return file_path


def _validate_mesh_path(path: PathInput) -> Path:
    mesh_path = _existing_file(path, "LS-DYNA mesh")
    if mesh_path.suffix.casefold() != ".k":
        raise ValueError(
            f"Expected an LS-DYNA .k file, received "
            f"{mesh_path.suffix or '<none>'!r}"
        )
    return mesh_path


def _copy_source_element_ids(deck: Deck, grid: pv.UnstructuredGrid) -> None:
    """Attach source IDs and element dimensions in ``Deck.to_grid`` order."""
    # lsdyna_mesh_reader constructs grid cells from shell sections first,
    # followed by solid sections. Preserve that exact ordering here.
    sections = deck.element_shell_sections + deck.element_solid_sections
    element_ids = np.hstack(
        [np.asarray(section.eid, dtype=np.int64) for section in sections]
    )
    dimensions = np.hstack(
        [
            np.full(len(section.eid), 2, dtype=np.int8)
            for section in deck.element_shell_sections
        ]
        + [
            np.full(len(section.eid), 3, dtype=np.int8)
            for section in deck.element_solid_sections
        ]
    )
    if element_ids.size != grid.n_cells or dimensions.size != grid.n_cells:
        raise ValueError(
            "LS-DYNA element count does not match the converted cell count"
        )
    grid.cell_data["element_id"] = element_ids
    grid.cell_data["element_dimension"] = dimensions
    grid.cell_data["element_type"] = np.where(
        dimensions == 2,
        "shell",
        "solid",
    )


def _copy_normalised_ids(grid: pv.UnstructuredGrid) -> None:
    """Add consistently named ID arrays while preserving source arrays."""
    for name in list(grid.cell_data.keys()):
        normalised = _normalise_key(name)
        if normalised in {"pid", "partid"}:
            grid.cell_data["part_id"] = np.asarray(
                grid.cell_data[name],
                dtype=np.int32,
            )
        if normalised in {"eid", "elementid"}:
            grid.cell_data["element_id"] = np.asarray(
                grid.cell_data[name],
                dtype=np.int64,
            )

    for name in list(grid.point_data.keys()):
        if _normalise_key(name) in {"nid", "nodeid"}:
            grid.point_data["node_id"] = np.asarray(
                grid.point_data[name],
                dtype=np.int64,
            )


def _source_element_indices(path: Path) -> tuple[NDArray[np.int64], int]:
    """Return source-file element order aligned to shell-then-solid grid order."""
    source_index = 0
    section: str | None = None
    shell_indices: list[int] = []
    solid_indices: list[int] = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("*ELEMENT_SHELL"):
                section = "shell"
                continue
            if upper.startswith("*ELEMENT_SOLID"):
                section = "solid"
                continue
            if line.startswith("*"):
                section = None
                continue
            if section is None or not line or line.startswith("$"):
                continue
            if section == "shell":
                shell_indices.append(source_index)
            else:
                solid_indices.append(source_index)
            source_index += 1

    return np.asarray(shell_indices + solid_indices, dtype=np.int64), source_index


def k_to_unstructured_grid(input_file: PathInput) -> pv.UnstructuredGrid:
    """Read the main LS-DYNA model and return an indexed UnstructuredGrid."""
    input_path = _validate_mesh_path(input_file)
    deck = Deck(input_path)
    grid = deck.to_grid()
    if not isinstance(grid, pv.UnstructuredGrid):
        raise TypeError(
            "LS-DYNA conversion returned "
            f"{type(grid).__name__}, not an UnstructuredGrid"
        )
    if grid.n_points == 0 or grid.n_cells == 0:
        raise ValueError(f"LS-DYNA mesh is empty: {input_path}")

    _copy_source_element_ids(deck, grid)
    _copy_normalised_ids(grid)

    # ``original_index`` is the stable zero-based index of the freshly
    # converted PyVista cell. ``source_index`` additionally records where the
    # element appeared in the .k file, since Deck.to_grid places shells first.
    grid.cell_data["original_index"] = np.arange(grid.n_cells, dtype=np.int64)
    source_indices, source_count = _source_element_indices(input_path)
    if source_indices.size == grid.n_cells and source_count == grid.n_cells:
        grid.cell_data["source_index"] = source_indices
    else:
        warnings.warn(
            "Could not align source-file element order with the converted "
            "grid; source_index will use the PyVista cell order",
            stacklevel=2,
        )
        grid.cell_data["source_index"] = np.arange(grid.n_cells, dtype=np.int64)

    grid.cell_data["vtk_cell_type"] = np.asarray(grid.celltypes, dtype=np.uint8)
    grid.field_data["source_mesh"] = np.asarray([str(input_path)])
    return grid


def _iter_keyword_cards(
    path: Path,
) -> Iterable[tuple[str, tuple[str, ...]]]:
    """Yield non-comment lines grouped by LS-DYNA keyword."""
    keyword: str | None = None
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if stripped.startswith("*"):
                if keyword is not None:
                    yield keyword, tuple(lines)
                keyword = stripped.split(",", maxsplit=1)[0].upper()
                lines = []
            elif (
                keyword is not None
                and _is_metadata_keyword(keyword)
                and stripped
                and not stripped.startswith("$")
            ):
                lines.append(raw_line.rstrip("\r\n"))
    if keyword is not None:
        yield keyword, tuple(lines)


def _numeric_fields(line: str) -> list[str] | None:
    """Split one LS-DYNA numeric row in fixed-width or delimited form."""
    if "," in line:
        fields = [field.strip() for field in line.split(",")]
    else:
        fixed = [line[index : index + 10].strip() for index in range(0, len(line), 10)]
        fields = fixed
        try:
            int(float(fields[0].replace("D", "E").replace("d", "e")))
        except (IndexError, ValueError):
            fields = line.split()

    if not fields or not fields[0]:
        return None
    try:
        int(float(fields[0].replace("D", "E").replace("d", "e")))
    except ValueError:
        return None
    return fields


def _integer_field(fields: Sequence[str], index: int) -> int | None:
    if index >= len(fields) or not fields[index]:
        return None
    try:
        value = float(fields[index].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    if not np.isfinite(value) or value != np.floor(value):
        return None
    return int(value)


def _card_title(lines: Sequence[str]) -> str:
    for line in lines:
        if _numeric_fields(line) is None:
            return line.strip()
    return ""


def _card_numeric_rows(lines: Sequence[str]) -> list[list[str]]:
    return [fields for line in lines if (fields := _numeric_fields(line)) is not None]


def _new_metadata_dictionary() -> dict[str, Any]:
    return {
        "files": [],
        "card_counts": {},
        "parts": {},
        "materials": {},
        "sections": {},
        "hourglasses": {},
        "part_sets": {},
    }


def _is_metadata_keyword(keyword: str) -> bool:
    """Return whether card contents are useful to the metadata parser."""
    return (
        keyword == "*PART"
        or keyword.startswith("*PART_")
        or keyword.startswith("*MAT_")
        or keyword.startswith("*SECTION_")
        or keyword.startswith("*HOURGLASS")
        or keyword.startswith("*SET_PART_LIST")
    )


def analyze_auxiliary_files(files: Iterable[PathInput]) -> dict[str, Any]:
    """Analyze companion LS-DYNA files and return ID-keyed dictionaries.

    Recognized definitions are ``*PART``, ``*MAT_*``, ``*SECTION_*``,
    ``*HOURGLASS``, and ``*SET_PART_LIST*``.  The dictionary is intentionally
    composed only of JSON-serializable values so it can also be embedded in
    the output VTU as field metadata.
    """
    metadata = _new_metadata_dictionary()
    counts: Counter[str] = Counter()
    seen_paths: set[Path] = set()

    for value in files:
        path = _existing_file(value, "Auxiliary")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        metadata["files"].append(str(path))

        for keyword, lines in _iter_keyword_cards(path):
            counts[keyword] += 1
            if not _is_metadata_keyword(keyword):
                continue
            rows = _card_numeric_rows(lines)
            if not rows:
                continue
            title = _card_title(lines)
            source = str(path)

            if keyword == "*PART" or keyword.startswith("*PART_"):
                fields = rows[0]
                pid = _integer_field(fields, 0)
                if pid is None:
                    continue
                names = (
                    "part_id",
                    "section_id",
                    "material_id",
                    "eos_id",
                    "hourglass_id",
                    "gravity",
                    "adaptive_option",
                    "thermal_material_id",
                )
                record = {
                    name: _integer_field(fields, index)
                    for index, name in enumerate(names)
                }
                record.update({"name": title, "keyword": keyword, "source": source})
                metadata["parts"][pid] = record
                continue

            if keyword.startswith("*MAT_"):
                material_id = _integer_field(rows[0], 0)
                if material_id is not None:
                    metadata["materials"][material_id] = {
                        "material_id": material_id,
                        "name": title,
                        "keyword": keyword,
                        "source": source,
                    }
                continue

            if keyword.startswith("*SECTION_"):
                section_id = _integer_field(rows[0], 0)
                if section_id is not None:
                    metadata["sections"][section_id] = {
                        "section_id": section_id,
                        "name": title,
                        "keyword": keyword,
                        "source": source,
                    }
                continue

            if keyword.startswith("*HOURGLASS"):
                hourglass_id = _integer_field(rows[0], 0)
                if hourglass_id is not None:
                    metadata["hourglasses"][hourglass_id] = {
                        "hourglass_id": hourglass_id,
                        "keyword": keyword,
                        "source": source,
                    }
                continue

            if keyword.startswith("*SET_PART_LIST"):
                set_id = _integer_field(rows[0], 0)
                if set_id is None:
                    continue
                part_ids = [
                    part_id
                    for row in rows[1:]
                    for index in range(len(row))
                    if (part_id := _integer_field(row, index)) not in (None, 0)
                ]
                metadata["part_sets"][set_id] = {
                    "set_id": set_id,
                    "name": title,
                    "part_ids": part_ids,
                    "keyword": keyword,
                    "source": source,
                }

    metadata["card_counts"] = dict(sorted(counts.items()))
    return metadata


def _discover_auxiliary_files(mesh_path: Path) -> list[Path]:
    """Find conventional companion keyword files in common project folders."""
    discovered: list[Path] = []
    search_directories = [mesh_path.parent]

    # A generated subject mesh normally lives in ``<subject>/output`` while
    # its visualization wrapper and part list live in ``<subject>/visualise``.
    subject_directory = mesh_path.parent.parent
    for name in ("visualise", "simulation"):
        candidate = subject_directory / name
        if candidate.is_dir():
            search_directories.append(candidate)

    # ReCoDE keeps reusable material/part/set definitions here.  Looking for
    # that exact relative directory keeps discovery useful but well scoped.
    for ancestor in mesh_path.parents:
        candidate = ancestor / "src" / "dependencies" / "simulation"
        if candidate.is_dir():
            search_directories.append(candidate)
            break

    for directory in search_directories:
        for pattern in ("part*.k", "material*.k", "set*.k"):
            for candidate in sorted(directory.glob(pattern)):
                resolved = candidate.resolve()
                if (
                    resolved != mesh_path
                    and candidate.is_file()
                    and resolved not in discovered
                ):
                    discovered.append(resolved)
    return discovered


def _tissue_attributes(part_id: int) -> tuple[int, str]:
    for index, (name, part_ids) in enumerate(_TISSUE_PIDS):
        if part_id in part_ids:
            return index, name
    return len(_TISSUE_PIDS), "Other"


def _safe_array_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").casefold()
    return result or "unnamed"


def add_pid_attributes(
    grid: pv.UnstructuredGrid,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Add commonly used part/material/tissue attributes to every cell."""
    if "part_id" not in grid.cell_data:
        raise ValueError("The converted grid does not contain part_id cell data")

    metadata_dict = _new_metadata_dictionary() if metadata is None else metadata
    parts = metadata_dict.get("parts", {})
    materials = metadata_dict.get("materials", {})
    sections = metadata_dict.get("sections", {})
    part_ids = np.asarray(grid.cell_data["part_id"], dtype=np.int64)

    section_ids = np.full(grid.n_cells, -1, dtype=np.int32)
    material_ids = np.full(grid.n_cells, -1, dtype=np.int32)
    hourglass_ids = np.full(grid.n_cells, -1, dtype=np.int32)
    tissue_ids = np.empty(grid.n_cells, dtype=np.int16)
    part_names: list[str] = []
    material_names: list[str] = []
    material_keywords: list[str] = []
    section_names: list[str] = []
    section_keywords: list[str] = []
    tissue_names: list[str] = []
    mre134_unified_labels: list[str] = []

    for cell_index, raw_pid in enumerate(part_ids):
        pid = int(raw_pid)
        part = parts.get(pid, {})
        section_id = part.get("section_id")
        material_id = part.get("material_id")
        hourglass_id = part.get("hourglass_id")
        if section_id is not None:
            section_ids[cell_index] = int(section_id)
        if material_id is not None:
            material_ids[cell_index] = int(material_id)
        if hourglass_id is not None:
            hourglass_ids[cell_index] = int(hourglass_id)

        part_names.append(
            str(part.get("name") or _FALLBACK_PART_NAMES.get(pid, f"PID {pid}"))
        )
        material = materials.get(material_id, {})
        material_names.append(str(material.get("name") or "Unknown"))
        material_keywords.append(str(material.get("keyword") or ""))
        section = sections.get(section_id, {})
        section_names.append(str(section.get("name") or "Unknown"))
        section_keywords.append(str(section.get("keyword") or ""))
        tissue_id, tissue_name = _tissue_attributes(pid)
        tissue_ids[cell_index] = tissue_id
        tissue_names.append(tissue_name)
        mre134_unified_labels.append(
            unified_label_for_recode_part(pid) or "Unmapped MRE134 ROI"
        )

    grid.cell_data["part_name"] = np.asarray(part_names)
    grid.cell_data["section_id"] = section_ids
    grid.cell_data["section_name"] = np.asarray(section_names)
    grid.cell_data["section_keyword"] = np.asarray(section_keywords)
    grid.cell_data["material_id"] = material_ids
    grid.cell_data["material_name"] = np.asarray(material_names)
    grid.cell_data["material_keyword"] = np.asarray(material_keywords)
    grid.cell_data["hourglass_id"] = hourglass_ids
    grid.cell_data["tissue_id"] = tissue_ids
    grid.cell_data["tissue_name"] = np.asarray(tissue_names)
    grid.cell_data["mre134_unified_label"] = np.asarray(mre134_unified_labels)

    for raw_set_id, part_set in metadata_dict.get("part_sets", {}).items():
        set_id = int(raw_set_id)
        set_name = str(part_set.get("name") or f"set_{set_id}")
        array_name = f"part_set_{set_id}_{_safe_array_name(set_name)}"
        members = np.asarray(part_set.get("part_ids", ()), dtype=np.int64)
        grid.cell_data[array_name] = np.isin(part_ids, members).astype(np.uint8)

    # Lookup tables and the complete dictionaries remain available without
    # repeating them on every cell.
    grid.field_data["tissue_names"] = np.asarray(_TISSUE_NAMES)
    grid.field_data["lsdyna_metadata_json"] = np.asarray(
        [json.dumps(metadata_dict, sort_keys=True)]
    )
    grid.field_data["mre134_recode_label_mapping_json"] = np.asarray(
        [json.dumps(mre134_recode_mapping_document(), sort_keys=True)]
    )


def _first_content_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "$")):
                return stripped
    raise ValueError(f"Result file is empty: {path}")


def _records_to_dictionary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, NDArray[Any]]:
    if not records:
        raise ValueError("JSON result record list is empty")
    keys = list(records[0])
    if any(set(record) != set(keys) for record in records):
        raise ValueError("Every JSON result record must contain the same keys")
    return {key: np.asarray([record[key] for record in records]) for key in keys}


def _normalise_data_dictionary(values: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {}
    row_count: int | None = None
    for raw_name, raw_values in values.items():
        name = str(raw_name)
        array = np.asarray(raw_values)
        if array.ndim == 0:
            array = array.reshape(1)
        if row_count is None:
            row_count = int(array.shape[0])
        elif array.shape[0] != row_count:
            raise ValueError(
                f"Result column {name!r} has {array.shape[0]} rows; "
                f"expected {row_count}"
            )
        arrays[name] = array
    if not arrays:
        raise ValueError("Result dictionary is empty")
    return arrays


def load_file_as_dictionary(path: PathInput) -> dict[str, NDArray[Any]]:
    """Load a tabular result file into a dictionary of NumPy arrays."""
    source = _existing_file(path, "Result")
    suffix = source.suffix.casefold()

    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            return _normalise_data_dictionary(
                {name: archive[name] for name in archive.files}
            )

    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        if values.dtype.names:
            return _normalise_data_dictionary(
                {name: values[name] for name in values.dtype.names}
            )
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(
                "Plain NPY results must be a two-dimensional array whose "
                "first column is element_id"
            )
        names = ["element_id"] + (
            ["strain"]
            if values.shape[1] == 2
            else [f"strain_{index}" for index in range(values.shape[1] - 1)]
        )
        return _normalise_data_dictionary(
            {name: values[:, index] for index, name in enumerate(names)}
        )

    if suffix == ".json":
        with source.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
        if isinstance(payload, list) and all(
            isinstance(item, Mapping) for item in payload
        ):
            return _normalise_data_dictionary(_records_to_dictionary(payload))
        if isinstance(payload, Mapping):
            return _normalise_data_dictionary(payload)
        raise ValueError(
            "JSON results must be an object of arrays or a list of records"
        )

    if suffix not in {".csv", ".txt", ".dat"}:
        raise ValueError(
            "Result files must use .csv, .txt, .dat, .json, .npy, or .npz"
        )

    first_line = _first_content_line(source)
    delimiter = "," if "," in first_line else None
    first_token = (
        first_line.split(",", maxsplit=1)[0]
        if delimiter
        else first_line.split()[0]
    )
    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        table = np.genfromtxt(
            source,
            delimiter=delimiter,
            names=True,
            comments="#",
            encoding="utf-8-sig",
            dtype=None,
        )
        names = table.dtype.names or ()
        if not names:
            raise ValueError(f"Could not read result headers from: {source}")
        table = np.atleast_1d(table)
        return _normalise_data_dictionary({name: table[name] for name in names})

    table = np.loadtxt(source, delimiter=delimiter, comments="#", ndmin=2)
    if table.shape[1] < 2:
        raise ValueError(
            "Headerless results need at least element_id and one value column"
        )
    names = ["element_id"] + (
        ["strain"]
        if table.shape[1] == 2
        else [f"strain_{index}" for index in range(table.shape[1] - 1)]
    )
    return _normalise_data_dictionary(
        {name: table[:, index] for index, name in enumerate(names)}
    )


def _integer_array(values: Any, name: str) -> NDArray[np.int64]:
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Result column {name!r} must contain integer IDs") from exc
    if numeric.ndim != 1:
        raise ValueError(f"Result ID column {name!r} must be one-dimensional")
    if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"Result column {name!r} must contain finite integer IDs")
    return numeric.astype(np.int64)


def _result_value_keys(
    data: Mapping[str, NDArray[Any]],
    excluded: set[str],
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        keys: list[str] = []
        for value in requested:
            match = _find_mapping_key(data, value, ())
            if match is None:
                available = ", ".join(data)
                raise ValueError(
                    f"Result field {value!r} was not found. Available: {available}"
                )
            keys.append(match)
        return keys

    keys = []
    for name, values in data.items():
        if name in excluded:
            continue
        if np.asarray(values).dtype.kind in "biufc":
            keys.append(name)
    if not keys:
        raise ValueError("The result file contains no numeric value columns")
    return keys


def map_results_by_element_id(
    grid: pv.UnstructuredGrid,
    result_data: Mapping[str, Any],
    *,
    element_id_key: str | None = None,
    part_id_key: str | None = None,
    value_keys: Sequence[str] | None = None,
    target: ResultTarget = "solid",
) -> list[str]:
    """Rearrange result rows into mesh cell order using source element IDs.

    Missing mesh results are stored as NaN.  If a ``time`` column is present
    and element IDs repeat, each selected scalar is stored as an
    ``(n_cells, n_times)`` cell array and the time vector is stored in field
    data.  When a result contains ``part_id``, the join uses ``(part_id,
    element_id)`` to disambiguate repeated IDs.
    """
    if target not in {"solid", "shell", "all"}:
        raise ValueError("target must be 'solid', 'shell', or 'all'")
    if "element_id" not in grid.cell_data:
        raise ValueError("The grid does not contain element_id cell data")

    data = _normalise_data_dictionary(result_data)
    eid_key = _find_mapping_key(data, element_id_key, _ELEMENT_ID_KEYS)
    if eid_key is None:
        available = ", ".join(data)
        raise ValueError(
            "Results need an element_id/eid column. "
            f"Available columns: {available}"
        )
    pid_key = _find_mapping_key(data, part_id_key, _PART_ID_KEYS)
    time_key = _find_mapping_key(data, None, _TIME_KEYS)

    result_eids = _integer_array(data[eid_key], eid_key)
    result_pids = (
        _integer_array(data[pid_key], pid_key) if pid_key is not None else None
    )
    if result_pids is not None and result_pids.size != result_eids.size:
        raise ValueError("Result part_id and element_id columns have different lengths")

    dimensions = np.asarray(
        grid.cell_data.get("element_dimension", np.full(grid.n_cells, 3)),
        dtype=np.int8,
    )
    if target == "solid":
        candidates = np.flatnonzero(dimensions == 3)
    elif target == "shell":
        candidates = np.flatnonzero(dimensions == 2)
    else:
        candidates = np.arange(grid.n_cells, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError(f"The grid contains no {target} result target cells")

    mesh_eids = np.asarray(grid.cell_data["element_id"], dtype=np.int64)
    mesh_pids = np.asarray(
        grid.cell_data.get("part_id", np.zeros(grid.n_cells)),
        dtype=np.int64,
    )
    use_part_id = result_pids is not None
    lookup: dict[int | tuple[int, int], int] = {}
    for cell_index in candidates:
        key: int | tuple[int, int]
        if use_part_id:
            key = (int(mesh_pids[cell_index]), int(mesh_eids[cell_index]))
        else:
            key = int(mesh_eids[cell_index])
        if key in lookup:
            hint = " Include part_id in the result file." if not use_part_id else ""
            raise ValueError(
                f"Element key {key!r} is not unique among {target} cells.{hint}"
            )
        lookup[key] = int(cell_index)

    row_cells = np.full(result_eids.size, -1, dtype=np.int64)
    for row_index, eid in enumerate(result_eids):
        key = (
            (int(result_pids[row_index]), int(eid))
            if use_part_id and result_pids is not None
            else int(eid)
        )
        row_cells[row_index] = lookup.get(key, -1)
    matched = row_cells >= 0
    if not matched.any():
        raise ValueError("None of the result element IDs match the selected mesh cells")
    if not matched.all():
        warnings.warn(
            f"Ignored {np.count_nonzero(~matched)} result row(s) whose element "
            f"IDs were not found among {target} cells",
            stacklevel=2,
        )

    excluded = {eid_key}
    if pid_key is not None:
        excluded.add(pid_key)
    if time_key is not None:
        excluded.add(time_key)
    selected_values = _result_value_keys(data, excluded, value_keys)

    keys_for_duplicates = [
        (
            (int(result_pids[index]), int(eid))
            if use_part_id and result_pids is not None
            else int(eid)
        )
        for index, eid in enumerate(result_eids)
        if matched[index]
    ]
    has_repeated_elements = len(keys_for_duplicates) != len(set(keys_for_duplicates))
    mapped_names: list[str] = []

    if has_repeated_elements:
        if time_key is None:
            raise ValueError(
                "Result element IDs repeat; provide part_id to disambiguate "
                "parts or a time column for long-form temporal results"
            )
        times = np.asarray(data[time_key], dtype=np.float64)
        if times.ndim != 1 or times.size != result_eids.size:
            raise ValueError("The result time column must have one value per row")
        if not np.isfinite(times).all():
            raise ValueError("Result times must be finite")
        unique_times = np.unique(times)
        time_indices = np.searchsorted(unique_times, times)
        occupied: set[tuple[int, int]] = set()
        for row_index in np.flatnonzero(matched):
            location = (int(row_cells[row_index]), int(time_indices[row_index]))
            if location in occupied:
                raise ValueError(
                    "Results contain duplicate rows for one element and time"
                )
            occupied.add(location)

        for name in selected_values:
            values = np.asarray(data[name], dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(
                    f"Long-form temporal field {name!r} must be one-dimensional"
                )
            mapped = np.full((grid.n_cells, unique_times.size), np.nan)
            mapped[row_cells[matched], time_indices[matched]] = values[matched]
            grid.cell_data[name] = mapped
            mapped_names.append(name)
        grid.field_data["time"] = unique_times
    else:
        for name in selected_values:
            values = np.asarray(data[name])
            if values.shape[0] != result_eids.size:
                raise ValueError(
                    f"Result field {name!r} does not align with element_id"
                )
            try:
                numeric = values.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Result field {name!r} must be numeric") from exc
            mapped_shape = (grid.n_cells,) + numeric.shape[1:]
            mapped = np.full(mapped_shape, np.nan, dtype=np.float64)
            mapped[row_cells[matched]] = numeric[matched]
            grid.cell_data[name] = mapped
            mapped_names.append(name)

    available = np.zeros(grid.n_cells, dtype=np.uint8)
    available[row_cells[matched]] = 1
    grid.cell_data["result_available"] = available
    grid.field_data["result_target"] = np.asarray([target])
    grid.field_data["result_fields"] = np.asarray(mapped_names)
    return mapped_names


def build_unified_model(
    input_file: PathInput,
    *,
    metadata_files: Iterable[PathInput] | None = None,
    result_file: PathInput | None = None,
    element_id_key: str | None = None,
    part_id_key: str | None = None,
    result_fields: Sequence[str] | None = None,
    result_target: ResultTarget = "solid",
    auto_metadata: bool = True,
) -> pv.UnstructuredGrid:
    """Build one mesh containing geometry, PID metadata, and strain/results."""
    input_path = _validate_mesh_path(input_file)
    grid = k_to_unstructured_grid(input_path)

    companion_files: list[PathInput] = [input_path]
    if auto_metadata:
        companion_files.extend(_discover_auxiliary_files(input_path))
    if metadata_files is not None:
        companion_files.extend(metadata_files)
    metadata = analyze_auxiliary_files(companion_files)
    add_pid_attributes(grid, metadata)

    if result_file is not None:
        result_path = _existing_file(result_file, "Result")
        result_data = load_file_as_dictionary(result_path)
        map_results_by_element_id(
            grid,
            result_data,
            element_id_key=element_id_key,
            part_id_key=part_id_key,
            value_keys=result_fields,
            target=result_target,
        )
        grid.field_data["result_source"] = np.asarray([str(result_path)])
    return grid


def convert_k_to_vtu(
    input_file: PathInput,
    output_file: PathInput | None = None,
    *,
    metadata_files: Iterable[PathInput] | None = None,
    result_file: PathInput | None = None,
    strain_file: PathInput | None = None,
    element_id_key: str | None = None,
    part_id_key: str | None = None,
    result_fields: Sequence[str] | None = None,
    result_target: ResultTarget = "solid",
    auto_metadata: bool = True,
    binary: bool = True,
) -> Path:
    """Build and save one analysis-ready ``.vtu`` model.

    ``strain_file`` is a readable alias for ``result_file``.  Use only one.
    Existing two-argument calls remain valid and now also receive source
    indices and PID-derived attributes.
    """
    input_path = _validate_mesh_path(input_file)
    if result_file is not None and strain_file is not None:
        raise ValueError("Use either result_file or strain_file, not both")
    selected_result = result_file if result_file is not None else strain_file

    output_path = (
        input_path.with_suffix(".vtu")
        if output_file is None
        else Path(output_file).expanduser().resolve()
    )
    if output_path.suffix.casefold() != ".vtu":
        raise ValueError(f"VTU output must use the .vtu extension: {output_path}")
    if output_path == input_path:
        raise ValueError("Input and output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = build_unified_model(
        input_path,
        metadata_files=metadata_files,
        result_file=selected_result,
        element_id_key=element_id_key,
        part_id_key=part_id_key,
        result_fields=result_fields,
        result_target=result_target,
        auto_metadata=auto_metadata,
    )
    grid.save(output_path, binary=binary)
    return output_path


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Build one VTU from an LS-DYNA mesh, companion keyword metadata, "
            "and element-ID-indexed strain/results."
        )
    )
    parser.add_argument("input_file", type=Path, help="main LS-DYNA .k mesh")
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        help="output .vtu file (default: beside the input mesh)",
    )
    parser.add_argument(
        "--metadata-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "companion LS-DYNA part/material/set file; repeat for multiple "
            "files"
        ),
    )
    parser.add_argument(
        "--no-auto-metadata",
        action="store_true",
        help="do not discover part*, material*, and set*.k beside the mesh",
    )
    parser.add_argument(
        "--result-file",
        "--strain-file",
        dest="result_file",
        type=Path,
        help="CSV/TXT/JSON/NPY/NPZ results containing element_id",
    )
    parser.add_argument(
        "--element-id-key",
        help="result column containing element IDs (default: auto-detect)",
    )
    parser.add_argument(
        "--part-id-key",
        help="optional result column containing part IDs (default: auto-detect)",
    )
    parser.add_argument(
        "--result-field",
        action="append",
        default=[],
        help="result column to add; repeat as needed (default: all numeric fields)",
    )
    parser.add_argument(
        "--result-target",
        choices=("solid", "shell", "all"),
        default="solid",
        help="element family receiving results (default: solid)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="write ASCII instead of compact binary data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args: Any = _build_argument_parser().parse_args(argv)
    output_path = convert_k_to_vtu(
        args.input_file,
        args.output_file,
        metadata_files=args.metadata_file,
        result_file=args.result_file,
        element_id_key=args.element_id_key,
        part_id_key=args.part_id_key,
        result_fields=args.result_field or None,
        result_target=args.result_target,
        auto_metadata=not args.no_auto_metadata,
        binary=not args.ascii,
    )
    mesh = pv.read(output_path)
    result_names = list(np.asarray(mesh.field_data.get("result_fields", ()), dtype=str))
    result_text = f", results: {', '.join(result_names)}" if result_names else ""
    print(
        f"Created {output_path} from {args.input_file} "
        f"({mesh.n_points} points, {mesh.n_cells} cells{result_text})"
    )
    return 0


__all__ = [
    "add_pid_attributes",
    "analyze_auxiliary_files",
    "build_unified_model",
    "convert_k_to_vtu",
    "k_to_unstructured_grid",
    "load_file_as_dictionary",
    "main",
    "map_results_by_element_id",
]


if __name__ == "__main__":
    raise SystemExit(main())
