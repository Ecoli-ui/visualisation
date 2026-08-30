"""Load a mesh, simulation time vector, and metadata.

The mesh can be any format supported by :func:`pyvista.read`, including the
VTK, STL, and PLY files in ``brain-meshing``. LS-DYNA ``.k`` meshes are
converted to temporary VTU files before PyVista reads them. Time vectors may
be stored as NumPy, JSON, CSV, or whitespace-delimited text files. Metadata
may be JSON or TOML.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from ..observation_case import ObservationCase
from .sources import (
    AdaptedSource,
    SourceAdapterError,
    load_adapted_source,
)

PathInput = str | PathLike[str]


class DataLoadError(ValueError):
    """Raised when an input file exists but cannot be loaded or validated."""


@dataclass(slots=True)
class LoadedData:
    """Container returned by :func:`load_data`.

    Attributes
    ----------
    mesh:
        The first mesh snapshot returned by PyVista.
    time:
        A one-dimensional array of time values, or ``None`` when no time file
        was supplied and neither the reader nor mesh provides time data.
    metadata:
        User metadata loaded from JSON or TOML.
    frames:
        Every mesh snapshot exposed by a temporal reader. Static files contain
        the single value stored in ``mesh``.
    adapter_name:
        Registered source adapter that performed the numerical-data read.
    """

    mesh: pv.DataObject
    time: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: tuple[pv.DataSet, ...] = field(default_factory=tuple)
    adapter_name: str | None = None


def _existing_file(path: PathInput, description: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {file_path}")
    return file_path


def _load_source(path: PathInput) -> AdaptedSource:
    """Translate adapter failures to the loader's stable public error type."""
    try:
        return load_adapted_source(path)
    except SourceAdapterError as exc:
        raise DataLoadError(str(exc)) from exc


def load_mesh(path: PathInput) -> pv.DataObject:
    """Read a source through its registered adapter and return its mesh."""
    return _load_source(path).mesh


def load_mesh_frames(
    path: PathInput,
) -> tuple[tuple[pv.DataSet, ...], NDArray[np.float64] | None]:
    """Read all snapshots and reader-provided times from a mesh file.

    Readers for temporal formats such as PVD, XDMF, EnSight, and Exodus expose
    a ``time_values`` sequence. Calling :func:`pyvista.read` alone selects only
    one of those values, so each time point is selected and copied here.
    """
    loaded = _load_source(path)
    return loaded.frames, loaded.time


def _select_mapping_value(
    values: dict[str, Any], key: str, source: Path
) -> Any:
    """Select a key case-insensitively from a mapping."""
    matching_key = next(
        (name for name in values if name.casefold() == key.casefold()), None
    )
    if matching_key is None:
        available = ", ".join(map(str, values)) or "<none>"
        raise DataLoadError(
            f"Time key {key!r} was not found in {source}. "
            f"Available keys: {available}"
        )
    return values[matching_key]


def _first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise DataLoadError(f"Time file is empty: {path}")


def _read_delimited_time(path: Path, key: str) -> NDArray[Any]:
    delimiter = "," if path.suffix.casefold() == ".csv" else None
    first_line = _first_data_line(path)
    first_token = first_line.split("," if delimiter else None, maxsplit=1)[0]

    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        table = np.genfromtxt(
            path,
            delimiter=delimiter,
            names=True,
            comments="#",
            encoding="utf-8-sig",
        )
        names = table.dtype.names or ()
        matching_name = next(
            (name for name in names if name.casefold() == key.casefold()), None
        )
        if matching_name is None:
            available = ", ".join(names) or "<none>"
            raise DataLoadError(
                f"Time column {key!r} was not found in {path}. "
                f"Available columns: {available}"
            )
        return np.asarray(table[matching_name])

    table = np.loadtxt(path, delimiter=delimiter, comments="#")
    if table.ndim == 2:
        if table.shape[1] != 1:
            raise DataLoadError(
                f"{path} has {table.shape[1]} columns but no header. "
                "Provide a one-column file or add a 'time' header."
            )
        table = table[:, 0]
    return np.asarray(table)


def _validate_time(values: Any, source: Path) -> NDArray[np.float64]:
    try:
        time = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataLoadError(f"Time values in {source} must be numeric") from exc

    # A single value may arrive from NumPy as a scalar.
    if time.ndim == 0:
        time = time.reshape(1)
    if time.ndim != 1:
        raise DataLoadError(
            f"Time values in {source} must be one-dimensional; "
            f"received shape {time.shape}"
        )
    if time.size == 0:
        raise DataLoadError(f"Time vector is empty: {source}")
    if not np.isfinite(time).all():
        raise DataLoadError(f"Time vector contains NaN or infinity: {source}")
    if np.any(np.diff(time) < 0):
        raise DataLoadError(f"Time values are not ordered from low to high: {source}")
    return time


def load_time(path: PathInput, *, key: str = "time") -> NDArray[np.float64]:
    """Read a numeric time vector.

    Supported formats are ``.npy``, ``.npz``, ``.json``, ``.csv``, ``.txt``,
    and ``.dat``. For NPZ/JSON files and CSV files with a header, ``key``
    selects the array or column to read.
    """
    time_path = _existing_file(path, "Time")
    suffix = time_path.suffix.casefold()

    try:
        if suffix == ".npy":
            values = np.load(time_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(time_path, allow_pickle=False) as archive:
                archive_key = next(
                    (
                        name
                        for name in archive.files
                        if name.casefold() == key.casefold()
                    ),
                    None,
                )
                if archive_key is None:
                    available = ", ".join(archive.files) or "<none>"
                    raise DataLoadError(
                        f"Time key {key!r} was not found in {time_path}. "
                        f"Available keys: {available}"
                    )
                values = archive[archive_key]
        elif suffix == ".json":
            with time_path.open("r", encoding="utf-8") as stream:
                content = json.load(stream)
            values = (
                _select_mapping_value(content, key, time_path)
                if isinstance(content, dict)
                else content
            )
        elif suffix in {".csv", ".txt", ".dat"}:
            values = _read_delimited_time(time_path, key)
        else:
            raise DataLoadError(
                f"Unsupported time format {suffix!r}. Expected one of "
                ".npy, .npz, .json, .csv, .txt, or .dat"
            )
    except DataLoadError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Could not read time file: {time_path}") from exc

    return _validate_time(values, time_path)


def load_metadata(path: PathInput) -> dict[str, Any]:
    """Read metadata from a JSON or TOML object."""
    metadata_path = _existing_file(path, "Metadata")
    suffix = metadata_path.suffix.casefold()

    try:
        with metadata_path.open("rb") as stream:
            if suffix == ".json":
                metadata = json.load(stream)
            elif suffix == ".toml":
                metadata = tomllib.load(stream)
            else:
                raise DataLoadError(
                    f"Unsupported metadata format {suffix!r}. "
                    "Expected .json or .toml"
                )
    except DataLoadError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Could not read metadata file: {metadata_path}") from exc

    if not isinstance(metadata, dict):
        raise DataLoadError(
            f"Metadata must contain a top-level object/table: {metadata_path}"
        )
    return metadata


def load_observation_case(path: PathInput) -> ObservationCase:
    """Read typed strain-research metadata from JSON or TOML.

    Use :func:`load_metadata` when the file is an arbitrary user dictionary;
    use this function when it follows the :class:`ObservationCase` model.
    """
    metadata_path = _existing_file(path, "Observation case")
    try:
        return ObservationCase.from_dict(load_metadata(metadata_path))
    except DataLoadError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise DataLoadError(
            f"Invalid ObservationCase metadata: {metadata_path}"
        ) from exc


def _embedded_time(mesh: pv.DataObject, key: str) -> NDArray[np.float64] | None:
    """Read a time array from VTK field data when one is present."""
    field_data = getattr(mesh, "field_data", None)
    if field_data is None:
        return None

    matching_key = next(
        (name for name in field_data.keys() if name.casefold() == key.casefold()),
        None,
    )
    if matching_key is None:
        return None
    return _validate_time(field_data[matching_key], Path(f"<mesh:{matching_key}>"))


def load_data(
    mesh_path: PathInput,
    time_path: PathInput | None = None,
    metadata_path: PathInput | None = None,
    *,
    time_key: str = "time",
) -> LoadedData:
    """Load all available inputs into one :class:`LoadedData` object.

    Temporal mesh readers are expanded into all available snapshots. If
    ``time_path`` is omitted, their time values take precedence over a matching
    array in the mesh's VTK field data. Missing time and metadata are
    represented by ``None`` and an empty dictionary, respectively.
    """
    adapted = _load_source(mesh_path)
    frames = adapted.frames
    mesh = frames[0]
    time = (
        load_time(time_path, key=time_key)
        if time_path is not None
        else (
            adapted.time
            if adapted.time is not None
            else _embedded_time(mesh, time_key)
        )
    )
    metadata = dict(adapted.metadata)
    if metadata_path is not None:
        metadata.update(load_metadata(metadata_path))
    return LoadedData(
        mesh=mesh,
        time=time,
        metadata=metadata,
        frames=frames,
        adapter_name=adapted.adapter_name,
    )


__all__ = [
    "DataLoadError",
    "LoadedData",
    "ObservationCase",
    "load_data",
    "load_mesh",
    "load_mesh_frames",
    "load_metadata",
    "load_observation_case",
    "load_time",
]
