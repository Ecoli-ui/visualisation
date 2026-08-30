"""Adapter for the DRYAD20210528 brain-motion atlas VTK sequences.

The release stores head rotation (``HR``) and neck extension (``NE``) as ten
separate legacy VTK files.  Each file is a complete, deformed tetrahedral mesh
for one time point, so treating the cell arrays as frames on one static mesh
would discard the measured motion.  This module discovers a sequence from any
one of its files, validates the shared topology and array schema, and returns
all changing mesh snapshots with their 18 ms time spacing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

PathInput = str | PathLike[str]
DryadCondition = Literal["HR", "NE"]

DRYAD_FRAME_INTERVAL_SECONDS = 0.018
DRYAD_DEFAULT_FIELD = "GmaxT2"
DRYAD_POINT_SCALARS = ("T1", "T1_std", "disp_std")
DRYAD_POINT_VECTORS = ("disp",)
DRYAD_CELL_SCALARS = (
    "GmaxT1",
    "GmaxT2",
    "GmaxT2_std",
    "GmaxT3",
    "GmaxT3_std",
    "E1",
)
DRYAD_CELL_VECTORS = ("V1",)

_FRAME_NAME = re.compile(r"^(HR|NE)_MESH_(\d+)\.vtk$", re.IGNORECASE)
_CONDITION_NAMES: dict[DryadCondition, str] = {
    "HR": "head rotation",
    "NE": "neck extension",
}


class DryadAdapterError(ValueError):
    """Raised when a file resembles Dryad data but its sequence is invalid."""


@dataclass(frozen=True, slots=True)
class DryadFrame:
    """One indexed VTK snapshot discovered in a Dryad sequence."""

    index: int
    path: Path


@dataclass(slots=True)
class DryadSequence:
    """Validated time-ordered Dryad mesh snapshots and dataset metadata."""

    condition: DryadCondition
    frame_records: tuple[DryadFrame, ...]
    frames: tuple[pv.UnstructuredGrid, ...]
    times: NDArray[np.float64]
    metadata: dict[str, Any]

    @property
    def mesh(self) -> pv.UnstructuredGrid:
        """Return the baseline mesh snapshot."""
        return self.frames[0]

    def scalar_series(
        self, field_name: str = DRYAD_DEFAULT_FIELD
    ) -> NDArray[np.float64]:
        """Stack one scalar cell field into ``(n_frames, n_cells)`` order."""
        if field_name not in DRYAD_CELL_SCALARS:
            available = ", ".join(DRYAD_CELL_SCALARS)
            raise DryadAdapterError(
                f"{field_name!r} is not a Dryad scalar strain field; "
                f"choose one of: {available}"
            )
        return np.stack(
            [
                np.asarray(frame.cell_data[field_name], dtype=np.float64)
                for frame in self.frames
            ],
            axis=0,
        )

    def as_loaded_data(self) -> Any:
        """Return the repository's generic ``LoadedData`` representation."""
        # Imported lazily to keep the adapter usable without a module cycle.
        from ..io.loader import LoadedData

        return LoadedData(
            mesh=self.mesh,
            time=self.times.copy(),
            metadata=dict(self.metadata),
            frames=self.frames,
        )


def _frame_identity(path: Path) -> tuple[DryadCondition, int] | None:
    match = _FRAME_NAME.fullmatch(path.name)
    if match is None:
        return None
    condition = match.group(1).upper()
    assert condition in _CONDITION_NAMES
    return condition, int(match.group(2))  # type: ignore[return-value]


def is_dryad_frame_path(path: PathInput) -> bool:
    """Return whether a path has the Dryad ``HR/NE_MESH_n.vtk`` naming form."""
    return _frame_identity(Path(path)) is not None


def discover_dryad_frames(
    path: PathInput,
) -> tuple[DryadCondition, tuple[DryadFrame, ...]]:
    """Discover and naturally order the complete sibling sequence for ``path``."""
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Dryad frame file does not exist: {selected}")

    identity = _frame_identity(selected)
    if identity is None:
        raise DryadAdapterError(
            "Dryad frame names must match HR_MESH_<number>.vtk or "
            f"NE_MESH_<number>.vtk: {selected.name}"
        )
    condition, _ = identity

    indexed: dict[int, Path] = {}
    for candidate in selected.parent.iterdir():
        candidate_identity = _frame_identity(candidate)
        if candidate_identity is None or candidate_identity[0] != condition:
            continue
        frame_index = candidate_identity[1]
        if frame_index < 1:
            raise DryadAdapterError(
                f"Dryad frame indices must start at 1: {candidate.name}"
            )
        previous = indexed.get(frame_index)
        if previous is not None and previous != candidate:
            raise DryadAdapterError(
                f"Duplicate Dryad frame index {frame_index}: "
                f"{previous.name}, {candidate.name}"
            )
        indexed[frame_index] = candidate.resolve()

    indices = sorted(indexed)
    if not indices:
        raise DryadAdapterError(f"No {condition} Dryad frames found beside {selected}")
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        missing = ", ".join(map(str, sorted(set(expected) - set(indices))))
        raise DryadAdapterError(
            f"{condition} Dryad sequence is incomplete; missing frame(s): {missing}"
        )

    return condition, tuple(DryadFrame(index, indexed[index]) for index in indices)


def dryad_sequence_metadata(path: PathInput) -> dict[str, Any]:
    """Build lightweight, JSON-compatible metadata without reading mesh arrays."""
    condition, records = discover_dryad_frames(path)
    return {
        "adapter": "dryad",
        "dataset": "DRYAD20210528",
        "condition": condition,
        "condition_name": _CONDITION_NAMES[condition],
        "frame_interval_seconds": DRYAD_FRAME_INTERVAL_SECONDS,
        "frame_indices": [record.index for record in records],
        "frame_paths": [str(record.path) for record in records],
        "default_field": DRYAD_DEFAULT_FIELD,
        "cell_scalar_fields": list(DRYAD_CELL_SCALARS),
        "cell_vector_fields": list(DRYAD_CELL_VECTORS),
        "point_scalar_fields": list(DRYAD_POINT_SCALARS),
        "point_vector_fields": list(DRYAD_POINT_VECTORS),
        "time_origin": "frame 1 baseline",
        "time_units": "s",
    }


def _validate_array_shape(
    mesh: pv.UnstructuredGrid,
    name: str,
    *,
    association: Literal["point", "cell"],
    components: int,
    source: Path,
) -> None:
    data = mesh.point_data if association == "point" else mesh.cell_data
    if name not in data:
        raise DryadAdapterError(
            f"Dryad frame {source.name} is missing {association} array {name!r}"
        )
    values = np.asarray(data[name])
    count = mesh.n_points if association == "point" else mesh.n_cells
    expected = (count,) if components == 1 else (count, components)
    if values.shape != expected:
        raise DryadAdapterError(
            f"Dryad {association} array {name!r} in {source.name} has shape "
            f"{values.shape}; expected {expected}"
        )


def _validate_frame_schema(mesh: pv.UnstructuredGrid, source: Path) -> None:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise DryadAdapterError(f"Dryad frame is empty: {source}")
    if not np.all(np.asarray(mesh.celltypes) == int(pv.CellType.TETRA)):
        raise DryadAdapterError(
            f"Dryad frame must contain only tetrahedral cells: {source.name}"
        )
    for name in DRYAD_POINT_SCALARS:
        _validate_array_shape(
            mesh, name, association="point", components=1, source=source
        )
    for name in DRYAD_POINT_VECTORS:
        _validate_array_shape(
            mesh, name, association="point", components=3, source=source
        )
    for name in DRYAD_CELL_SCALARS:
        _validate_array_shape(
            mesh, name, association="cell", components=1, source=source
        )
    for name in DRYAD_CELL_VECTORS:
        _validate_array_shape(
            mesh, name, association="cell", components=3, source=source
        )


def load_dryad_sequence(path: PathInput) -> DryadSequence:
    """Read and validate every HR or NE frame discovered beside ``path``."""
    condition, records = discover_dryad_frames(path)
    loaded: list[pv.UnstructuredGrid] = []

    for record in records:
        try:
            data = pv.read(record.path)
        except Exception as exc:
            raise DryadAdapterError(
                f"Could not read Dryad frame: {record.path}"
            ) from exc
        if not isinstance(data, pv.UnstructuredGrid):
            raise DryadAdapterError(
                f"Dryad frame {record.path.name} must be an UnstructuredGrid, "
                f"not {type(data).__name__}"
            )
        _validate_frame_schema(data, record.path)
        data.set_active_scalars(DRYAD_DEFAULT_FIELD, preference="cell")
        loaded.append(data)

    baseline = loaded[0]
    for record, frame in zip(records[1:], loaded[1:], strict=True):
        if frame.n_points != baseline.n_points or frame.n_cells != baseline.n_cells:
            raise DryadAdapterError(
                f"Dryad topology size changes at {record.path.name}: "
                f"{frame.n_points} points/{frame.n_cells} cells; expected "
                f"{baseline.n_points} points/{baseline.n_cells} cells"
            )
        if not np.array_equal(frame.cells, baseline.cells):
            raise DryadAdapterError(
                f"Dryad cell connectivity changes at {record.path.name}"
            )
        if not np.array_equal(frame.celltypes, baseline.celltypes):
            raise DryadAdapterError(f"Dryad cell types change at {record.path.name}")

    times = np.arange(len(records), dtype=np.float64) * DRYAD_FRAME_INTERVAL_SECONDS
    metadata = dryad_sequence_metadata(records[0].path)
    return DryadSequence(
        condition=condition,
        frame_records=records,
        frames=tuple(loaded),
        times=times,
        metadata=metadata,
    )


__all__ = [
    "DRYAD_CELL_SCALARS",
    "DRYAD_CELL_VECTORS",
    "DRYAD_DEFAULT_FIELD",
    "DRYAD_FRAME_INTERVAL_SECONDS",
    "DRYAD_POINT_SCALARS",
    "DRYAD_POINT_VECTORS",
    "DryadAdapterError",
    "DryadCondition",
    "DryadFrame",
    "DryadSequence",
    "discover_dryad_frames",
    "dryad_sequence_metadata",
    "is_dryad_frame_path",
    "load_dryad_sequence",
]
