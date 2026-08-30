"""Prepare loaded mesh data for the viewer's scalar display paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv

from .loader import LoadedData


@dataclass(frozen=True, slots=True)
class PreparedDisplayData:
    """Validated mesh, scalar series, time data, and metadata for one case."""

    mesh: pv.DataSet
    frames: tuple[pv.DataSet, ...]
    metadata: dict[str, Any]
    field_name: str
    scalar_series: npt.NDArray[np.float64] | None
    times: npt.NDArray[np.float64] | None
    source_times: npt.NDArray[np.float64] | None


def resolve_viewer_field(loaded: LoadedData, requested: str) -> str:
    """Use an adapter-provided default when the generic MPS field is absent."""
    mesh = loaded.mesh
    if not isinstance(mesh, pv.DataSet) or requested in mesh.cell_data:
        return requested
    default_field = loaded.metadata.get("default_field")
    if (
        requested == "MPS"
        and loaded.metadata.get("adapter") == "dryad"
        and isinstance(default_field, str)
        and default_field in mesh.cell_data
    ):
        return default_field
    return requested


def load_scalar_series(
    path: str | Path,
    *,
    key: str | None = None,
) -> npt.NDArray[np.float64]:
    """Load a two-dimensional scalar series from NPY or NPZ."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scalar-series file does not exist: {source}")
    try:
        if source.suffix.casefold() == ".npy":
            values = np.load(source, allow_pickle=False)
        elif source.suffix.casefold() == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                if key is None:
                    if len(archive.files) != 1:
                        available = ", ".join(archive.files) or "<none>"
                        raise ValueError(
                            f"{source} contains multiple arrays ({available}); "
                            "select one with --series-key"
                        )
                    selected = archive.files[0]
                else:
                    selected = next(
                        (
                            name
                            for name in archive.files
                            if name.casefold() == key.casefold()
                        ),
                        "",
                    )
                    if not selected:
                        available = ", ".join(archive.files) or "<none>"
                        raise ValueError(
                            f"Series key {key!r} was not found. "
                            f"Available arrays: {available}"
                        )
                values = archive[selected]
        else:
            raise ValueError("Scalar series must use .npy or .npz")
    except OSError as exc:
        raise ValueError(f"Could not load scalar series: {source}") from exc
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("Scalar series must have shape (n_times, n_cells)")
    return result


def series_from_cell_data(
    mesh: pv.DataSet,
    field_name: str,
) -> npt.NDArray[np.float64] | None:
    """Interpret one cell array as a time-by-cell scalar series."""
    if field_name not in mesh.cell_data:
        return None
    values = np.asarray(mesh.cell_data[field_name], dtype=np.float64)
    if values.ndim == 1 and values.size == mesh.n_cells:
        return values.reshape(1, mesh.n_cells)
    if values.ndim == 2:
        if values.shape[1] == mesh.n_cells:
            return values
        if values.shape[0] == mesh.n_cells:
            return values.T
    raise ValueError(
        f"Cell array {field_name!r} cannot be interpreted as scalar frames"
    )


def series_from_mesh_frames(
    frames: tuple[pv.DataSet, ...],
    field_name: str,
) -> npt.NDArray[np.float64] | None:
    """Build one time-by-cell scalar series from loaded mesh snapshots."""
    if not frames:
        return None
    if len(frames) == 1:
        return series_from_cell_data(frames[0], field_name)

    scalar_frames: list[npt.NDArray[np.float64]] = []
    for index, frame in enumerate(frames):
        if field_name not in frame.cell_data:
            if index == 0 and all(
                field_name not in item.cell_data for item in frames
            ):
                return None
            raise ValueError(
                f"Cell array {field_name!r} is missing from time frame "
                f"{index + 1}"
            )
        values = np.asarray(frame.cell_data[field_name], dtype=np.float64)
        if values.ndim == 2 and 1 in values.shape:
            values = values.reshape(-1)
        if values.ndim != 1 or values.size != frame.n_cells:
            raise ValueError(
                f"Cell array {field_name!r} in time frame {index + 1} "
                "must contain one scalar per cell"
            )
        scalar_frames.append(values)
    return np.stack(scalar_frames, axis=0)


def prepare_display_data(
    loaded: LoadedData,
    requested_field: str,
    *,
    scalar_series_path: str | Path | None = None,
    series_key: str | None = None,
    role: str = "Model",
    require_results: bool = False,
) -> PreparedDisplayData:
    """Prepare one loaded case through the viewer's shared validation path."""
    mesh = loaded.mesh
    if not isinstance(mesh, pv.DataSet):
        raise ValueError(f"Loaded data contains {type(mesh).__name__}, not one DataSet")

    frames = loaded.frames or (mesh,)
    field_name = resolve_viewer_field(loaded, requested_field)
    series = (
        load_scalar_series(scalar_series_path, key=series_key)
        if scalar_series_path is not None
        else series_from_mesh_frames(frames, field_name)
    )
    if series is not None and not np.isfinite(series).any():
        series = None
    if require_results and series is None:
        raise ValueError(
            f"{role} field {field_name!r} contains no finite cell-scalar result data"
        )

    source_times = (
        np.asarray(loaded.time, dtype=np.float64)
        if loaded.time is not None
        else None
    )
    times: npt.NDArray[np.float64] | None = None
    if series is not None:
        frame_count = int(series.shape[0])
        if source_times is not None:
            if source_times.size != frame_count:
                raise ValueError(
                    f"{role} time data contains {source_times.size} frame(s), but "
                    f"{field_name!r} contains {frame_count} frame(s)"
                )
            times = source_times
        else:
            times = np.arange(frame_count, dtype=np.float64)

    return PreparedDisplayData(
        mesh=mesh,
        frames=frames,
        metadata=loaded.metadata,
        field_name=field_name,
        scalar_series=series,
        times=times,
        source_times=source_times,
    )


__all__ = [
    "PreparedDisplayData",
    "load_scalar_series",
    "prepare_display_data",
    "resolve_viewer_field",
    "series_from_cell_data",
    "series_from_mesh_frames",
]
