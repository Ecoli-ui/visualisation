"""Inspect mesh data, detect hotspots, trace them over time, and annotate plots.

The functions in this module deliberately operate on ordinary PyVista
``DataSet`` objects.  They do not require a particular brain mesh or a
particular scalar-array name, and point and cell data are both supported.

Typical use
-----------
Import a mesh, call ``detect_hotspots(mesh, "strain")``, add the mesh to a
``pyvista.Plotter``, and pass the result to ``add_hotspot_annotations``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike, NDArray

from .io.loader import DataLoadError, load_mesh

Association: TypeAlias = Literal["point", "cell", "field"]
AssociationPreference: TypeAlias = Literal["auto", "point", "cell", "field"]
HotspotMode: TypeAlias = Literal["high", "low", "absolute"]


class AnalysisError(ValueError):
    """Raised when mesh data cannot be analysed as requested."""


class DataType(StrEnum):
    """Semantic type of a mesh-data array."""

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    MULTI_COMPONENT = "multi-component"
    TEXT = "text"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


# ``DataKind`` is kept as a readable alternative for callers that prefer that
# terminology.
DataKind = DataType


@dataclass(frozen=True, slots=True)
class ArrayInfo:
    """Description of one point-, cell-, or field-data array."""

    name: str
    association: Association
    data_type: DataType
    dtype: str
    shape: tuple[int, ...]
    components: int
    finite_count: int | None
    missing_count: int
    minimum: float | None = None
    maximum: float | None = None
    unique_count: int | None = None

    @property
    def kind(self) -> DataType:
        """Alias for :attr:`data_type`."""
        return self.data_type

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "association": self.association,
            "data_type": self.data_type.value,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "components": self.components,
            "finite_count": self.finite_count,
            "missing_count": self.missing_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unique_count": self.unique_count,
        }


@dataclass(frozen=True, slots=True)
class Hotspot:
    """One spatial hotspot detected in one frame."""

    index: int
    position: tuple[float, float, float]
    value: float
    score: float
    array_name: str
    association: Literal["point", "cell"]
    frame_index: int = 0
    time: float | None = None

    @property
    def point(self) -> NDArray[np.float64]:
        """Position as a NumPy array."""
        return np.asarray(self.position, dtype=np.float64)


@dataclass(slots=True)
class HotspotTrace:
    """A hotspot linked across one or more frames."""

    trace_id: int
    detections: list[Hotspot] = field(default_factory=list)

    @property
    def hotspots(self) -> tuple[Hotspot, ...]:
        """Immutable view of the detections."""
        return tuple(self.detections)

    @property
    def positions(self) -> NDArray[np.float64]:
        """An ``(n, 3)`` array containing the trace path."""
        if not self.detections:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray([item.position for item in self.detections])

    @property
    def values(self) -> NDArray[np.float64]:
        """Detected values along the trace."""
        return np.asarray([item.value for item in self.detections])

    @property
    def times(self) -> NDArray[np.float64]:
        """Detection times, using frame indices when explicit times are absent."""
        return np.asarray(
            [
                item.time if item.time is not None else float(item.frame_index)
                for item in self.detections
            ],
            dtype=np.float64,
        )

    @property
    def start_frame(self) -> int:
        if not self.detections:
            raise AnalysisError("An empty hotspot trace has no start frame")
        return self.detections[0].frame_index

    @property
    def end_frame(self) -> int:
        if not self.detections:
            raise AnalysisError("An empty hotspot trace has no end frame")
        return self.detections[-1].frame_index

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "trace_id": self.trace_id,
            "detections": [
                {
                    "frame_index": item.frame_index,
                    "time": item.time,
                    "index": item.index,
                    "position": list(item.position),
                    "value": item.value,
                    "score": item.score,
                    "array_name": item.array_name,
                    "association": item.association,
                }
                for item in self.detections
            ],
        }


@dataclass(frozen=True, slots=True)
class GlobalPeak:
    """Location and time of the largest finite value in a scalar series."""

    value: float
    time_index: int
    element_index: int
    time: float
    position: tuple[float, ...] | None = None

    @property
    def peak_value(self) -> float:
        """Alias for :attr:`value`."""
        return self.value

    @property
    def peak_time(self) -> float:
        """Alias for :attr:`time`."""
        return self.time

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "value": self.value,
            "time_index": self.time_index,
            "element_index": self.element_index,
            "time": self.time,
            "position": list(self.position) if self.position is not None else None,
        }


# A shorter name is convenient and maintains compatibility with code that calls
# the result a track rather than a trace.
HotspotTrack = HotspotTrace


def _normalise_values(values: ArrayLike) -> NDArray[Any]:
    array = np.asanyarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _component_count(array: NDArray[Any]) -> int:
    if array.ndim <= 1:
        return 1
    return int(np.prod(array.shape[1:]))


def _is_categorical_numeric(
    array: NDArray[Any], categorical_threshold: int
) -> bool:
    if array.dtype.kind not in "iu":
        return False
    if array.size == 0:
        return False
    unique = np.unique(array)
    return unique.size <= categorical_threshold


def detect_data_type(
    values: ArrayLike,
    *,
    categorical: bool | Literal["auto"] = "auto",
    categorical_threshold: int = 20,
) -> DataType:
    """Infer the semantic type of an array.

    Shape determines whether numeric data is scalar, vector, tensor, or a
    generic multi-component array.  In ``"auto"`` mode, integer scalar arrays
    with at most ``categorical_threshold`` unique values are treated as
    categorical.  Floating-point IDs can be forced to categorical with
    ``categorical=True``.
    """
    if categorical not in (True, False, "auto"):
        raise ValueError("categorical must be True, False, or 'auto'")
    if categorical_threshold < 1:
        raise ValueError("categorical_threshold must be at least 1")

    array = _normalise_values(values)
    kind = array.dtype.kind
    components = _component_count(array)

    if kind == "b":
        return DataType.BOOLEAN
    if kind in "SU":
        if categorical is True:
            return DataType.CATEGORICAL
        if categorical == "auto" and array.size:
            return (
                DataType.CATEGORICAL
                if np.unique(array).size <= categorical_threshold
                else DataType.TEXT
            )
        return DataType.TEXT
    if kind == "O":
        return DataType.CATEGORICAL if categorical is True else DataType.UNKNOWN
    if kind in "mM":
        return DataType.TEMPORAL
    if kind not in "iufc":
        return DataType.UNKNOWN

    if components == 1:
        if categorical is True or (
            categorical == "auto"
            and _is_categorical_numeric(array, categorical_threshold)
        ):
            return DataType.CATEGORICAL
        return DataType.SCALAR
    if components in (2, 3):
        return DataType.VECTOR
    if array.ndim >= 3 or components in (4, 6, 9):
        return DataType.TENSOR
    return DataType.MULTI_COMPONENT


def describe_array(
    name: str,
    values: ArrayLike,
    association: Association,
    *,
    categorical: bool | Literal["auto"] = "auto",
    categorical_threshold: int = 20,
) -> ArrayInfo:
    """Build an :class:`ArrayInfo` record for an array."""
    if association not in ("point", "cell", "field"):
        raise ValueError(f"Unknown association: {association!r}")

    array = _normalise_values(values)
    data_type = detect_data_type(
        array,
        categorical=categorical,
        categorical_threshold=categorical_threshold,
    )
    numeric = array.dtype.kind in "iufc"
    finite_count: int | None = None
    missing_count = 0
    minimum: float | None = None
    maximum: float | None = None

    if numeric:
        if array.dtype.kind == "c":
            finite = np.isfinite(array.real) & np.isfinite(array.imag)
            comparable = np.abs(array)
        else:
            finite = np.isfinite(array)
            comparable = array
        finite_count = int(np.count_nonzero(finite))
        missing_count = int(array.size - finite_count)
        if finite_count:
            finite_values = comparable[finite]
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
    elif array.dtype.kind in "mM":
        missing = np.isnat(array)
        missing_count = int(np.count_nonzero(missing))

    unique_count: int | None
    try:
        unique_count = int(np.unique(array).size)
    except (TypeError, ValueError):
        unique_count = None

    return ArrayInfo(
        name=name,
        association=association,
        data_type=data_type,
        dtype=str(array.dtype),
        shape=tuple(int(item) for item in array.shape),
        components=_component_count(array),
        finite_count=finite_count,
        missing_count=missing_count,
        minimum=minimum,
        maximum=maximum,
        unique_count=unique_count,
    )


def inspect_mesh_data(
    mesh: pv.DataSet,
    *,
    categorical_threshold: int = 20,
    include_field_data: bool = True,
) -> list[ArrayInfo]:
    """Describe all data arrays attached to a mesh."""
    if not isinstance(mesh, pv.DataSet):
        raise TypeError("inspect_mesh_data requires a PyVista DataSet")

    collections: list[tuple[Association, Any]] = [
        ("point", mesh.point_data),
        ("cell", mesh.cell_data),
    ]
    if include_field_data:
        collections.append(("field", mesh.field_data))

    result: list[ArrayInfo] = []
    for association, data in collections:
        for name in data.keys():
            result.append(
                describe_array(
                    str(name),
                    data[name],
                    association,
                    categorical_threshold=categorical_threshold,
                )
            )
    return result


# Descriptive aliases for interactive use.
detect_mesh_data_types = inspect_mesh_data
inspect_arrays = inspect_mesh_data


def _data_collection(mesh: pv.DataSet, association: Association) -> Any:
    return {
        "point": mesh.point_data,
        "cell": mesh.cell_data,
        "field": mesh.field_data,
    }[association]


def resolve_array(
    mesh: pv.DataSet,
    name: str,
    association: AssociationPreference = "auto",
) -> tuple[NDArray[Any], Association]:
    """Resolve an array name and return its values and association.

    Automatic resolution raises an error when the same name exists in more
    than one association; callers must then choose explicitly.
    """
    if not isinstance(mesh, pv.DataSet):
        raise TypeError("resolve_array requires a PyVista DataSet")
    if association not in ("auto", "point", "cell", "field"):
        raise ValueError(f"Unknown association preference: {association!r}")

    associations: tuple[Association, ...] = (
        ("point", "cell", "field")
        if association == "auto"
        else (association,)
    )
    matches = [
        item
        for item in associations
        if name in _data_collection(mesh, item).keys()
    ]
    if not matches:
        available = ", ".join(
            f"{item.association}:{item.name}" for item in inspect_mesh_data(mesh)
        )
        raise AnalysisError(
            f"Array {name!r} was not found"
            + (f" in {association} data" if association != "auto" else "")
            + f". Available arrays: {available or '<none>'}"
        )
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise AnalysisError(
            f"Array {name!r} is ambiguous ({joined}); specify association"
        )
    selected = matches[0]
    return np.asanyarray(_data_collection(mesh, selected)[name]), selected


def scalar_values(values: ArrayLike, component: int | None = None) -> NDArray:
    """Convert numeric mesh data to one scalar per point or cell.

    Scalar arrays are returned unchanged.  For multi-component data, an
    explicit component may be selected; otherwise the Euclidean/Frobenius
    magnitude is used.
    """
    array = _normalise_values(values)
    if array.dtype.kind not in "iufc":
        raise AnalysisError(
            f"Hotspot analysis requires numeric data, received {array.dtype}"
        )

    if array.ndim == 1:
        if component not in (None, 0):
            raise AnalysisError("A scalar array only has component 0")
        result = array
    else:
        flattened = array.reshape(array.shape[0], -1)
        if component is None:
            result = np.linalg.norm(flattened, axis=1)
        else:
            if component < 0 or component >= flattened.shape[1]:
                raise AnalysisError(
                    f"Component {component} is outside the valid range "
                    f"0..{flattened.shape[1] - 1}"
                )
            result = flattened[:, component]

    # Complex scalar values are ranked by magnitude, consistent with the
    # multi-component case.
    if np.iscomplexobj(result):
        result = np.abs(result)
    return np.asarray(result)


def extract_element_history(
    scalar_series: ArrayLike,
    element_index: int,
) -> NDArray[np.float64]:
    """Extract one element's values from a ``(n_times, n_elements)`` series.

    A copy is returned so callers may transform the history without modifying
    the source array.
    """
    try:
        series = np.asarray(scalar_series, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("scalar_series must contain numeric values") from exc
    if series.ndim != 2:
        raise AnalysisError(
            "scalar_series must have shape (n_times, n_elements)"
        )
    if series.shape[0] == 0 or series.shape[1] == 0:
        raise AnalysisError("scalar_series cannot be empty")
    if isinstance(element_index, bool) or not isinstance(
        element_index, (int, np.integer)
    ):
        raise TypeError("element_index must be an integer")
    index = int(element_index)
    if index < 0 or index >= series.shape[1]:
        raise IndexError(
            f"element_index must be in 0..{series.shape[1] - 1}"
        )
    return series[:, index].copy()


def find_global_peak(
    scalar_series: ArrayLike,
    times: ArrayLike,
    element_positions: ArrayLike | None = None,
) -> GlobalPeak:
    """Find the largest finite value in a time-by-element scalar series.

    Ties are resolved deterministically in row-major order: the earliest time
    is selected first, then the lowest element index.
    """
    try:
        series = np.asarray(scalar_series, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("scalar_series must contain numeric values") from exc
    if series.ndim != 2:
        raise AnalysisError(
            "scalar_series must have shape (n_times, n_elements)"
        )
    if series.shape[0] == 0 or series.shape[1] == 0:
        raise AnalysisError("scalar_series cannot be empty")

    try:
        time_values = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("times must contain numeric values") from exc
    if time_values.ndim != 1 or time_values.size != series.shape[0]:
        raise AnalysisError(
            f"times must contain exactly {series.shape[0]} values"
        )
    if not np.isfinite(time_values).all():
        raise AnalysisError("times cannot contain NaN or infinity")
    if np.any(np.diff(time_values) < 0):
        raise AnalysisError("times must be ordered from low to high")

    finite = np.isfinite(series)
    if not finite.any():
        raise AnalysisError("scalar_series contains no finite values")
    comparable = np.where(finite, series, -np.inf)
    flat_index = int(np.argmax(comparable))
    time_index, element_index = np.unravel_index(flat_index, series.shape)

    position: tuple[float, ...] | None = None
    if element_positions is not None:
        try:
            positions = np.asarray(element_positions, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AnalysisError(
                "element_positions must contain numeric coordinates"
            ) from exc
        if positions.ndim != 2 or positions.shape[0] != series.shape[1]:
            raise AnalysisError(
                "element_positions must have shape (n_elements, n_dimensions)"
            )
        if positions.shape[1] == 0:
            raise AnalysisError(
                "element_positions must contain at least one coordinate"
            )
        selected_position = positions[element_index]
        if not np.isfinite(selected_position).all():
            raise AnalysisError("The global peak position is not finite")
        position = tuple(float(item) for item in selected_position)

    return GlobalPeak(
        value=float(series[time_index, element_index]),
        time_index=int(time_index),
        element_index=int(element_index),
        time=float(time_values[time_index]),
        position=position,
    )


def _spatial_values(
    mesh: pv.DataSet,
    array_name: str,
    association: AssociationPreference,
    component: int | None,
) -> tuple[NDArray, NDArray[np.float64], Literal["point", "cell"]]:
    values, resolved = resolve_array(mesh, array_name, association)
    if resolved == "field":
        raise AnalysisError("Field data has no point/cell position for hotspots")
    scalars = scalar_values(values, component)
    positions = (
        np.asarray(mesh.points, dtype=np.float64)
        if resolved == "point"
        else np.asarray(mesh.cell_centers().points, dtype=np.float64)
    )
    if scalars.shape[0] != positions.shape[0]:
        raise AnalysisError(
            f"{resolved} array {array_name!r} has {scalars.shape[0]} values, "
            f"but the mesh has {positions.shape[0]} {resolved}s"
        )
    return scalars, positions, resolved


def _validate_hotspot_options(
    *,
    mode: HotspotMode,
    percentile: float,
    max_hotspots: int | None,
    min_distance: float,
) -> None:
    if mode not in ("high", "low", "absolute"):
        raise ValueError("mode must be 'high', 'low', or 'absolute'")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if max_hotspots is not None and max_hotspots < 1:
        raise ValueError("max_hotspots must be positive or None")
    if min_distance < 0 or not np.isfinite(min_distance):
        raise ValueError("min_distance must be finite and non-negative")


def detect_hotspots(
    mesh: pv.DataSet,
    scalars: str,
    *,
    association: AssociationPreference = "auto",
    component: int | None = None,
    mode: HotspotMode = "high",
    threshold: float | None = None,
    percentile: float = 95.0,
    max_hotspots: int | None = 10,
    min_distance: float = 0.0,
    frame_index: int = 0,
    time: float | None = None,
) -> list[Hotspot]:
    """Detect spatial hotspots using thresholding and non-maximum suppression.

    ``mode`` selects high values, low values, or large absolute values.
    ``threshold`` is interpreted in the original value domain; when omitted it
    is calculated from ``percentile`` (the complementary percentile is used
    for low-valued hotspots).  Candidates are ordered by extremeness, then
    greedily filtered so no two returned positions are closer than
    ``min_distance``.
    """
    _validate_hotspot_options(
        mode=mode,
        percentile=percentile,
        max_hotspots=max_hotspots,
        min_distance=min_distance,
    )
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if time is not None and not np.isfinite(time):
        raise ValueError("time must be finite")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    values, positions, resolved = _spatial_values(
        mesh, scalars, association, component
    )
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return []

    if mode == "high":
        scores = values
        cutoff = (
            float(threshold)
            if threshold is not None
            else float(np.percentile(values[finite], percentile))
        )
        candidate_mask = finite & (values >= cutoff)
    elif mode == "low":
        scores = -values
        cutoff = (
            float(threshold)
            if threshold is not None
            else float(np.percentile(values[finite], 100.0 - percentile))
        )
        candidate_mask = finite & (values <= cutoff)
    else:
        scores = np.abs(values)
        cutoff = (
            abs(float(threshold))
            if threshold is not None
            else float(np.percentile(scores[finite], percentile))
        )
        candidate_mask = finite & (scores >= cutoff)

    candidate_indices = np.flatnonzero(candidate_mask)
    # Stable sorting makes equal-valued plateaus deterministic.
    order = np.argsort(-scores[candidate_indices], kind="stable")
    ordered_indices = candidate_indices[order]

    selected: list[Hotspot] = []
    selected_positions: list[NDArray[np.float64]] = []
    for index_value in ordered_indices:
        index = int(index_value)
        position = positions[index]
        if min_distance and any(
            np.linalg.norm(position - previous) < min_distance
            for previous in selected_positions
        ):
            continue
        selected.append(
            Hotspot(
                index=index,
                position=tuple(float(item) for item in position),
                value=float(values[index]),
                score=float(scores[index]),
                array_name=scalars,
                association=resolved,
                frame_index=frame_index,
                time=float(time) if time is not None else None,
            )
        )
        selected_positions.append(position)
        if max_hotspots is not None and len(selected) >= max_hotspots:
            break
    return selected


def _normalise_frames_and_scalars(
    frames: pv.DataSet | Sequence[pv.DataSet],
    scalars: str | Sequence[str],
) -> tuple[list[pv.DataSet], list[str]]:
    if isinstance(frames, pv.DataSet):
        scalar_names = [scalars] if isinstance(scalars, str) else list(scalars)
        if not scalar_names:
            raise ValueError("At least one scalar array name is required")
        return [frames] * len(scalar_names), scalar_names

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("At least one frame is required")
    if not all(isinstance(item, pv.DataSet) for item in frame_list):
        raise TypeError("Every frame must be a PyVista DataSet")
    scalar_names = (
        [scalars] * len(frame_list)
        if isinstance(scalars, str)
        else list(scalars)
    )
    if len(scalar_names) != len(frame_list):
        raise ValueError(
            "Provide one scalar name for all frames or one name per frame"
        )
    return frame_list, scalar_names


def _normalise_times(
    times: Sequence[float] | NDArray | None, frame_count: int
) -> list[float | None]:
    if times is None:
        return [None] * frame_count
    result = np.asarray(times, dtype=np.float64)
    if result.ndim != 1 or result.size != frame_count:
        raise ValueError(f"times must contain exactly {frame_count} values")
    if not np.isfinite(result).all():
        raise ValueError("times cannot contain NaN or infinity")
    if np.any(np.diff(result) < 0):
        raise ValueError("times must be ordered from low to high")
    return [float(item) for item in result]


def _default_trace_distance(frames: Sequence[pv.DataSet]) -> float:
    diagonals: list[float] = []
    for mesh in frames:
        bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
        diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
        if np.isfinite(diagonal) and diagonal > 0:
            diagonals.append(diagonal)
    return 0.05 * max(diagonals) if diagonals else 0.0


def trace_hotspots(
    frames: pv.DataSet | Sequence[pv.DataSet],
    scalars: str | Sequence[str],
    *,
    times: Sequence[float] | NDArray | None = None,
    association: AssociationPreference = "auto",
    component: int | None = None,
    mode: HotspotMode = "high",
    threshold: float | None = None,
    percentile: float = 95.0,
    max_hotspots: int | None = 10,
    min_distance: float = 0.0,
    max_link_distance: float | None = None,
    max_frame_gap: int = 0,
    minimum_trace_length: int = 1,
) -> list[HotspotTrace]:
    """Detect and link hotspots through a time-ordered mesh sequence.

    A single mesh plus a sequence of scalar names is useful when each time
    step is stored as a separate array.  A mesh sequence plus one scalar name
    handles one mesh per time step.  Detections are linked greedily by spatial
    distance, with each detection and trace used at most once per frame.
    """
    frame_list, scalar_names = _normalise_frames_and_scalars(frames, scalars)
    time_values = _normalise_times(times, len(frame_list))
    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be non-negative")
    if minimum_trace_length < 1:
        raise ValueError("minimum_trace_length must be at least 1")
    if max_link_distance is None:
        max_link_distance = _default_trace_distance(frame_list)
    if max_link_distance < 0 or not np.isfinite(max_link_distance):
        raise ValueError("max_link_distance must be finite and non-negative")

    traces: list[HotspotTrace] = []
    next_trace_id = 0

    for frame_index, (mesh, array_name, time) in enumerate(
        zip(frame_list, scalar_names, time_values, strict=True)
    ):
        detections = detect_hotspots(
            mesh,
            array_name,
            association=association,
            component=component,
            mode=mode,
            threshold=threshold,
            percentile=percentile,
            max_hotspots=max_hotspots,
            min_distance=min_distance,
            frame_index=frame_index,
            time=time,
        )

        active = [
            trace
            for trace in traces
            if trace.detections
            and frame_index - trace.detections[-1].frame_index
            <= max_frame_gap + 1
        ]
        possible_links: list[tuple[float, int, int]] = []
        for trace_index, trace in enumerate(active):
            previous = trace.detections[-1].point
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(previous - detection.point))
                if distance <= max_link_distance:
                    possible_links.append(
                        (distance, trace_index, detection_index)
                    )
        possible_links.sort(key=lambda item: (item[0], item[1], item[2]))

        used_traces: set[int] = set()
        used_detections: set[int] = set()
        for _, trace_index, detection_index in possible_links:
            if (
                trace_index in used_traces
                or detection_index in used_detections
            ):
                continue
            active[trace_index].detections.append(detections[detection_index])
            used_traces.add(trace_index)
            used_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in used_detections:
                traces.append(
                    HotspotTrace(
                        trace_id=next_trace_id, detections=[detection]
                    )
                )
                next_trace_id += 1

    return [
        trace
        for trace in traces
        if len(trace.detections) >= minimum_trace_length
    ]


def hotspot_label(
    hotspot: Hotspot,
    *,
    precision: int = 4,
    include_time: bool = False,
    prefix: str | None = None,
) -> str:
    """Format a concise label for a hotspot annotation."""
    if precision < 0:
        raise ValueError("precision must be non-negative")
    parts = [prefix] if prefix else []
    parts.append(f"{hotspot.array_name}={hotspot.value:.{precision}g}")
    if include_time:
        time = (
            hotspot.time
            if hotspot.time is not None
            else float(hotspot.frame_index)
        )
        parts.append(f"t={time:.{precision}g}")
    return " ".join(parts)


def add_hotspot_annotations(
    plotter: pv.Plotter,
    hotspots: Iterable[Hotspot],
    *,
    color: str = "red",
    point_size: float = 14.0,
    show_labels: bool = True,
    label_precision: int = 4,
    include_time: bool = False,
    label_kwargs: dict[str, Any] | None = None,
    point_kwargs: dict[str, Any] | None = None,
) -> Any | None:
    """Add hotspot markers and optional labels to a PyVista plotter.

    The point actor is returned.  An empty hotspot iterable is a no-op and
    returns ``None``.
    """
    items = list(hotspots)
    if not items:
        return None
    points = np.asarray([item.position for item in items], dtype=np.float64)
    marker_options: dict[str, Any] = {
        "color": color,
        "point_size": point_size,
        "render_points_as_spheres": True,
    }
    if point_kwargs:
        marker_options.update(point_kwargs)
    actor = plotter.add_points(points, **marker_options)

    if show_labels:
        labels = [
            hotspot_label(
                item,
                precision=label_precision,
                include_time=include_time,
            )
            for item in items
        ]
        options: dict[str, Any] = {
            "text_color": color,
            "font_size": 12,
            "point_size": 0,
            "shape_opacity": 0.55,
            "always_visible": True,
        }
        if label_kwargs:
            options.update(label_kwargs)
        plotter.add_point_labels(points, labels, **options)
    return actor


def add_trace_annotations(
    plotter: pv.Plotter,
    traces: Iterable[HotspotTrace],
    *,
    color: str = "yellow",
    line_width: float = 4.0,
    point_size: float = 9.0,
    show_trace_ids: bool = True,
    line_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Draw hotspot paths and their detections on a PyVista plotter."""
    actors: list[Any] = []
    for trace in traces:
        if not trace.detections:
            continue
        points = trace.positions
        if len(points) >= 2:
            options: dict[str, Any] = {
                "color": color,
                "line_width": line_width,
            }
            if line_kwargs:
                options.update(line_kwargs)
            actors.append(
                plotter.add_mesh(pv.lines_from_points(points), **options)
            )
        actors.append(
            plotter.add_points(
                points,
                color=color,
                point_size=point_size,
                render_points_as_spheres=True,
            )
        )
        if show_trace_ids:
            options = {
                "text_color": color,
                "font_size": 12,
                "point_size": 0,
                "shape_opacity": 0.55,
                "always_visible": True,
            }
            if label_kwargs:
                options.update(label_kwargs)
            plotter.add_point_labels(
                points[-1:].copy(), [f"trace {trace.trace_id}"], **options
            )
    return actors


def format_array_summary(info: ArrayInfo) -> str:
    """Format one :class:`ArrayInfo` as a human-readable line."""
    summary = (
        f"{info.association}:{info.name} — {info.data_type.value}, "
        f"dtype={info.dtype}, shape={info.shape}"
    )
    if info.minimum is not None and info.maximum is not None:
        summary += f", range=[{info.minimum:.6g}, {info.maximum:.6g}]"
    if info.missing_count:
        summary += f", missing={info.missing_count}"
    return summary


def add_data_annotation(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    *,
    names: Sequence[str] | None = None,
    position: str = "upper_left",
    font_size: int = 10,
    categorical_threshold: int = 20,
    **text_kwargs: Any,
) -> str:
    """Add a compact data-type summary to a plot and return its text."""
    infos = inspect_mesh_data(
        mesh, categorical_threshold=categorical_threshold
    )
    if names is not None:
        selected = set(names)
        infos = [item for item in infos if item.name in selected]
    text = "\n".join(format_array_summary(item) for item in infos)
    if not text:
        text = "No mesh data arrays"
    plotter.add_text(
        text,
        position=position,
        font_size=font_size,
        **text_kwargs,
    )
    return text


# Short verb-style aliases make the annotation API easy to discover.
annotate_hotspots = add_hotspot_annotations
annotate_traces = add_trace_annotations
annotate_data = add_data_annotation


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Inspect a mesh and optionally visualise scalar hotspots."
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help="mesh readable by PyVista or an LS-DYNA .k mesh",
    )
    parser.add_argument(
        "--scalars", help="point or cell array to analyse for hotspots"
    )
    parser.add_argument(
        "--association",
        choices=("auto", "point", "cell"),
        default="auto",
    )
    parser.add_argument(
        "--mode", choices=("high", "low", "absolute"), default="high"
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--max-hotspots", type=int, default=10)
    parser.add_argument("--min-distance", type=float, default=0.0)
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an annotated PyVista window",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="save an off-screen annotated image",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    try:
        mesh = load_mesh(args.mesh)
        if not isinstance(mesh, pv.DataSet):
            raise AnalysisError(
                f"{args.mesh} contains {type(mesh).__name__}, not one DataSet"
            )
        infos = inspect_mesh_data(mesh)
        for info in infos:
            print(format_array_summary(info))

        hotspots: list[Hotspot] = []
        if args.scalars:
            hotspots = detect_hotspots(
                mesh,
                args.scalars,
                association=args.association,
                mode=args.mode,
                threshold=args.threshold,
                percentile=args.percentile,
                max_hotspots=args.max_hotspots,
                min_distance=args.min_distance,
            )
            for number, hotspot in enumerate(hotspots, start=1):
                print(
                    f"hotspot {number}: index={hotspot.index}, "
                    f"value={hotspot.value:.6g}, position={hotspot.position}"
                )

        if args.show or args.screenshot:
            plotter = pv.Plotter(off_screen=bool(args.screenshot and not args.show))
            mesh_options: dict[str, Any] = {"show_edges": False}
            if args.scalars:
                mesh_options["scalars"] = args.scalars
                if args.association != "auto":
                    mesh_options["preference"] = args.association
            plotter.add_mesh(mesh, **mesh_options)
            add_hotspot_annotations(plotter, hotspots)
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                plotter.show(
                    screenshot=str(args.screenshot),
                    auto_close=not args.show,
                )
            else:
                plotter.show()
    except (DataLoadError, OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"analysis: error: {exc}") from exc
    return 0


__all__ = [
    "AnalysisError",
    "ArrayInfo",
    "DataKind",
    "DataType",
    "GlobalPeak",
    "Hotspot",
    "HotspotTrace",
    "HotspotTrack",
    "add_data_annotation",
    "add_hotspot_annotations",
    "add_trace_annotations",
    "annotate_data",
    "annotate_hotspots",
    "annotate_traces",
    "describe_array",
    "detect_data_type",
    "detect_hotspots",
    "detect_mesh_data_types",
    "extract_element_history",
    "find_global_peak",
    "format_array_summary",
    "hotspot_label",
    "inspect_arrays",
    "inspect_mesh_data",
    "main",
    "resolve_array",
    "scalar_values",
    "trace_hotspots",
]


if __name__ == "__main__":
    raise SystemExit(main())
