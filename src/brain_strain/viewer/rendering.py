"""Interactive, frame-aware PyVista visualisation for simulation meshes.

``MeshVisualisation`` owns the plot state while keeping the analysis code in
``analysis.py`` independent of a rendering window.  It supports a sequence of
meshes with the same scalar name or one mesh plus an explicit array of scalar
frames.

Example
-------
Load a mesh and create a view:

    viewer = MeshVisualisation(mesh, "strain", scalar_frames=strain_over_time)
    viewer.initialize_user_view()
    viewer.show_slices()
    viewer.enable_cell_picking()
    viewer.show()
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike, NDArray

from ..analysis import (
    AnalysisError,
    DataType,
    Hotspot,
    detect_data_type,
    detect_hotspots,
    inspect_mesh_data,
    resolve_array,
    scalar_values,
)
from ..io.loader import DataLoadError, load_mesh

Association: TypeAlias = Literal["point", "cell"]
AssociationPreference: TypeAlias = Literal["auto", "point", "cell"]
PickCallback: TypeAlias = Callable[[NDArray[np.int64]], None]

_ORIGINAL_CELL_IDS = "__visualisation_original_cell_ids__"
_MAIN_ACTOR = "visualisation-main-mesh"
_SLICE_ACTOR_PREFIX = "visualisation-slice-"
_HOTSPOT_ACTOR = "visualisation-hotspot-cells"
_HIGHLIGHT_ACTOR = "visualisation-highlighted-cells"
_STATUS_ACTOR = "visualisation-status"


class VisualisationError(ValueError):
    """Raised when a requested view cannot be constructed."""


def _as_dataset_sequence(
    mesh: pv.DataSet | Sequence[pv.DataSet],
) -> list[pv.DataSet]:
    if isinstance(mesh, pv.DataSet):
        return [mesh]
    frames = list(mesh)
    if not frames:
        raise VisualisationError("At least one mesh is required")
    if not all(isinstance(item, pv.DataSet) for item in frames):
        raise TypeError("Every mesh frame must be a PyVista DataSet")
    return frames


def _normalise_times(
    times: Sequence[float] | ArrayLike | None, frame_count: int
) -> NDArray[np.float64] | None:
    if times is None:
        return None
    result = np.asarray(times, dtype=np.float64)
    if result.ndim != 1 or result.size != frame_count:
        raise VisualisationError(
            f"times must contain exactly {frame_count} values"
        )
    if not np.isfinite(result).all():
        raise VisualisationError("times cannot contain NaN or infinity")
    if np.any(np.diff(result) < 0):
        raise VisualisationError("times must be ordered from low to high")
    return result


def _automatic_scalar_name(mesh: pv.DataSet) -> str:
    active = mesh.active_scalars_name
    if active:
        try:
            values, association = resolve_array(mesh, active)
        except AnalysisError:
            pass
        else:
            if association in ("point", "cell") and values.dtype.kind in "iufc":
                return active

    supported = {
        DataType.BOOLEAN,
        DataType.CATEGORICAL,
        DataType.SCALAR,
        DataType.VECTOR,
        DataType.TENSOR,
        DataType.MULTI_COMPONENT,
    }
    match = next(
        (
            info
            for info in inspect_mesh_data(mesh, include_field_data=False)
            if info.data_type in supported
        ),
        None,
    )
    if match is None:
        raise VisualisationError(
            "The mesh has no numeric point or cell arrays to display"
        )
    return match.name


def _resolve_association_from_size(
    mesh: pv.DataSet,
    value_count: int,
    preference: AssociationPreference,
) -> Association:
    if preference == "point":
        if value_count != mesh.n_points:
            raise VisualisationError(
                f"Scalar frame has {value_count} values; expected "
                f"{mesh.n_points} point values"
            )
        return "point"
    if preference == "cell":
        if value_count != mesh.n_cells:
            raise VisualisationError(
                f"Scalar frame has {value_count} values; expected "
                f"{mesh.n_cells} cell values"
            )
        return "cell"

    matches: list[Association] = []
    if value_count == mesh.n_points:
        matches.append("point")
    if value_count == mesh.n_cells:
        matches.append("cell")
    if not matches:
        raise VisualisationError(
            f"Scalar frame has {value_count} values, but the mesh has "
            f"{mesh.n_points} points and {mesh.n_cells} cells"
        )
    if len(matches) > 1:
        raise VisualisationError(
            "Point and cell counts are equal; specify association='point' "
            "or association='cell'"
        )
    return matches[0]


def _split_scalar_frames(
    values: ArrayLike | Sequence[ArrayLike],
    mesh: pv.DataSet,
    association: AssociationPreference,
) -> tuple[list[NDArray[Any]], Association]:
    """Normalise scalar input into one spatial array per frame."""
    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, np.ndarray)
    ):
        frame_list = [np.asanyarray(item) for item in values]
        if not frame_list:
            raise VisualisationError("scalar_frames cannot be empty")
        if all(frame.ndim >= 1 for frame in frame_list):
            first_count = int(frame_list[0].shape[0])
            try:
                resolved = _resolve_association_from_size(
                    mesh, first_count, association
                )
            except VisualisationError:
                pass
            else:
                expected = (
                    mesh.n_points if resolved == "point" else mesh.n_cells
                )
                for index, frame in enumerate(frame_list):
                    if frame.shape[0] != expected:
                        raise VisualisationError(
                            f"Scalar frame {index} does not contain {expected} "
                            f"{resolved} values"
                        )
                return frame_list, resolved

    array = np.asanyarray(values)

    if array.ndim == 0:
        raise VisualisationError("scalar_frames must contain spatial arrays")
    if array.ndim == 1:
        resolved = _resolve_association_from_size(
            mesh, int(array.shape[0]), association
        )
        return [array], resolved

    # Prefer (frames, spatial values), then accept (spatial values, frames).
    try:
        resolved = _resolve_association_from_size(
            mesh, int(array.shape[1]), association
        )
    except VisualisationError as row_error:
        try:
            resolved = _resolve_association_from_size(
                mesh, int(array.shape[0]), association
            )
        except VisualisationError:
            raise row_error
        return [array[:, index] for index in range(array.shape[1])], resolved
    return [array[index] for index in range(array.shape[0])], resolved


class MeshVisualisation:
    """Manage a scalar mesh view and its interactive overlays.

    Parameters
    ----------
    mesh:
        One PyVista data set or a time-ordered sequence of data sets.
    scalar_name:
        Point/cell array to display.  The active or first numeric array is used
        when omitted.
    scalar_frames:
        Optional scalar values for one static mesh or a frame-aligned mesh
        sequence. Accepted layouts are ``(n_frames, n_values)`` and
        ``(n_values, n_frames)``. A sequence of multi-component arrays is also
        accepted. When the scalar-frame and mesh-frame counts match, each
        scalar frame is displayed on its corresponding geometry snapshot.
    association:
        Resolve values as point or cell data.  ``"auto"`` is unambiguous for
        named mesh arrays and for meshes whose point/cell counts differ.
    """

    def __init__(
        self,
        mesh: pv.DataSet | Sequence[pv.DataSet],
        scalar_name: str | None = None,
        *,
        scalar_frames: ArrayLike | Sequence[ArrayLike] | None = None,
        association: AssociationPreference = "auto",
        times: Sequence[float] | ArrayLike | None = None,
        plotter: pv.Plotter | None = None,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1200, 850),
        title: str = "Mesh visualisation",
        cmap: str = "viridis",
    ) -> None:
        if association not in ("auto", "point", "cell"):
            raise ValueError("association must be 'auto', 'point', or 'cell'")

        source_meshes = _as_dataset_sequence(mesh)
        self.scalar_name = scalar_name or _automatic_scalar_name(source_meshes[0])
        self.title = title
        self.cmap = cmap
        self.plotter = (
            plotter
            if plotter is not None
            else pv.Plotter(off_screen=off_screen, window_size=window_size)
        )
        self._owns_plotter = plotter is None
        self._source_meshes = source_meshes
        self._scalar_frames: list[NDArray[Any]] | None = None

        if scalar_frames is not None:
            frame_values, resolved = _split_scalar_frames(
                scalar_frames, source_meshes[0], association
            )
            self._scalar_frames = frame_values
            self.association: Association = resolved
            frame_count = len(frame_values)
        else:
            _, resolved = resolve_array(
                source_meshes[0], self.scalar_name, association
            )
            if resolved == "field":
                raise VisualisationError("Field data cannot color a mesh")
            self.association = resolved
            for index, frame_mesh in enumerate(source_meshes[1:], start=1):
                try:
                    _, frame_association = resolve_array(
                        frame_mesh, self.scalar_name, self.association
                    )
                except AnalysisError as exc:
                    raise VisualisationError(
                        f"Mesh frame {index} has no usable "
                        f"{self.association} array {self.scalar_name!r}"
                    ) from exc
                if frame_association != self.association:
                    raise VisualisationError(
                        f"Scalar association changes in mesh frame {index}"
                    )
            frame_count = len(source_meshes)

        self.times = _normalise_times(times, frame_count)
        self.frame_count = frame_count
        self.current_frame = 0
        self.mesh = self._mesh_for_frame(0)
        self.scalar_range = self._global_scalar_range()

        self._main_actor: Any | None = None
        self._slice_actors: list[Any] = []
        self.slices: list[pv.DataSet] = []
        self._slice_settings: dict[str, Any] | None = None
        self._hotspot_actor: Any | None = None
        self._hotspot_settings: dict[str, Any] | None = None
        self.hotspots: list[Hotspot] = []
        self.hotspot_cell_ids = np.empty(0, dtype=np.int64)
        self._highlight_actor: Any | None = None
        self._highlight_settings: dict[str, Any] | None = None
        self.highlighted_cell_ids = np.empty(0, dtype=np.int64)
        self._status_actor: Any | None = None
        self.status_text = ""
        self._pick_callback: PickCallback | None = None
        self._pick_additive = False
        self._initialized = False

    def _mesh_for_frame(self, frame_index: int) -> pv.DataSet:
        use_frame_geometry = len(self._source_meshes) == self.frame_count
        source = self._source_meshes[frame_index if use_frame_geometry else 0]
        result = source.copy(deep=True)
        if self._scalar_frames is not None:
            data = (
                result.point_data
                if self.association == "point"
                else result.cell_data
            )
            data[self.scalar_name] = np.asanyarray(
                self._scalar_frames[frame_index]
            )
        result.cell_data[_ORIGINAL_CELL_IDS] = np.arange(
            result.n_cells, dtype=np.int64
        )
        return result

    def _frame_values(self, frame_index: int) -> NDArray[Any]:
        if self._scalar_frames is not None:
            return np.asanyarray(self._scalar_frames[frame_index])
        source = self._source_meshes[frame_index]
        values, _ = resolve_array(source, self.scalar_name, self.association)
        return values

    def _global_scalar_range(self) -> tuple[float, float]:
        minima: list[float] = []
        maxima: list[float] = []
        for index in range(self.frame_count):
            values = np.asarray(scalar_values(self._frame_values(index)))
            finite = values[np.isfinite(values)]
            if finite.size:
                minima.append(float(np.min(finite)))
                maxima.append(float(np.max(finite)))
        if not minima:
            raise VisualisationError(
                f"Scalar array {self.scalar_name!r} has no finite values"
            )
        minimum, maximum = min(minima), max(maxima)
        if minimum == maximum:
            padding = max(abs(minimum) * 1e-6, 1e-12)
            minimum -= padding
            maximum += padding
        return minimum, maximum

    @property
    def current_time(self) -> float | None:
        """Time of the current frame, if a time vector was provided."""
        if self.times is None:
            return None
        return float(self.times[self.current_frame])

    @property
    def scalar_data_type(self) -> DataType:
        """Detected semantic type of the active scalar array."""
        return detect_data_type(self._frame_values(self.current_frame))

    @property
    def main_actor(self) -> Any | None:
        """Actor used for the primary scalar mesh."""
        return self._main_actor

    @property
    def slice_settings(self) -> dict[str, Any]:
        """Copy of the active slice presentation settings."""
        return dict(self._slice_settings or {})

    def mesh_options(self) -> dict[str, Any]:
        """Return the active scalar rendering options."""
        return self._mesh_options()

    def configure_pick_handler(
        self,
        callback: PickCallback | None,
        *,
        additive: bool = False,
    ) -> None:
        """Configure selection handling for an externally managed picker."""
        self._pick_callback = callback
        self._pick_additive = bool(additive)

    def picked_original_ids(
        self,
        picked: pv.DataSet | pv.MultiBlock | None,
    ) -> NDArray[np.int64]:
        """Resolve a picker output to original mesh cell IDs."""
        return self._picked_original_ids(picked)

    def handle_picked_cells(
        self,
        picked: pv.DataSet | pv.MultiBlock | None,
    ) -> None:
        """Apply a picker output through the configured selection handler."""
        self._handle_picked_cells(picked)

    def replace_main_actor(self, **mesh_options: Any) -> Any:
        """Recreate the primary actor, optionally overriding its presentation."""
        return self._replace_main_actor(mesh_options or None)

    def replace_slice_actors(self, actors: Iterable[Any]) -> None:
        """Replace the renderer-owned slice actors with an external set."""
        self._remove_slice_actors()
        self._slice_actors.extend(actors)

    def _mesh_options(self) -> dict[str, Any]:
        data_type = self.scalar_data_type
        return {
            "scalars": self.scalar_name,
            "preference": self.association,
            "clim": self.scalar_range,
            "cmap": self.cmap,
            "categories": data_type
            in (DataType.BOOLEAN, DataType.CATEGORICAL),
            "show_edges": False,
            "scalar_bar_args": {"title": self.scalar_name},
        }

    def _render(self, render: bool) -> None:
        if render:
            self.plotter.render()

    def initialize_user_view(
        self,
        *,
        camera_position: Any | None = None,
        background: str = "#202124",
        show_axes: bool = True,
        enable_picking: bool = False,
        render: bool = True,
    ) -> Any:
        """Create the main actor and initialise camera, axes, and status."""
        self.plotter.set_background(background)
        self._main_actor = self.plotter.add_mesh(
            self.mesh,
            name=_MAIN_ACTOR,
            reset_camera=False,
            render=False,
            **self._mesh_options(),
        )
        if show_axes:
            self.plotter.show_axes()
        if camera_position is None:
            self.plotter.view_isometric(render=False)
            self.plotter.reset_camera(render=False)
        else:
            self.plotter.camera_position = camera_position
        self._initialized = True
        self.update_status_text(render=False)
        if enable_picking:
            self.enable_cell_picking()
        self._render(render)
        return self._main_actor

    def _replace_main_actor(
        self,
        mesh_options: dict[str, Any] | None = None,
    ) -> Any:
        camera_position = (
            self.plotter.camera_position if self._initialized else None
        )
        if self._main_actor is not None:
            self.plotter.remove_actor(
                self._main_actor, reset_camera=False, render=False
            )
        self._main_actor = self.plotter.add_mesh(
            self.mesh,
            name=_MAIN_ACTOR,
            reset_camera=False,
            render=False,
            **(self._mesh_options() if mesh_options is None else mesh_options),
        )
        if camera_position is not None:
            self.plotter.camera_position = camera_position
        return self._main_actor

    def update_scalar_frame(
        self, frame_index: int, *, render: bool = True
    ) -> pv.DataSet:
        """Switch the displayed scalar frame while preserving the camera."""
        if isinstance(frame_index, bool) or not isinstance(
            frame_index, (int, np.integer)
        ):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError(
                f"frame_index must be in 0..{self.frame_count - 1}"
            )
        if not self._initialized:
            self.initialize_user_view(render=False)

        self.current_frame = index
        self.mesh = self._mesh_for_frame(index)
        self._replace_main_actor()

        if self._slice_settings is not None:
            settings = dict(self._slice_settings)
            self.show_slices(render=False, **settings)
        if self._hotspot_settings is not None:
            settings = dict(self._hotspot_settings)
            self.show_hotspot_cells(render=False, **settings)
        if self.highlighted_cell_ids.size:
            valid = self.highlighted_cell_ids[
                self.highlighted_cell_ids < self.mesh.n_cells
            ]
            settings = dict(self._highlight_settings or {})
            self.highlight_cells(valid, render=False, **settings)

        self.update_status_text(render=False)
        self._render(render)
        return self.mesh

    def replace_scalar_frames(
        self,
        scalar_frames: ArrayLike | Sequence[ArrayLike],
        *,
        times: Sequence[float] | ArrayLike | None = None,
        scalar_name: str | None = None,
        frame_index: int = 0,
        render: bool = True,
    ) -> pv.DataSet:
        """Replace the active scalar series while retaining source geometry.

        This is used when the UI explicitly switches between real results and
        simulated data. The association stays unchanged; ``scalar_name`` may
        update the displayed quantity label. Frame count, times, range,
        actors, and overlays are refreshed for the new values. A frame-aligned
        geometry sequence is
        used when its length matches the replacement series; otherwise the
        first geometry snapshot is reused.
        """
        frame_values, resolved = _split_scalar_frames(
            scalar_frames,
            self._source_meshes[0],
            self.association,
        )
        if resolved != self.association:
            raise VisualisationError("Replacement scalar association changed")
        if isinstance(frame_index, bool) or not isinstance(
            frame_index, (int, np.integer)
        ):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        if index < 0 or index >= len(frame_values):
            raise IndexError(
                f"frame_index must be in 0..{len(frame_values) - 1}"
            )

        if scalar_name is not None:
            if not scalar_name.strip():
                raise ValueError("scalar_name cannot be empty")
            self.scalar_name = scalar_name
        self._scalar_frames = frame_values
        self.frame_count = len(frame_values)
        self.times = _normalise_times(times, self.frame_count)
        self.scalar_range = self._global_scalar_range()

        if not self._initialized:
            self.current_frame = index
            self.mesh = self._mesh_for_frame(index)
            self._render(render)
            return self.mesh
        return self.update_scalar_frame(index, render=render)

    def set_scalar_range(
        self,
        minimum: float | Sequence[float] | None = None,
        maximum: float | None = None,
        *,
        render: bool = True,
    ) -> tuple[float, float]:
        """Set the color range, or restore the global range with no values."""
        if (
            minimum is not None
            and not np.isscalar(minimum)
            and not isinstance(minimum, str)
        ):
            range_values = np.asarray(minimum, dtype=np.float64)
            if maximum is not None or range_values.shape != (2,):
                raise ValueError("Pass either (minimum, maximum) or two values")
            minimum, maximum = map(float, range_values)
        elif minimum is None and maximum is None:
            minimum, maximum = self._global_scalar_range()
        elif minimum is None or maximum is None:
            raise ValueError("Both minimum and maximum are required")
        else:
            minimum, maximum = float(minimum), float(maximum)

        if not np.isfinite([minimum, maximum]).all():
            raise ValueError("Scalar range must be finite")
        if minimum >= maximum:
            raise ValueError("Scalar range minimum must be below maximum")
        self.scalar_range = (float(minimum), float(maximum))

        actors = [self._main_actor, *self._slice_actors]
        for actor in actors:
            mapper = getattr(actor, "mapper", None)
            if mapper is not None:
                mapper.scalar_range = self.scalar_range
        self.update_status_text(render=False)
        self._render(render)
        return self.scalar_range

    def _remove_slice_actors(self) -> None:
        for actor in self._slice_actors:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
        self._slice_actors.clear()

    def _nearest_nonempty_slice(
        self,
        normal: str | Sequence[float],
        origin: tuple[float, float, float],
    ) -> pv.DataSet:
        """Return the requested slice, using the nearest cell if it is empty."""
        sliced = self.mesh.slice(normal=normal, origin=origin)
        if sliced.n_points or not self.mesh.n_cells:
            return sliced

        if isinstance(normal, str):
            normal_vector = {
                "x": (1.0, 0.0, 0.0),
                "y": (0.0, 1.0, 0.0),
                "z": (0.0, 0.0, 1.0),
                "-x": (-1.0, 0.0, 0.0),
                "-y": (0.0, -1.0, 0.0),
                "-z": (0.0, 0.0, -1.0),
            }.get(normal.lower())
            if normal_vector is None:
                return sliced
            direction = np.asarray(normal_vector, dtype=np.float64)
        else:
            direction = np.asarray(normal, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all():
                return sliced

        magnitude = float(np.linalg.norm(direction))
        if magnitude == 0.0:
            return sliced
        direction /= magnitude

        centers = np.asarray(self.mesh.cell_centers().points)
        finite_centers = centers[np.isfinite(centers).all(axis=1)]
        if not finite_centers.size:
            return sliced

        requested_offset = float(np.dot(origin, direction))
        cell_offsets = finite_centers @ direction
        nearest_offset = float(
            cell_offsets[np.argmin(np.abs(cell_offsets - requested_offset))]
        )
        fallback_origin = (
            np.asarray(origin) + (nearest_offset - requested_offset) * direction
        )
        return self.mesh.slice(normal=normal, origin=fallback_origin)

    def show_slices(
        self,
        visible: bool = True,
        *,
        origin: Sequence[float] | None = None,
        normals: Sequence[str | Sequence[float]] = ("x", "y", "z"),
        ensure_visible: bool | None = None,
        opacity: float = 1.0,
        show_edges: bool = True,
        line_width: float = 1.0,
        render: bool = True,
    ) -> list[pv.DataSet]:
        """Show or hide fixed slices through the active mesh.

        Centre slices that fall in a gap in the mesh are moved to the nearest
        cell-crossing plane.  Pass an explicit origin to retain exact-plane
        behavior, or set ``ensure_visible`` directly to override the default.
        """
        self._remove_slice_actors()
        if not visible:
            self.slices = []
            self._slice_settings = None
            self.update_status_text("Slices hidden", render=False)
            self._render(render)
            return []
        if not 0 <= opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")
        normal_values = tuple(normals)
        if not normal_values:
            raise ValueError("At least one slice normal is required")
        slice_origin = (
            tuple(float(item) for item in origin)
            if origin is not None
            else tuple(float(item) for item in self.mesh.center)
        )
        if len(slice_origin) != 3 or not np.isfinite(slice_origin).all():
            raise ValueError("origin must contain three finite coordinates")
        keep_slices_visible = (
            origin is None if ensure_visible is None else bool(ensure_visible)
        )

        slices: list[pv.DataSet] = []
        mesh_options = self._mesh_options()
        mesh_options.pop("show_edges")
        for index, normal in enumerate(normal_values):
            if keep_slices_visible:
                sliced = self._nearest_nonempty_slice(normal, slice_origin)
            else:
                sliced = self.mesh.slice(normal=normal, origin=slice_origin)
            slices.append(sliced)
            actor = self.plotter.add_mesh(
                sliced,
                name=f"{_SLICE_ACTOR_PREFIX}{index}",
                opacity=opacity,
                show_edges=show_edges,
                line_width=line_width,
                pickable=False,
                reset_camera=False,
                render=False,
                **mesh_options,
            )
            self._slice_actors.append(actor)

        self.slices = slices
        self._slice_settings = {
            "visible": True,
            "origin": slice_origin,
            "normals": normal_values,
            "ensure_visible": keep_slices_visible,
            "opacity": opacity,
            "show_edges": show_edges,
            "line_width": line_width,
        }
        self.update_status_text(
            f"Showing {len(slices)} slice(s)", render=False
        )
        self._render(render)
        return slices

    def hide_slices(self, *, render: bool = True) -> None:
        """Remove all slice actors."""
        self.show_slices(False, render=render)

    def _cell_ids_for_hotspots(
        self, hotspots: Iterable[Hotspot]
    ) -> NDArray[np.int64]:
        ids: set[int] = set()
        for hotspot in hotspots:
            if hotspot.association == "cell":
                ids.add(hotspot.index)
            else:
                ids.update(
                    int(item) for item in self.mesh.point_cell_ids(hotspot.index)
                )
        return np.asarray(sorted(ids), dtype=np.int64)

    def _remove_hotspot_actor(self) -> None:
        if self._hotspot_actor is not None:
            self.plotter.remove_actor(
                self._hotspot_actor, reset_camera=False, render=False
            )
            self._hotspot_actor = None

    def show_hotspot_cells(
        self,
        visible: bool = True,
        *,
        mode: Literal["high", "low", "absolute"] = "high",
        threshold: float | None = None,
        percentile: float = 95.0,
        max_hotspots: int | None = 10,
        min_distance: float = 0.0,
        component: int | None = None,
        color: str = "orangered",
        opacity: float = 0.85,
        show_edges: bool = True,
        render: bool = True,
    ) -> list[Hotspot]:
        """Detect hotspots and overlay every cell containing a detection."""
        self._remove_hotspot_actor()
        if not visible:
            self.hotspots = []
            self.hotspot_cell_ids = np.empty(0, dtype=np.int64)
            self._hotspot_settings = None
            self.update_status_text("Hotspots hidden", render=False)
            self._render(render)
            return []

        hotspots = detect_hotspots(
            self.mesh,
            self.scalar_name,
            association=self.association,
            component=component,
            mode=mode,
            threshold=threshold,
            percentile=percentile,
            max_hotspots=max_hotspots,
            min_distance=min_distance,
            frame_index=self.current_frame,
            time=self.current_time,
        )
        cell_ids = self._cell_ids_for_hotspots(hotspots)
        self.hotspots = hotspots
        self.hotspot_cell_ids = cell_ids
        if cell_ids.size:
            cells = self.mesh.extract_cells(cell_ids)
            self._hotspot_actor = self.plotter.add_mesh(
                cells,
                name=_HOTSPOT_ACTOR,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color="black",
                pickable=False,
                reset_camera=False,
                render=False,
            )

        self._hotspot_settings = {
            "visible": True,
            "mode": mode,
            "threshold": threshold,
            "percentile": percentile,
            "max_hotspots": max_hotspots,
            "min_distance": min_distance,
            "component": component,
            "color": color,
            "opacity": opacity,
            "show_edges": show_edges,
        }
        self.update_status_text(
            f"{len(hotspots)} hotspot(s), {len(cell_ids)} cell(s)",
            render=False,
        )
        self._render(render)
        return hotspots

    def hide_hotspot_cells(self, *, render: bool = True) -> None:
        """Remove hotspot cells from the view."""
        self.show_hotspot_cells(False, render=render)

    def highlight_cells(
        self,
        cell_ids: int | Iterable[int] | NDArray[np.integer],
        *,
        color: str = "yellow",
        opacity: float = 0.65,
        show_edges: bool = True,
        append: bool = False,
        render: bool = True,
    ) -> pv.DataSet | None:
        """Highlight cell IDs, replacing the previous selection by default."""
        raw_ids = [cell_ids] if np.isscalar(cell_ids) else list(cell_ids)
        raw = np.asarray(raw_ids)
        if raw.ndim != 1:
            raise ValueError("cell_ids must be one-dimensional")
        if raw.dtype.kind not in "iu":
            try:
                numeric = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError("cell IDs must be integers") from exc
            if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
                raise TypeError("cell IDs must be integers")
        requested = np.asarray(raw, dtype=np.int64)
        if requested.size and (
            np.any(requested < 0) or np.any(requested >= self.mesh.n_cells)
        ):
            raise IndexError(
                f"cell IDs must be in 0..{self.mesh.n_cells - 1}"
            )
        if append:
            requested = np.concatenate(
                (self.highlighted_cell_ids, requested)
            )
        requested = np.unique(requested)

        if self._highlight_actor is not None:
            self.plotter.remove_actor(
                self._highlight_actor, reset_camera=False, render=False
            )
            self._highlight_actor = None
        self.highlighted_cell_ids = requested
        self._highlight_settings = {
            "color": color,
            "opacity": opacity,
            "show_edges": show_edges,
            "append": False,
        }
        if not requested.size:
            self._highlight_settings = None
            self.update_status_text("Selection cleared", render=False)
            self._render(render)
            return None

        selected = self.mesh.extract_cells(requested)
        self._highlight_actor = self.plotter.add_mesh(
            selected,
            name=_HIGHLIGHT_ACTOR,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color="black",
            pickable=False,
            reset_camera=False,
            render=False,
        )
        preview = ", ".join(map(str, requested[:8]))
        if requested.size > 8:
            preview += ", …"
        self.update_status_text(
            f"Selected {requested.size} cell(s): {preview}", render=False
        )
        self._render(render)
        return selected

    def clear_highlighted_cells(self, *, render: bool = True) -> None:
        """Clear the highlighted selection."""
        self.highlight_cells([], render=render)

    def _picked_original_ids(
        self, picked: pv.DataSet | pv.MultiBlock | None
    ) -> NDArray[np.int64]:
        if isinstance(picked, pv.MultiBlock):
            block_ids = [
                self._picked_original_ids(block)
                for block in picked
                if isinstance(block, (pv.DataSet, pv.MultiBlock))
            ]
            nonempty = [ids for ids in block_ids if ids.size]
            if not nonempty:
                return np.empty(0, dtype=np.int64)
            return np.unique(np.concatenate(nonempty))
        if picked is None or picked.n_cells == 0:
            return np.empty(0, dtype=np.int64)
        for name in (
            _ORIGINAL_CELL_IDS,
            "vtkOriginalCellIds",
            "original_cell_ids",
        ):
            if name in picked.cell_data:
                return np.unique(
                    np.asarray(picked.cell_data[name], dtype=np.int64)
                )

        # This fallback handles pickers that strip all cell-data arrays.
        picked_centers = np.asarray(picked.cell_centers().points)
        mesh_centers = np.asarray(self.mesh.cell_centers().points)
        ids = [
            int(np.argmin(np.linalg.norm(mesh_centers - point, axis=1)))
            for point in picked_centers
        ]
        return np.unique(np.asarray(ids, dtype=np.int64))

    def _handle_picked_cells(
        self,
        picked: pv.DataSet | pv.MultiBlock | None,
    ) -> None:
        ids = self._picked_original_ids(picked)
        self.highlight_cells(
            ids, append=self._pick_additive, render=False
        )
        if self._pick_callback is not None:
            self._pick_callback(ids.copy())
        self._render(True)

    def enable_cell_picking(
        self,
        callback: PickCallback | None = None,
        *,
        through: bool = False,
        additive: bool = False,
        start: bool = False,
        show_message: bool = True,
        **picker_kwargs: Any,
    ) -> None:
        """Enable cell picking and feed original cell IDs to ``callback``."""
        if not self._initialized:
            self.initialize_user_view(render=False)
        self._pick_callback = callback
        self._pick_additive = additive
        self.plotter.enable_cell_picking(
            callback=self._handle_picked_cells,
            through=through,
            show=False,
            show_message=show_message,
            start=start,
            **picker_kwargs,
        )
        self.update_status_text(
            "Cell picking enabled (press R to select)", render=False
        )

    def disable_cell_picking(self) -> None:
        """Disable the active picker."""
        self.plotter.disable_picking()
        self.update_status_text("Cell picking disabled")

    def _default_status_text(self) -> str:
        frame = f"Frame {self.current_frame + 1}/{self.frame_count}"
        if self.current_time is not None:
            frame += f"  t={self.current_time:g}"
        low, high = self.scalar_range
        return (
            f"{frame}  |  {self.scalar_name} ({self.association}, "
            f"{self.scalar_data_type.value})  |  range [{low:.5g}, {high:.5g}]"
        )

    def update_status_text(
        self,
        text: str | None = None,
        *,
        position: str = "lower_left",
        font_size: int = 14,
        color: str = "white",
        render: bool = True,
    ) -> str:
        """Update the persistent status annotation and return its text."""
        status = self._default_status_text() if text is None else str(text)
        self.status_text = status
        self._status_actor = self.plotter.add_text(
            status,
            position=position,
            font_size=font_size,
            color=color,
            name=_STATUS_ACTOR,
            render=False,
        )
        self._render(render)
        return status

    def save_screenshot(
        self,
        path: str | Path,
        *,
        transparent_background: bool = False,
        scale: int = 1,
    ) -> Path:
        """Save the current render window to an image and return its path."""
        if scale < 1:
            raise ValueError("scale must be at least 1")
        destination = Path(path).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(".png")
        if destination.suffix.casefold() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }:
            raise ValueError("Screenshot must be PNG, JPEG, BMP, or TIFF")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self._initialized:
            self.initialize_user_view(render=False)
        self.update_status_text(render=False)
        self.plotter.screenshot(
            destination,
            transparent_background=transparent_background,
            scale=scale,
            return_img=False,
        )
        self.update_status_text(render=False)
        return destination.resolve()

    def show(self, **kwargs: Any) -> Any:
        """Open the interactive window."""
        if not self._initialized:
            self.initialize_user_view(render=False)
        return self.plotter.show(title=self.title, **kwargs)

    def close(self) -> None:
        """Close the render window."""
        self.plotter.close()


# US-spelling and shorter aliases are useful to downstream callers.
MeshVisualization = MeshVisualisation
Visualisation = MeshVisualisation
Visualization = MeshVisualisation


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Display a PyVista mesh with slices, hotspot cells, cell picking, "
            "and screenshots."
        )
    )
    parser.add_argument("mesh", type=Path, help="mesh readable by PyVista")
    parser.add_argument("--scalars", help="point or cell array to display")
    parser.add_argument(
        "--association",
        choices=("auto", "point", "cell"),
        default="auto",
    )
    parser.add_argument("--slices", action="store_true")
    parser.add_argument("--hotspots", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="create an off-screen plotter (requires a headless VTK backend)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    try:
        mesh = load_mesh(args.mesh)
        if not isinstance(mesh, pv.DataSet):
            raise VisualisationError(
                f"{args.mesh} contains {type(mesh).__name__}, not one DataSet"
            )
        viewer = MeshVisualisation(
            mesh,
            args.scalars,
            association=args.association,
            off_screen=args.off_screen or bool(args.screenshot),
        )
        viewer.initialize_user_view(render=False)
        if args.slices:
            viewer.show_slices(render=False)
        if args.hotspots:
            viewer.show_hotspot_cells(
                threshold=args.threshold,
                percentile=args.percentile,
                render=False,
            )
        if args.screenshot:
            viewer.save_screenshot(args.screenshot)
            viewer.close()
        else:
            viewer.show()
    except (DataLoadError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"visualisation: error: {exc}") from exc
    return 0


__all__ = [
    "MeshVisualisation",
    "MeshVisualization",
    "Visualisation",
    "Visualization",
    "VisualisationError",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
