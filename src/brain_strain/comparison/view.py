"""Side-by-side observation/simulation visualisation.

The comparison UI deliberately separates three ideas:

* observation and simulation fields are always rendered with their own units
  and colour ranges;
* metadata differences are reported without requiring aligned meshes; and
* the optional difference panel is a unit-free, normalized visual contrast,
  available only when both fields use the same cell geometry.

The difference panel is not a physical error field and must not be interpreted
as validation, material inference, or an injury metric.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv

from ..observation_case import ObservationCase
from .policy import (
    ObservationCaseComparability,
    decide_observation_case_comparability,
)

_OBSERVATION_ARRAY = "__comparison_observation__"
_SIMULATION_ARRAY = "__comparison_simulation__"
_DIFFERENCE_ARRAY = "__comparison_normalized_difference__"
_BACKGROUND = "#202124"
_TEXT_COLOR = "white"


CaseMetadata = ObservationCase | Mapping[str, Any] | None


def _compact_title(value: str, width: int = 54) -> str:
    return value if len(value) <= width else value[: width - 1].rstrip() + "…"


@dataclass(slots=True)
class CaseDisplayData:
    """One scalar case prepared for a comparison renderer."""

    role: str
    mesh: pv.DataSet
    times: npt.ArrayLike
    scalar_series: npt.ArrayLike
    field_name: str
    metadata: CaseMetadata = None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, pv.DataSet):
            raise TypeError(f"{self.role} mesh must be a PyVista DataSet")
        if self.mesh.n_cells == 0:
            raise ValueError(f"{self.role} mesh cannot be empty")
        self.times = np.asarray(self.times, dtype=np.float64)
        self.scalar_series = np.asarray(self.scalar_series, dtype=np.float64)
        if self.times.ndim != 1 or self.times.size == 0:
            raise ValueError(f"{self.role} times must be a non-empty vector")
        if not np.isfinite(self.times).all():
            raise ValueError(f"{self.role} times contain NaN or infinity")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{self.role} times must be strictly increasing")
        expected = (self.times.size, self.mesh.n_cells)
        if self.scalar_series.shape != expected:
            raise ValueError(
                f"{self.role} scalar_series must have shape {expected}; "
                f"received {self.scalar_series.shape}"
            )
        if not np.isfinite(self.scalar_series).any():
            raise ValueError(f"{self.role} scalar_series has no finite values")
        if not self.field_name.strip():
            raise ValueError(f"{self.role} field_name cannot be empty")

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def title(self) -> str:
        if isinstance(self.metadata, ObservationCase):
            return self.metadata.title
        if isinstance(self.metadata, Mapping):
            value = self.metadata.get("title")
            if value:
                return str(value)
        return f"{self.role.title()} case"


@dataclass(slots=True)
class ComparisonUIState:
    """Interactive state shared by the comparison renderers."""

    comparison_frame: int = 0
    show_difference: bool = False


def _finite_range(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Scalar data contains no finite values")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum == maximum:
        padding = max(abs(minimum) * 1e-6, 1e-12)
        return minimum - padding, maximum + padding
    return minimum, maximum


def normalize_visual_series(
    values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Normalize one complete series to [0, 1] while preserving NaNs.

    A constant finite series maps to 0.5.  The result describes relative
    colour position only; it has no physical units.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(result)
    if not finite.any():
        raise ValueError("Scalar data contains no finite values")
    minimum = float(np.min(result[finite]))
    maximum = float(np.max(result[finite]))
    if minimum == maximum:
        result[finite] = 0.5
    else:
        result[finite] = (result[finite] - minimum) / (maximum - minimum)
    return result


def meshes_share_cell_geometry(
    first: pv.DataSet,
    second: pv.DataSet,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether cell-wise subtraction is structurally safe.

    This is intentionally conservative.  It checks mesh type, point and cell
    counts, point coordinates, and explicit topology arrays when available.
    It does not perform registration or resampling.
    """
    if type(first) is not type(second):
        return False
    if first.n_points != second.n_points or first.n_cells != second.n_cells:
        return False
    if not np.allclose(
        np.asarray(first.points),
        np.asarray(second.points),
        rtol=0.0,
        atol=tolerance,
        equal_nan=False,
    ):
        return False

    topology_attributes = (
        "celltypes",
        "offset",
        "cell_connectivity",
        "faces",
        "lines",
        "strips",
        "verts",
    )
    for name in topology_attributes:
        first_values = getattr(first, name, None)
        second_values = getattr(second, name, None)
        if first_values is None and second_values is None:
            continue
        if first_values is None or second_values is None:
            return False
        if not np.array_equal(
            np.asarray(first_values), np.asarray(second_values)
        ):
            return False
    return True


class BrainComparisonUI:
    """Render observation, simulation, and an optional visual difference."""

    def __init__(
        self,
        observation: CaseDisplayData,
        simulation: CaseDisplayData,
        *,
        show_difference: bool = False,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1500, 760),
        render: bool = True,
    ) -> None:
        self.observation = observation
        self.simulation = simulation
        self._render_enabled = bool(render)
        self._callbacks_active = False
        self.state = ComparisonUIState()
        self.max_frame_count = max(
            observation.frame_count, simulation.frame_count
        )
        self.difference_available = meshes_share_cell_geometry(
            observation.mesh, simulation.mesh
        )
        self.difference_unavailable_reason = (
            None
            if self.difference_available
            else (
                "Difference unavailable: meshes do not share identical cell "
                "geometry. Registration and resampling are not performed."
            )
        )
        self.comparability = self._build_comparability()
        self._observation_normalized = normalize_visual_series(
            observation.scalar_series
        )
        self._simulation_normalized = normalize_visual_series(
            simulation.scalar_series
        )

        self.plotter = pv.Plotter(
            shape=(1, 3),
            off_screen=off_screen,
            window_size=window_size,
            border=True,
            border_color="white",
            border_width=1.0,
        )
        self.plotter.theme.font.color = _TEXT_COLOR
        self.plotter.theme.font.size = 13
        for renderer in self.plotter.renderers:
            renderer.set_background(_BACKGROUND)

        self._observation_mesh = observation.mesh.copy(deep=True)
        self._simulation_mesh = simulation.mesh.copy(deep=True)
        self._difference_mesh = (
            observation.mesh.copy(deep=True)
            if self.difference_available
            else None
        )
        self._observation_actor: Any | None = None
        self._simulation_actor: Any | None = None
        self._difference_actor: Any | None = None
        self.difference_button: Any | None = None
        self.frame_slider: Any | None = None

        self._initialize_case_panels()
        self._build_widgets()
        self._callbacks_active = True
        self.show_difference(bool(show_difference))

    def _build_comparability(self) -> ObservationCaseComparability | None:
        if self.observation.metadata is None or self.simulation.metadata is None:
            return None
        return decide_observation_case_comparability(
            self.observation.metadata,
            self.simulation.metadata,
        )

    def _case_frame_index(self, case: CaseDisplayData) -> int:
        if self.max_frame_count <= 1 or case.frame_count <= 1:
            return 0
        progress = self.state.comparison_frame / (self.max_frame_count - 1)
        return int(round(progress * (case.frame_count - 1)))

    def _difference_values(self) -> npt.NDArray[np.float64] | None:
        if not self.difference_available:
            return None
        observation_index = self._case_frame_index(self.observation)
        simulation_index = self._case_frame_index(self.simulation)
        return (
            self._observation_normalized[observation_index]
            - self._simulation_normalized[simulation_index]
        )

    @property
    def current_difference(self) -> npt.NDArray[np.float64] | None:
        """Current signed normalized difference, or ``None`` if unavailable."""
        values = self._difference_values()
        return None if values is None else values.copy()

    def _initialize_case_panels(self) -> None:
        observation_index = self._case_frame_index(self.observation)
        simulation_index = self._case_frame_index(self.simulation)
        self._observation_mesh.cell_data[_OBSERVATION_ARRAY] = (
            self.observation.scalar_series[observation_index]
        )
        self._simulation_mesh.cell_data[_SIMULATION_ARRAY] = (
            self.simulation.scalar_series[simulation_index]
        )

        self.plotter.subplot(0, 0)
        self._observation_actor = self.plotter.add_mesh(
            self._observation_mesh,
            name="comparison-observation",
            scalars=_OBSERVATION_ARRAY,
            preference="cell",
            clim=_finite_range(self.observation.scalar_series),
            cmap="viridis",
            show_edges=False,
            scalar_bar_args={"title": self.observation.field_name},
            render=False,
        )
        self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)

        self.plotter.subplot(0, 1)
        self._simulation_actor = self.plotter.add_mesh(
            self._simulation_mesh,
            name="comparison-simulation",
            scalars=_SIMULATION_ARRAY,
            preference="cell",
            clim=_finite_range(self.simulation.scalar_series),
            cmap="plasma",
            show_edges=False,
            scalar_bar_args={"title": self.simulation.field_name},
            render=False,
        )
        self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)

        self.plotter.subplot(0, 2)
        if self._difference_mesh is not None:
            difference = self._difference_values()
            assert difference is not None
            self._difference_mesh.cell_data[_DIFFERENCE_ARRAY] = difference
            self._difference_actor = self.plotter.add_mesh(
                self._difference_mesh,
                name="comparison-difference",
                scalars=_DIFFERENCE_ARRAY,
                preference="cell",
                clim=(-1.0, 1.0),
                cmap="coolwarm",
                show_edges=False,
                scalar_bar_args={
                    "title": "Normalized observation - simulation"
                },
                render=False,
            )
            self.plotter.view_isometric(render=False)
            self.plotter.reset_camera(render=False)
        else:
            self.plotter.add_text(
                self.difference_unavailable_reason or "Difference unavailable",
                position="upper_left",
                font_size=12,
                color="darkorange",
                name="comparison-difference-unavailable",
                render=False,
            )
        self._update_panel_text()

    def _build_widgets(self) -> None:
        self.plotter.subplot(0, 0)
        final_frame = self.max_frame_count - 1
        self.frame_slider = self.plotter.add_slider_widget(
            callback=self._on_frame_slider,
            rng=(0, max(final_frame, 1)),
            value=0,
            title=(
                "Comparison position (1 total)"
                if final_frame == 0
                else f"Comparison position ({self.max_frame_count} total)"
            ),
            pointa=(0.18, 0.82),
            pointb=(0.82, 0.82),
            fmt="1" if final_frame == 0 else "%.0f",
            interaction_event="end",
        )
        if final_frame == 0:
            self.frame_slider.SetProcessEvents(False)

        self.difference_button = self.plotter.add_checkbox_button_widget(
            callback=self._on_difference_toggle,
            value=False,
            position=(12, 74),
            size=34,
            color_on="tomato",
            color_off="grey",
        )
        self.plotter.add_text(
            "Show normalized visual difference",
            position=(56, 80),
            font_size=12,
            color=_TEXT_COLOR,
            name="comparison-difference-label",
            render=False,
        )
        if not self.difference_available:
            self.difference_button.SetProcessEvents(False)

        self.plotter.add_key_event("d", self.toggle_difference)
        self.plotter.add_key_event("v", self.reset_cameras)
        self.plotter.add_key_event("s", self.save_screenshot)

    def _on_frame_slider(self, value: float) -> None:
        if not self._callbacks_active:
            return
        self.set_comparison_frame(int(round(value)))

    def _on_difference_toggle(self, enabled: bool) -> None:
        if not self._callbacks_active:
            return
        self.show_difference(bool(enabled))

    @staticmethod
    def _set_button_state(button: Any | None, enabled: bool) -> None:
        if button is None:
            return
        representation = button.GetRepresentation()
        if representation is not None:
            representation.SetState(int(enabled))
            representation.Modified()

    def _apply_layout(self) -> None:
        renderers = self.plotter.renderers
        if self.state.show_difference:
            viewports = (
                (0.0, 0.0, 1.0 / 3.0, 1.0),
                (1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0),
                (2.0 / 3.0, 0.0, 1.0, 1.0),
            )
            for renderer, viewport in zip(renderers, viewports, strict=True):
                renderer.viewport = viewport
                renderer.SetDraw(True)
        else:
            renderers[0].viewport = (0.0, 0.0, 0.5, 1.0)
            renderers[1].viewport = (0.5, 0.0, 1.0, 1.0)
            renderers[0].SetDraw(True)
            renderers[1].SetDraw(True)
            renderers[2].SetDraw(False)

    def show_difference(self, enabled: bool = True) -> bool:
        """Show or hide the unit-free visual difference panel."""
        requested = bool(enabled)
        self.state.show_difference = requested and self.difference_available
        self._set_button_state(
            self.difference_button, self.state.show_difference
        )
        self._apply_layout()
        self._update_panel_text()
        self._render()
        return self.state.show_difference

    def toggle_difference(self) -> bool:
        return self.show_difference(not self.state.show_difference)

    def set_comparison_frame(self, frame_index: int) -> int:
        """Set relative progress and update all visible case panels."""
        if isinstance(frame_index, bool) or not isinstance(
            frame_index, (int, np.integer)
        ):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        if index < 0 or index >= self.max_frame_count:
            raise IndexError(
                f"frame_index must be in 0..{self.max_frame_count - 1}"
            )
        self.state.comparison_frame = index
        observation_index = self._case_frame_index(self.observation)
        simulation_index = self._case_frame_index(self.simulation)
        self._observation_mesh.cell_data[_OBSERVATION_ARRAY] = (
            self.observation.scalar_series[observation_index]
        )
        self._simulation_mesh.cell_data[_SIMULATION_ARRAY] = (
            self.simulation.scalar_series[simulation_index]
        )
        if self._difference_mesh is not None:
            difference = self._difference_values()
            assert difference is not None
            self._difference_mesh.cell_data[_DIFFERENCE_ARRAY] = difference
            self._difference_mesh.GetCellData().Modified()
            self._difference_mesh.Modified()
        self._observation_mesh.GetCellData().Modified()
        self._observation_mesh.Modified()
        self._simulation_mesh.GetCellData().Modified()
        self._simulation_mesh.Modified()
        if self.frame_slider is not None:
            representation = self.frame_slider.GetRepresentation()
            representation.SetValue(float(index))
            representation.Modified()
        self._update_panel_text()
        self._render()
        return index

    def _metadata_difference_text(self) -> str:
        result = self.comparability
        if result is None:
            return "Metadata comparison: case metadata not supplied"
        matching_pair = next(
            (
                pair
                for pair in result.field_pairs
                if pair.current.field_name == self.observation.field_name
                and pair.simulation.field_name == self.simulation.field_name
            ),
            result.field_pairs[0] if result.field_pairs else None,
        )
        if matching_pair is None:
            return "Metadata comparison: no matching scalar field records"
        differences = list(matching_pair.limitations)
        if not differences:
            differences.append("declared field semantics align")
        wrapped = [
            textwrap.fill(
                difference,
                width=42,
                initial_indent="- ",
                subsequent_indent="  ",
            )
            for difference in differences
        ]
        return "Metadata differences:\n" + "\n".join(wrapped)

    def _update_panel_text(self) -> None:
        observation_index = self._case_frame_index(self.observation)
        simulation_index = self._case_frame_index(self.simulation)

        self.plotter.subplot(0, 0)
        self.plotter.add_text(
            f"OBSERVATION\n{_compact_title(self.observation.title)}",
            position="upper_edge",
            font_size=13,
            color="mediumseagreen",
            name="comparison-observation-title",
            render=False,
        )
        self.plotter.add_text(
            f"{self.observation.field_name}\n"
            f"Frame {observation_index + 1}/{self.observation.frame_count}\n"
            f"Time {float(self.observation.times[observation_index]):.4g}",
            position="lower_left",
            font_size=11,
            color=_TEXT_COLOR,
            name="comparison-observation-status",
            render=False,
        )

        self.plotter.subplot(0, 1)
        difference_availability = (
            "Visual difference: available (press D)"
            if self.difference_available
            else "Visual difference: unavailable without aligned geometry"
        )
        self.plotter.add_text(
            f"SIMULATION\n{_compact_title(self.simulation.title)}",
            position="upper_edge",
            font_size=13,
            color="darkorange",
            name="comparison-simulation-title",
            render=False,
        )
        self.plotter.add_text(
            f"{self.simulation.field_name}\n"
            f"Frame {simulation_index + 1}/{self.simulation.frame_count}\n"
            f"Time {float(self.simulation.times[simulation_index]):.4g}\n\n"
            f"{self._metadata_difference_text()}\n\n"
            f"{difference_availability}\n"
            "Scope: VISUALISATION ONLY\n"
            "Physical calculation: disabled",
            position="lower_left",
            font_size=10,
            color=_TEXT_COLOR,
            name="comparison-simulation-status",
            render=False,
        )

        self.plotter.subplot(0, 2)
        self.plotter.add_text(
            "NORMALIZED VISUAL DIFFERENCE\nObservation - simulation",
            position="upper_edge",
            font_size=13,
            color="tomato",
            name="comparison-difference-title",
            render=False,
        )
        difference = self._difference_values()
        if difference is not None:
            finite = difference[np.isfinite(difference)]
            mean_absolute = (
                float(np.mean(np.abs(finite))) if finite.size else float("nan")
            )
            maximum_absolute = (
                float(np.max(np.abs(finite))) if finite.size else float("nan")
            )
            difference_text = (
                f"Mean |visual difference|: {mean_absolute:.4f}\n"
                f"Max |visual difference|: {maximum_absolute:.4f}\n\n"
                "Unit-free colour contrast only\n"
                "Not a physical error or validation metric"
            )
        else:
            difference_text = self.difference_unavailable_reason or "Unavailable"
        self.plotter.add_text(
            difference_text,
            position="lower_left",
            font_size=10,
            color=_TEXT_COLOR,
            name="comparison-difference-status",
            render=False,
        )

    def reset_cameras(self) -> None:
        for column in range(3):
            self.plotter.subplot(0, column)
            self.plotter.view_isometric(render=False)
            self.plotter.reset_camera(render=False)
        self._render()

    def save_screenshot(self, path: str | Path | None = None) -> Path:
        output = Path(path or "brain_case_comparison.png").expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(output)
        return output

    def _render(self) -> None:
        if self._render_enabled:
            self.plotter.render()

    def run(self) -> None:
        self.plotter.show(title="Observation and simulation comparison")

    def close(self) -> None:
        self.plotter.close()


__all__ = [
    "BrainComparisonUI",
    "CaseDisplayData",
    "ComparisonUIState",
    "meshes_share_cell_geometry",
    "normalize_visual_series",
]
