#!/usr/bin/env python3
"""Executable user-interface controller for the brain-strain viewer."""

from __future__ import annotations

import subprocess
import sys
import time
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ..analysis import (
    extract_element_history,
    find_global_peak,
)
from ..comparison.view import BrainComparisonUI, CaseDisplayData
from ..io.display import prepare_display_data
from ..io.export import (
    ResultExportData,
    ResultParameter,
    ResultSeries,
    build_result_workbook,
)
from ..io.loader import DataLoadError, load_data, load_mesh
from ..paths import DEFAULT_MESH, PROJECT_ROOT
from ..simulation import (
    DEFAULT_MAXWELL_MODEL,
    DEFAULT_TAU_MAX_SECONDS,
    DEFAULT_TAU_MIN_SECONDS,
    REFERENCE_DURATION_SECONDS,
    REFERENCE_FRAME_COUNT,
    GeneralizedMaxwellModel,
    ImpactMode,
    default_branch_fractions,
    logarithmic_relaxation_times,
    simulate_generalized_maxwell_strain,
)
from .dialogs import choose_open_file, choose_save_file
from .rendering import MeshVisualisation, VisualisationError

ROOT = PROJECT_ROOT
DEFAULT_SIMULATION_FRAME_COUNT = REFERENCE_FRAME_COUNT
SUPPORTED_LOCAL_MESH_EXTENSIONS = (
    ".k",
    ".pvd",
    ".xdmf",
    ".xmf",
    ".case",
    ".e",
    ".ex2",
    ".exii",
    ".exo",
    ".vtk",
    ".vtu",
    ".vtp",
    ".vti",
    ".vtr",
    ".vts",
    ".pvtu",
    ".pvti",
    ".pvtr",
    ".stl",
    ".ply",
    ".obj",
)
SUPPORTED_LOCAL_MESH_PATTERN = " ".join(
    f"*{extension}" for extension in SUPPORTED_LOCAL_MESH_EXTENSIONS
)
_PART_CATEGORY_ARRAY = "__ui_part_category__"
_PARTS_ACTOR = "ui-parts-mesh"
_PARTS_SCALAR_BAR = "Tissue / Part"
_UI_FONT_COLOR = "white"
_UI_LABEL_FONT_SIZE = 14
_UI_DETAIL_FONT_SIZE = 12
_UI_STATUS_FONT_SIZE = 14
_UI_SLIDER_TITLE_HEIGHT = 0.045
_UI_SLIDER_LABEL_HEIGHT = 0.035
_UI_CONTROL_COLUMNS = (10, 300, 590)
_UI_CONTROL_ROWS = (300, 205, 35)
_UI_LABEL_OFFSET = (45, 5)
_UI_CELL_INPUT_POSITION = (10, 135)
_UI_FILE_STATUS_POSITION = (10, 355)
_SLICE_PANEL_SPLIT = 0.72
_SLICE_PANEL_BACKGROUND = "#111827"
_SLICE_PANEL_TITLES = (
    "Sagittal (X)",
    "Coronal (Y)",
    "Axial (Z)",
)
_SLICE_PANEL_VIEWS = ("view_yz", "view_xz", "view_xy")
_SLICE_PANEL_ACTOR_PREFIX = "ui-2d-slice-"
_SLICE_PANEL_TITLE_PREFIX = "ui-2d-slice-title-"
_MAIN_RENDERER_LOCATION = (0, 0)
_SLICE_PANEL_LOCATIONS = ((0, 1), (1, 1), (2, 1))
_FULL_SLIDER_X_RANGE = (0.35, 0.90)
_SPLIT_SLIDER_X_RANGE = (0.25, 0.66)
_MAXWELL_WINDOW_SIZE = (720, 680)
_MAXWELL_WINDOW_BACKGROUND = "#111827"
_MAXWELL_PRIMARY_SLIDER_X_RANGE = (0.06, 0.45)
_MAXWELL_TAU_SLIDER_X_RANGE = (0.55, 0.94)
_MAXWELL_PRIMARY_SLIDER_Y_POSITIONS = (
    0.82,
    0.72,
    0.62,
    0.49,
    0.41,
    0.33,
    0.25,
    0.17,
    0.09,
)
_MAXWELL_TAU_SLIDER_Y_POSITIONS = (
    0.49,
    0.41,
    0.33,
    0.25,
    0.17,
    0.09,
)
_MAXWELL_SLIDER_TITLE_HEIGHT = 0.025
_MAXWELL_SLIDER_LABEL_HEIGHT = 0.020
_MAXWELL_SLIDER_LENGTH = 0.025
_MAXWELL_SLIDER_WIDTH = 0.025
_MAXWELL_TUBE_WIDTH = 0.004
_MAXWELL_END_CAP_LENGTH = 0.012
_MAXWELL_END_CAP_WIDTH = 0.012
_MAXWELL_MIN_TEXT_SCALE = 0.65
_MAXWELL_MAX_TEXT_SCALE = 1.75
_MAXWELL_NARROW_ASPECT_RATIO = 0.75
_MAXWELL_DESKTOP_ASPECT_RATIO = 1.25
_MAXWELL_DESKTOP_MIN_WIDTH = 1080
_MAXWELL_LOG10_TAU_DEFAULT_RANGE = (-6.0, 2.0)
_MAXWELL_LOG10_TAU_MIN_GAP = 0.01
_RESULT_WINDOW_SIZE = (720, 560)
_RESULT_WINDOW_BACKGROUND = "#111827"
_RESULT_EXPORT_METHOD = "Generalized Maxwell (reduced-order demonstration)"
_CELL_ID_DIGITS = "0123456789"
_UI_CAPTURED_CHAR_KEYS = frozenset(_CELL_ID_DIGITS + "r")
_CELL_CLICK_MAX_DRAG_SQUARED = 36


# ReCoDE part IDs from ``part_list_full.k`` and ``bmctk.py``. Structures
# outside these requested tissue groups are kept visible as neutral "Other".
_PART_GROUPS: tuple[tuple[str, str, frozenset[int]], ...] = (
    ("Skin", "red", frozenset({260})),
    ("Skull", "lightblue", frozenset({257})),
    ("CSF", "green", frozenset({24, 256, 258, 259})),
    (
        "Grey matter",
        "yellow",
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
        "brown",
        frozenset({2, 7, 41, 46, 77, 85, 251, 252, 253, 254, 255}),
    ),
    ("Ventricles", "darkblue", frozenset({4, 5, 14, 15, 43, 44})),
    ("Other", "lightgrey", frozenset()),
)


def _viewer_launch_command(mesh_path: Path, field_name: str) -> list[str]:
    """Return a source- or frozen-runtime command for a new viewer."""
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command.extend(("-m", "brain_strain"))
    command.extend((str(mesh_path), "--field", field_name))
    return command


def _initial_open_directory() -> Path:
    """Choose an existing directory for the first native file dialog."""
    default_directory = DEFAULT_MESH.parent
    return default_directory if default_directory.is_dir() else Path.home()


class LocalMeshOpeningUI:
    """Small shared boundary for the launcher's mesh-opening interaction."""

    field_name: str
    open_file_button: Any
    plotter: pv.Plotter
    _callbacks_active: bool
    _last_open_directory: Path

    def _render(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _set_toggle_button_state(button: Any, enabled: bool) -> None:
        representation = button.GetRepresentation()
        if representation is not None:
            representation.SetState(int(enabled))
            representation.Modified()

    def _on_open_file_button(self, _enabled: bool) -> None:
        if not self._callbacks_active:
            return
        try:
            try:
                self.open_local_file()
            except (
                FileNotFoundError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                self._update_file_status(str(exc), error=True)
        finally:
            # VTK supplies a checkbox widget, but this control is momentary.
            self._set_toggle_button_state(self.open_file_button, False)
            self._render()

    def _on_open_file_shortcut(self) -> None:
        self._on_open_file_button(False)

    def _update_file_status(
        self,
        message: str,
        *,
        error: bool = False,
        render: bool = True,
    ) -> None:
        self.plotter.add_text(
            message,
            position=_UI_FILE_STATUS_POSITION,
            font_size=_UI_DETAIL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="open_file_status",
            render=False,
        )
        if render:
            self._render()

    def _choose_local_file(self) -> str:
        return choose_open_file(
            self._last_open_directory,
            title="Open local mesh",
            pattern=SUPPORTED_LOCAL_MESH_PATTERN,
        )

    def open_local_file(
        self,
        path: str | Path | None = None,
        *,
        launch: bool = True,
    ) -> Path | None:
        """Choose and validate a mesh, then open it in a fresh viewer."""
        selected = self._choose_local_file() if path is None else str(path)
        if not selected:
            self._update_file_status("Open local mesh cancelled")
            return None

        mesh_path = Path(selected).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
        if not mesh_path.name.casefold().endswith(
            SUPPORTED_LOCAL_MESH_EXTENSIONS
        ):
            supported = ", ".join(SUPPORTED_LOCAL_MESH_EXTENSIONS)
            raise ValueError(
                f"Unsupported mesh format {mesh_path.suffix or '<none>'!r}. "
                f"Supported formats: {supported}"
            )
        opened_mesh = load_mesh(mesh_path)
        if not isinstance(opened_mesh, pv.DataSet):
            raise ValueError(
                f"{mesh_path} contains {type(opened_mesh).__name__}, "
                "not one mesh data set"
            )

        self._last_open_directory = mesh_path.parent
        if launch:
            subprocess.Popen(
                _viewer_launch_command(mesh_path, self.field_name),
                # A frozen executable has no source checkout to use as its
                # working directory.  Let it inherit the user's directory.
                cwd=None if getattr(sys, "frozen", False) else str(ROOT),
                start_new_session=True,
            )
            message = f"Opened in new viewer: {mesh_path.name}"
        else:
            message = f"Validated local mesh: {mesh_path.name}"
        self._update_file_status(message)
        return mesh_path


@dataclass
class UIState:
    """Current interactive application state."""

    time_index: int = 0
    threshold: float = 0.20
    show_hotspots: bool = True
    show_slices: bool = False
    show_parts: bool = False
    show_simulation_results: bool = False
    drag_select: bool = True
    selected_cell_id: int | None = None
    selected_cell_ids: tuple[int, ...] = ()


class BrainUI(LocalMeshOpeningUI):
    """Connect UI controls to analysis and visualization functions."""

    def __init__(
        self,
        mesh: pv.DataSet,
        times: npt.ArrayLike,
        scalar_series: npt.ArrayLike,
        *,
        mesh_frames: Sequence[pv.DataSet] | None = None,
        field_name: str = "MPS",
        data_is_simulated: bool = False,
        simulation_frame_count: int | None = None,
        simulation_duration: float = REFERENCE_DURATION_SECONDS,
        simulation_model: GeneralizedMaxwellModel = DEFAULT_MAXWELL_MODEL,
        simulation_impact_mode: ImpactMode = "neck-rotation",
        simulation_target_mean_mss: float | None = None,
        simulation_rotation_axis: npt.ArrayLike = (0.0, 0.0, 1.0),
        initial_threshold: float | None = None,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1200, 850),
        enable_picking: bool = True,
        render: bool = True,
    ) -> None:
        self.mesh = mesh
        self.mesh_frames = (
            tuple(mesh_frames) if mesh_frames is not None else (mesh,)
        )
        if not self.mesh_frames:
            raise ValueError("mesh_frames cannot be empty")
        if self.mesh_frames[0].n_cells != mesh.n_cells:
            raise ValueError("mesh_frames must begin with the displayed mesh")
        if any(frame.n_cells != mesh.n_cells for frame in self.mesh_frames):
            raise ValueError("mesh cell count must stay constant across frames")
        self.times = np.asarray(times, dtype=float)
        self.scalar_series = np.asarray(
            scalar_series,
            dtype=float,
        )
        self._real_field_name = field_name
        self.field_name = (
            "Maximum shear strain" if data_is_simulated else field_name
        )
        self.data_is_simulated = bool(data_is_simulated)
        self._real_times = (
            None if self.data_is_simulated else self.times.copy()
        )
        self._real_scalar_series = (
            None if self.data_is_simulated else self.scalar_series.copy()
        )
        self._simulated_times = (
            self.times.copy() if self.data_is_simulated else None
        )
        self._simulated_scalar_series = (
            self.scalar_series.copy() if self.data_is_simulated else None
        )
        self._simulation_frame_count = simulation_frame_count
        self._simulation_duration = float(simulation_duration)
        self._simulation_model = simulation_model
        self._simulation_impact_mode = simulation_impact_mode
        self._simulation_target_mean_mss = simulation_target_mean_mss
        self._simulation_rotation_axis = np.asarray(
            simulation_rotation_axis, dtype=np.float64
        )
        initial_log_taus = np.log10(self._simulation_model.relaxation_times)
        self._maxwell_log10_tau_bounds = (
            min(
                _MAXWELL_LOG10_TAU_DEFAULT_RANGE[0],
                float(np.floor(np.min(initial_log_taus)) - 1.0),
            ),
            max(
                _MAXWELL_LOG10_TAU_DEFAULT_RANGE[1],
                float(np.ceil(np.max(initial_log_taus)) + 1.0),
            ),
        )
        self._maxwell_widgets_updating = False
        self.maxwell_branch_slider: Any | None = None
        self.maxwell_modulus_slider: Any | None = None
        self.maxwell_equilibrium_slider: Any | None = None
        self.maxwell_g_sliders: list[Any] = []
        self.maxwell_tau_sliders: list[Any] = []
        if simulation_frame_count is not None and simulation_frame_count < 1:
            raise ValueError("simulation_frame_count must be positive")
        if (
            self._simulation_duration < 0
            or not np.isfinite(self._simulation_duration)
        ):
            raise ValueError(
                "simulation_duration must be finite and non-negative"
            )
        self._render_enabled = render
        self._callbacks_active = False
        self._cell_id_input = ""
        self._cell_id_input_error: str | None = None
        self._last_open_directory = _initial_open_directory()
        self._parts_actor: Any | None = None
        self._numeric_char_observer: int | None = None
        self._cell_press_observer: int | None = None
        self._cell_release_observer: int | None = None
        self._cell_click_start: tuple[int, int] | None = None
        self._cell_click_picker = vtkCellPicker()
        self._cell_click_picker.SetTolerance(0.005)
        self.element_id_array_name = self._find_element_id_array_name(mesh)
        self.element_ids = self._read_element_ids(mesh)
        self.part_array_name = self._find_part_array_name(mesh)
        self.part_ids = self._read_part_ids(mesh)

        self._validate_inputs()

        finite_values = self.scalar_series[
            np.isfinite(self.scalar_series)
        ]

        if finite_values.size == 0:
            raise ValueError(
                "scalar_series contains no finite values"
            )

        self.global_clim = (
            float(np.min(finite_values)),
            float(np.max(finite_values)),
        )

        if self.global_clim[0] == self.global_clim[1]:
            self.global_clim = (
                self.global_clim[0],
                self.global_clim[0] + 1.0,
            )

        default_threshold = float(
            np.nanpercentile(self.scalar_series, 90)
        )

        self.state = UIState(
            threshold=(
                default_threshold
                if initial_threshold is None
                else float(initial_threshold)
            ),
            show_simulation_results=self.data_is_simulated,
        )
        if not np.isfinite(self.state.threshold):
            raise ValueError("initial_threshold must be finite")

        self.cell_centers = (
            self.mesh.cell_centers().points
        )

        plotter = pv.Plotter(
            shape=(3, 2),
            groups=[([0, 1, 2], [0])],
            row_weights=(1.0, 1.0, 1.0),
            col_weights=(
                _SLICE_PANEL_SPLIT,
                1.0 - _SLICE_PANEL_SPLIT,
            ),
            off_screen=off_screen,
            window_size=window_size,
            border=True,
            border_color="white",
            border_width=1.0,
        )
        self.visualizer = MeshVisualisation(
            self.mesh_frames,
            self.field_name,
            scalar_frames=self.scalar_series,
            association="cell",
            times=self.times,
            plotter=plotter,
            title=f"Brain strain — {self.field_name}",
            cmap="viridis",
        )
        self.plotter = self.visualizer.plotter
        self._slice_panel_renderers = tuple(self.plotter.renderers[1:])
        self._slice_panel_viewports = tuple(
            tuple(renderer.viewport)
            for renderer in self._slice_panel_renderers
        )
        self.plotter.renderers[0].viewport = (0.0, 0.0, 1.0, 1.0)
        for renderer in self._slice_panel_renderers:
            renderer.SetDraw(False)
        self.plotter.subplot(*_MAIN_RENDERER_LOCATION)
        self.plotter.theme.font.color = _UI_FONT_COLOR
        self.plotter.theme.font.size = _UI_LABEL_FONT_SIZE
        self._parameter_window_off_screen = bool(off_screen)
        self._parameter_window_visible = False
        self.parameter_plotter = self._create_parameter_plotter()
        self._result_window_off_screen = bool(off_screen)
        self._result_window_visible = False
        self._result_export_status = (
            "Choose Export Excel to save this result snapshot."
        )
        self._last_result_directory = Path.cwd()
        self.result_plotter = self._create_result_plotter()

        self.peak = find_global_peak(
            self.scalar_series,
            self.times,
            self.cell_centers,
        )

        self._initialise_visualization()
        self._build_widgets()
        if enable_picking:
            self._enable_picking()
        self._callbacks_active = True
        self._refresh_scene()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_inputs(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if self.times.size == 0:
            raise ValueError("times cannot be empty")
        if not np.isfinite(self.times).all():
            raise ValueError("times cannot contain NaN or infinity")

        if self.scalar_series.ndim != 2:
            raise ValueError(
                "scalar_series must have shape "
                "(n_times, n_cells)"
            )

        expected_shape = (
            self.times.size,
            self.mesh.n_cells,
        )

        if self.scalar_series.shape != expected_shape:
            raise ValueError(
                f"Expected scalar_series shape "
                f"{expected_shape}, received "
                f"{self.scalar_series.shape}"
            )

        if np.any(np.diff(self.times) <= 0):
            raise ValueError(
                "times must be strictly increasing"
            )

    @property
    def has_real_results(self) -> bool:
        """Return whether a genuine embedded or supplied scalar series exists."""
        return (
            self._real_times is not None
            and self._real_scalar_series is not None
        )

    def _ensure_simulated_results(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Create the generalized-Maxwell series when the user requests it."""
        if (
            self._simulated_times is not None
            and self._simulated_scalar_series is not None
        ):
            return self._simulated_times, self._simulated_scalar_series

        frame_count = self._simulation_frame_count
        use_real_times = frame_count is None and self.has_real_results
        if frame_count is None:
            frame_count = (
                int(self._real_scalar_series.shape[0])
                if self._real_scalar_series is not None
                else DEFAULT_SIMULATION_FRAME_COUNT
            )
        generated_times, generated_series = simulate_generalized_maxwell_strain(
            self.mesh,
            frame_count=frame_count,
            duration=self._simulation_duration,
            times=self._real_times if use_real_times else None,
            model=self._simulation_model,
            impact_mode=self._simulation_impact_mode,
            target_mean_maximum_shear_strain=self._simulation_target_mean_mss,
            rotation_axis=self._simulation_rotation_axis,
        )
        self._simulated_times = generated_times
        self._simulated_scalar_series = generated_series
        return generated_times, generated_series

    def _configure_result_sliders(self) -> None:
        """Update both slider ranges after changing the active data source."""
        final_frame = self.times.size - 1
        time_representation = self.time_slider.GetRepresentation()
        time_representation.SetMinimumValue(0.0)
        time_representation.SetMaximumValue(float(max(final_frame, 1)))
        time_representation.SetValue(0.0)
        time_representation.SetTitleText(
            "Time frame (1 total)"
            if final_frame == 0
            else f"Time frame ({self.times.size} total)"
        )
        self.time_slider.SetProcessEvents(final_frame != 0)

        threshold_representation = self.threshold_slider.GetRepresentation()
        threshold_representation.SetMinimumValue(self.global_clim[0])
        threshold_representation.SetMaximumValue(self.global_clim[1])
        threshold_representation.SetValue(self.state.threshold)
        time_representation.Modified()
        threshold_representation.Modified()

    def _update_data_source_badge(self, *, render: bool = True) -> None:
        """Keep a prominent, unambiguous real/simulated label on screen."""
        label = (
            "SIMULATED DATA — DEMONSTRATION ONLY"
            if self.data_is_simulated
            else "REAL RESULT DATA"
        )
        color = "darkorange" if self.data_is_simulated else "mediumseagreen"
        self.plotter.add_text(
            label,
            position="lower_right",
            font_size=_UI_STATUS_FONT_SIZE,
            color=color,
            name="data_source_badge",
            render=False,
        )
        if render:
            self._render()

    def _activate_results(
        self,
        times: npt.ArrayLike,
        scalar_series: npt.ArrayLike,
        *,
        simulated: bool,
    ) -> None:
        """Replace the active values and refresh all source-dependent state."""
        self.times = np.asarray(times, dtype=np.float64)
        self.scalar_series = np.asarray(scalar_series, dtype=np.float64)
        self._validate_inputs()
        finite_values = self.scalar_series[np.isfinite(self.scalar_series)]
        if finite_values.size == 0:
            raise ValueError("scalar_series contains no finite values")

        self.global_clim = (
            float(np.min(finite_values)),
            float(np.max(finite_values)),
        )
        if self.global_clim[0] == self.global_clim[1]:
            self.global_clim = (
                self.global_clim[0],
                self.global_clim[0] + 1.0,
            )
        self.data_is_simulated = simulated
        next_field_name = (
            "Maximum shear strain" if simulated else self._real_field_name
        )
        if next_field_name != self.field_name:
            self._remove_scalar_bar(self.field_name)
            self.field_name = next_field_name
            self.visualizer.title = f"Brain strain — {self.field_name}"
        self.state.show_simulation_results = simulated
        self.state.time_index = 0
        self.state.threshold = float(
            np.nanpercentile(self.scalar_series, 90)
        )
        self.peak = find_global_peak(
            self.scalar_series,
            self.times,
            self.cell_centers,
        )

        self.visualizer.replace_scalar_frames(
            self.scalar_series,
            times=self.times,
            scalar_name=self.field_name,
            frame_index=0,
            render=False,
        )
        self.visualizer.set_scalar_range(self.global_clim, render=False)
        self._configure_result_sliders()
        self._set_toggle_button_state(self.simulation_button, simulated)
        self._update_data_source_badge(render=False)
        self._sync_maxwell_widgets()
        self._refresh_scene()

    def show_simulation_results(self, enabled: bool = True) -> bool:
        """Show simulated results, or return to real data when available."""
        requested = bool(enabled)
        if requested:
            if not self.data_is_simulated:
                times, series = self._ensure_simulated_results()
                self._activate_results(times, series, simulated=True)
            return True

        if not self.has_real_results:
            self.state.show_simulation_results = True
            self._set_toggle_button_state(self.simulation_button, True)
            self._update_data_source_badge(render=False)
            self._update_file_status(
                "No real result field is available; showing simulated data",
                render=False,
            )
            self._update_status_text()
            self._render()
            return True

        assert self._real_times is not None
        assert self._real_scalar_series is not None
        if self.data_is_simulated:
            self._activate_results(
                self._real_times,
                self._real_scalar_series,
                simulated=False,
            )
        self._close_parameter_window()
        return False

    # --------------------------------------------------
    # Scene and widgets
    # --------------------------------------------------

    def _initialise_visualization(self) -> None:
        self.visualizer.set_scalar_range(
            self.global_clim,
            render=False,
        )
        self.visualizer.initialize_user_view(render=False)
        # Trackball camera mode provides left-drag rotation and mouse-wheel
        # zoom whenever cell-selection mode is not active.
        self.plotter.enable_trackball_style()
        self.plotter.render_window.StereoRenderOff()
        self._numeric_char_observer = (
            self.plotter.iren.interactor.AddObserver(
                "CharEvent",
                self._suppress_vtk_ui_char,
                1.0,
            )
        )

    @staticmethod
    def _suppress_vtk_ui_char(
        interactor: Any,
        _event: str,
    ) -> None:
        """Keep handled UI keys out of VTK's built-in character shortcuts."""
        key_sym = interactor.GetKeySym()
        key_code = interactor.GetKeyCode()
        if (
            key_sym not in _UI_CAPTURED_CHAR_KEYS
            and key_code not in _UI_CAPTURED_CHAR_KEYS
        ):
            return

        render_window = interactor.GetRenderWindow()
        if render_window is not None and (
            key_sym in _CELL_ID_DIGITS or key_code in _CELL_ID_DIGITS
        ):
            render_window.StereoRenderOff()
        # The UI has already handled the preceding KeyPressEvent. Replace the
        # character before the interactor style receives CharEvent. Otherwise
        # VTK interprets 3 as stereo rendering and handles R a second time.
        interactor.SetKeyCode("\0")
        interactor.SetKeySym("NoSymbol")

    @staticmethod
    def _style_slider_text(slider: Any) -> None:
        representation = slider.GetRepresentation()
        representation.SetTitleHeight(_UI_SLIDER_TITLE_HEIGHT)
        representation.SetLabelHeight(_UI_SLIDER_LABEL_HEIGHT)
        representation.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        representation.GetLabelProperty().SetColor(1.0, 1.0, 1.0)

    @staticmethod
    def _style_maxwell_slider(slider: Any) -> None:
        """Style a slider in the dedicated Maxwell control window."""
        representation = slider.GetRepresentation()
        representation.SetTitleHeight(_MAXWELL_SLIDER_TITLE_HEIGHT)
        representation.SetLabelHeight(_MAXWELL_SLIDER_LABEL_HEIGHT)
        representation.SetSliderLength(_MAXWELL_SLIDER_LENGTH)
        representation.SetSliderWidth(_MAXWELL_SLIDER_WIDTH)
        representation.SetTubeWidth(_MAXWELL_TUBE_WIDTH)
        representation.SetEndCapLength(_MAXWELL_END_CAP_LENGTH)
        representation.SetEndCapWidth(_MAXWELL_END_CAP_WIDTH)
        representation.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        representation.GetLabelProperty().SetColor(1.0, 1.0, 1.0)

    def _create_parameter_plotter(self) -> pv.Plotter:
        """Create the initially hidden simulation-parameter window."""
        self._parameter_layout_size: tuple[int, int] | None = None
        self._parameter_text_base_sizes: dict[str, int] = {}
        self._parameter_resize_observers: list[int] = []
        plotter = pv.Plotter(
            off_screen=self._parameter_window_off_screen,
            window_size=_MAXWELL_WINDOW_SIZE,
            title="Simulation parameters",
        )
        plotter.set_background(_MAXWELL_WINDOW_BACKGROUND)
        plotter.theme.font.color = _UI_FONT_COLOR
        plotter.theme.font.size = _UI_LABEL_FONT_SIZE
        return plotter

    def _create_result_plotter(self) -> pv.Plotter:
        """Create the initially hidden result summary/export window."""
        plotter = pv.Plotter(
            off_screen=self._result_window_off_screen,
            window_size=_RESULT_WINDOW_SIZE,
            title="Result output",
        )
        plotter.set_background(_RESULT_WINDOW_BACKGROUND)
        plotter.theme.font.color = _UI_FONT_COLOR
        plotter.theme.font.size = _UI_LABEL_FONT_SIZE
        return plotter

    def _build_result_widgets(self) -> None:
        """Build the extendable result summary and Excel export action."""
        title = self.result_plotter.add_text(
            "Result output",
            position=(0.5, 0.92),
            viewport=True,
            font_size=20,
            color=_UI_FONT_COLOR,
            name="result_window_title",
            render=False,
        )
        title.GetTextProperty().SetJustificationToCentered()
        sections = self.result_plotter.add_text(
            (
                "Excel sections\n"
                "Results Summary  •  Parameters  •  Frame Results\n"
                "Selected Cell Results  •  Data Dictionary"
            ),
            position=(0.5, 0.67),
            viewport=True,
            font_size=_UI_DETAIL_FONT_SIZE,
            color="lightsteelblue",
            name="result_window_sections",
            render=False,
        )
        sections.GetTextProperty().SetJustificationToCentered()
        self.result_export_button = (
            self.result_plotter.add_checkbox_button_widget(
                callback=self._on_result_export_button,
                value=False,
                position=(35, 35),
                size=44,
                color_on="mediumseagreen",
                color_off="grey",
            )
        )
        self.result_plotter.add_text(
            "Export Excel (.xlsx)",
            position=(94, 45),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="result_export_label",
            render=False,
        )
        self._update_result_window_text(render=False)

    def _update_result_window_text(self, *, render: bool = True) -> None:
        """Refresh result metrics without calculating a missing simulation."""
        if not hasattr(self, "result_plotter") or self.result_plotter._closed:
            return
        real_peak = (
            float(np.nanmax(self._real_scalar_series))
            if self._real_scalar_series is not None
            else None
        )
        simulated_peak = (
            float(np.nanmax(self._simulated_scalar_series))
            if self._simulated_scalar_series is not None
            else None
        )
        active_label = "SIMULATED" if self.data_is_simulated else "REAL"
        lines = [
            f"Active source: {active_label}",
            f"Active field: {self.field_name}",
            f"Active frame: {self.state.time_index}",
            f"Active time: {self.times[self.state.time_index]:.6g} s",
            f"Active global maximum: {self.peak.value:.6g}",
            (
                f"Actual real-data maximum: {real_peak:.6g}"
                if real_peak is not None
                else "Actual real-data maximum: unavailable"
            ),
            (
                f"Calculated maximum strain: {simulated_peak:.6g}"
                if simulated_peak is not None
                else "Calculated maximum strain: calculated during export"
            ),
        ]
        self.result_plotter.add_text(
            "\n".join(lines),
            position=(0.08, 0.33),
            viewport=True,
            font_size=_UI_STATUS_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="result_window_summary",
            render=False,
        )
        self.result_plotter.add_text(
            self._result_export_status,
            position=(0.08, 0.13),
            viewport=True,
            font_size=_UI_DETAIL_FONT_SIZE,
            color=(
                "lightcoral"
                if self._result_export_status.startswith("Export failed")
                else "lightgreen"
            ),
            name="result_window_status",
            render=False,
        )
        if render and self._result_window_visible:
            self.result_plotter.render()

    def _add_maxwell_slider(
        self,
        *,
        callback: Any,
        value_range: tuple[float, float],
        value: float,
        title: str,
        x_range: tuple[float, float],
        y_position: float,
        fmt: str,
    ) -> Any:
        slider = self.parameter_plotter.add_slider_widget(
            callback=callback,
            rng=value_range,
            value=value,
            title=title,
            pointa=(x_range[0], y_position),
            pointb=(x_range[1], y_position),
            fmt=fmt,
            interaction_event="end",
        )
        self._style_maxwell_slider(slider)
        return slider

    @staticmethod
    def _set_slider_representation(
        slider: Any,
        *,
        minimum: float,
        maximum: float,
        value: float,
        title: str,
    ) -> None:
        representation = slider.GetRepresentation()
        representation.SetMinimumValue(float(minimum))
        representation.SetMaximumValue(float(maximum))
        representation.SetValue(float(value))
        representation.SetTitleText(title)
        representation.Modified()

    @staticmethod
    def _set_slider_visible(slider: Any, visible: bool) -> None:
        slider.SetProcessEvents(bool(visible))
        slider.SetEnabled(bool(visible))
        representation = slider.GetRepresentation()
        representation.SetVisibility(bool(visible))
        representation.Modified()

    @staticmethod
    def _maxwell_g_limits(model: GeneralizedMaxwellModel) -> tuple[float, float]:
        transient_fraction = 1.0 - model.equilibrium_ratio
        epsilon = min(
            1e-6,
            transient_fraction / (1000.0 * model.branch_count),
        )
        maximum = transient_fraction - epsilon * (model.branch_count - 1)
        return epsilon, maximum

    def _build_maxwell_widgets(self) -> None:
        """Build interactive Prony-parameter sliders for simulated results."""
        model = self._simulation_model
        primary_y = _MAXWELL_PRIMARY_SLIDER_Y_POSITIONS
        tau_y = _MAXWELL_TAU_SLIDER_Y_POSITIONS
        window_title = self.parameter_plotter.add_text(
            "Generalized Maxwell model",
            position=(0.5, 0.95),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="maxwell_window_title",
            viewport=True,
            render=False,
        )
        window_title.GetTextProperty().SetJustificationToCentered()
        primary_heading = self.parameter_plotter.add_text(
            "Model parameters",
            position=(0.19, 0.76),
            font_size=_UI_DETAIL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="maxwell_primary_heading",
            viewport=True,
            render=False,
        )
        primary_heading.GetTextProperty().SetJustificationToCentered()
        fraction_heading = self.parameter_plotter.add_text(
            "Material fractions",
            position=(0.25, 0.515),
            font_size=_UI_DETAIL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="maxwell_fraction_heading",
            viewport=True,
            render=False,
        )
        fraction_heading.GetTextProperty().SetJustificationToCentered()
        tau_heading = self.parameter_plotter.add_text(
            "Relaxation times",
            position=(0.75, 0.515),
            font_size=_UI_DETAIL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="maxwell_tau_heading",
            viewport=True,
            render=False,
        )
        tau_heading.GetTextProperty().SetJustificationToCentered()
        self.maxwell_branch_slider = self._add_maxwell_slider(
            callback=self._on_maxwell_branch_slider,
            value_range=(3.0, 6.0),
            value=float(model.branch_count),
            title="Maxwell branches N",
            x_range=_MAXWELL_PRIMARY_SLIDER_X_RANGE,
            y_position=primary_y[0],
            fmt="%.0f",
        )
        self.maxwell_modulus_slider = self._add_maxwell_slider(
            callback=self._on_maxwell_modulus_slider,
            value_range=(
                0.5 * model.estimated_instantaneous_modulus,
                2.0 * model.estimated_instantaneous_modulus,
            ),
            value=(
                model.estimated_instantaneous_modulus * model.modulus_scale
            ),
            title=model.modulus_kind,
            x_range=_MAXWELL_PRIMARY_SLIDER_X_RANGE,
            y_position=primary_y[1],
            fmt="%.3g",
        )
        self.maxwell_equilibrium_slider = self._add_maxwell_slider(
            callback=self._on_maxwell_equilibrium_slider,
            value_range=(0.001, 0.999),
            value=model.equilibrium_ratio,
            title="r_inf",
            x_range=_MAXWELL_PRIMARY_SLIDER_X_RANGE,
            y_position=primary_y[2],
            fmt="%.3f",
        )
        self.maxwell_g_sliders = [
            self._add_maxwell_slider(
                callback=(
                    lambda value, index=index: self._on_maxwell_g_slider(
                        index, value
                    )
                ),
                value_range=(1e-6, 1.0),
                value=(
                    model.branch_fractions[index]
                    if index < model.branch_count
                    else 1e-6
                ),
                title=f"g{index + 1}",
                x_range=_MAXWELL_PRIMARY_SLIDER_X_RANGE,
                y_position=primary_y[index + 3],
                fmt="%.3g",
            )
            for index in range(6)
        ]
        self.maxwell_tau_sliders = [
            self._add_maxwell_slider(
                callback=(
                    lambda value, index=index: self._on_maxwell_tau_slider(
                        index, value
                    )
                ),
                value_range=self._maxwell_log10_tau_bounds,
                value=(
                    float(np.log10(model.relaxation_times[index]))
                    if index < model.branch_count
                    else self._maxwell_log10_tau_bounds[0]
                ),
                title=f"log10 tau{index + 1} (s)",
                x_range=_MAXWELL_TAU_SLIDER_X_RANGE,
                y_position=tau_y[index],
                fmt="%.2f",
            )
            for index in range(6)
        ]
        self._sync_maxwell_widgets()
        interactor = self.parameter_plotter.iren
        if interactor is not None:
            self._parameter_resize_observers.append(
                interactor.interactor.AddObserver(
                    "ConfigureEvent",
                    self._on_parameter_window_resize,
                )
            )
        render_window = self.parameter_plotter.render_window
        if render_window is not None:
            self._parameter_resize_observers.append(
                render_window.AddObserver(
                    "WindowResizeEvent",
                    self._on_parameter_window_resize,
                )
            )

    def _remember_parameter_text_sizes(self) -> None:
        """Record the native-DPI font sizes used by parameter-window text."""
        for name in (
            "maxwell_window_title",
            "maxwell_primary_heading",
            "maxwell_fraction_heading",
            "maxwell_tau_heading",
            "maxwell_window_status",
        ):
            actor = self.parameter_plotter.renderer.actors.get(name)
            if actor is not None and hasattr(actor, "GetTextProperty"):
                self._parameter_text_base_sizes.setdefault(
                    name,
                    int(actor.GetTextProperty().GetFontSize()),
                )

    @staticmethod
    def _set_slider_position(
        slider: Any,
        x_range: tuple[float, float],
        y_position: float,
    ) -> None:
        representation = slider.GetRepresentation()
        point1 = representation.GetPoint1Coordinate()
        point2 = representation.GetPoint2Coordinate()
        point1.SetCoordinateSystemToNormalizedDisplay()
        point2.SetCoordinateSystemToNormalizedDisplay()
        point1.SetValue(
            x_range[0],
            y_position,
        )
        point2.SetValue(
            x_range[1],
            y_position,
        )
        representation.Modified()

    def _on_parameter_window_resize(
        self,
        _interactor: Any,
        _event: str,
    ) -> None:
        """Update the parameter controls after a native window resize."""
        self._update_parameter_window_layout()

    def _update_parameter_window_layout(
        self,
        *,
        force: bool = False,
    ) -> bool:
        """Scale and reflow parameter controls for the current window size."""
        render_window = self.parameter_plotter.render_window
        if render_window is None or self.maxwell_branch_slider is None:
            return False
        width, height = map(int, render_window.GetActualSize())
        if width <= 0 or height <= 0:
            return False
        if not force and self._parameter_layout_size == (width, height):
            return False
        self._parameter_layout_size = (width, height)
        self.parameter_plotter.renderer.viewport = (0.0, 0.0, 1.0, 1.0)

        base_width, base_height = _MAXWELL_WINDOW_SIZE
        text_scale = float(
            np.clip(
                min(width / base_width, height / base_height),
                _MAXWELL_MIN_TEXT_SCALE,
                _MAXWELL_MAX_TEXT_SCALE,
            )
        )
        width_adjustment = base_width * text_scale / width
        height_adjustment = base_height * text_scale / height
        all_sliders = (
            self.maxwell_branch_slider,
            self.maxwell_modulus_slider,
            self.maxwell_equilibrium_slider,
            *self.maxwell_g_sliders,
            *self.maxwell_tau_sliders,
        )
        for slider in all_sliders:
            representation = slider.GetRepresentation()
            representation.SetTitleHeight(
                _MAXWELL_SLIDER_TITLE_HEIGHT * height_adjustment
            )
            representation.SetLabelHeight(
                _MAXWELL_SLIDER_LABEL_HEIGHT * height_adjustment
            )
            representation.SetSliderLength(
                _MAXWELL_SLIDER_LENGTH * width_adjustment
            )
            representation.SetSliderWidth(
                _MAXWELL_SLIDER_WIDTH * height_adjustment
            )
            representation.SetTubeWidth(
                _MAXWELL_TUBE_WIDTH * height_adjustment
            )
            representation.SetEndCapLength(
                _MAXWELL_END_CAP_LENGTH * width_adjustment
            )
            representation.SetEndCapWidth(
                _MAXWELL_END_CAP_WIDTH * height_adjustment
            )
            representation.Modified()

        actors = self.parameter_plotter.renderer.actors
        self._remember_parameter_text_sizes()
        for name, base_size in self._parameter_text_base_sizes.items():
            actor = actors.get(name)
            if actor is not None and hasattr(actor, "GetTextProperty"):
                actor.GetTextProperty().SetFontSize(
                    max(8, int(round(base_size * text_scale)))
                )
                actor.Modified()

        aspect_ratio = width / height
        narrow_layout = aspect_ratio < _MAXWELL_NARROW_ASPECT_RATIO
        desktop_layout = (
            not narrow_layout
            and aspect_ratio >= _MAXWELL_DESKTOP_ASPECT_RATIO
            and width >= _MAXWELL_DESKTOP_MIN_WIDTH
        )
        primary_heading = actors.get("maxwell_primary_heading")
        fraction_heading = actors.get("maxwell_fraction_heading")
        tau_heading = actors.get("maxwell_tau_heading")
        status = actors.get("maxwell_window_status")
        if primary_heading is not None:
            primary_heading.SetVisibility(desktop_layout)
        for heading in (fraction_heading, tau_heading):
            if heading is not None:
                heading.SetVisibility(not narrow_layout)
        if status is not None:
            status.SetVisibility(not narrow_layout)

        if narrow_layout:
            full_width = (0.10, 0.90)
            y_positions = np.linspace(0.88, 0.10, 15)
            primary_y = y_positions[:3]
            g_y = y_positions[3::2]
            tau_y = y_positions[4::2]
            primary_x = full_width
            g_x = full_width
            tau_x = full_width
        elif desktop_layout:
            outer_margin = 0.06
            column_gap = 0.05
            column_width = (
                1.0 - 2.0 * outer_margin - 2.0 * column_gap
            ) / 3.0
            primary_x = (
                outer_margin,
                outer_margin + column_width,
            )
            g_x = (
                primary_x[1] + column_gap,
                primary_x[1] + column_gap + column_width,
            )
            tau_x = (
                g_x[1] + column_gap,
                1.0 - outer_margin,
            )
            primary_y = (0.60, 0.45, 0.30)
            g_y = np.linspace(0.68, 0.18, 6)
            tau_y = g_y
            heading_y = 0.77
            for heading, heading_x in (
                (primary_heading, sum(primary_x) / 2.0),
                (fraction_heading, sum(g_x) / 2.0),
                (tau_heading, sum(tau_x) / 2.0),
            ):
                if heading is not None:
                    heading_position = heading.GetPositionCoordinate()
                    heading_position.SetCoordinateSystemToNormalizedViewport()
                    heading.SetPosition(heading_x, heading_y)
        else:
            horizontal_padding = max(0.04, min(0.10, 28.0 / width))
            column_gap = max(0.08, min(0.18, 60.0 / width))
            primary_x = (
                horizontal_padding,
                0.5 - column_gap / 2.0,
            )
            tau_x = (
                0.5 + column_gap / 2.0,
                1.0 - horizontal_padding,
            )
            primary_y = _MAXWELL_PRIMARY_SLIDER_Y_POSITIONS[:3]
            g_y = _MAXWELL_PRIMARY_SLIDER_Y_POSITIONS[3:]
            tau_y = _MAXWELL_TAU_SLIDER_Y_POSITIONS
            g_x = primary_x
            if fraction_heading is not None:
                fraction_position = fraction_heading.GetPositionCoordinate()
                fraction_position.SetCoordinateSystemToNormalizedViewport()
                fraction_heading.SetPosition(
                    sum(primary_x) / 2.0,
                    0.515,
                )
            if tau_heading is not None:
                tau_position = tau_heading.GetPositionCoordinate()
                tau_position.SetCoordinateSystemToNormalizedViewport()
                tau_heading.SetPosition(sum(tau_x) / 2.0, 0.515)

        for slider, y_position in zip(
            (
                self.maxwell_branch_slider,
                self.maxwell_modulus_slider,
                self.maxwell_equilibrium_slider,
            ),
            primary_y,
            strict=True,
        ):
            self._set_slider_position(slider, primary_x, float(y_position))
        for slider, y_position in zip(
            self.maxwell_g_sliders,
            g_y,
            strict=True,
        ):
            self._set_slider_position(slider, g_x, float(y_position))
        for slider, y_position in zip(
            self.maxwell_tau_sliders,
            tau_y,
            strict=True,
        ):
            self._set_slider_position(slider, tau_x, float(y_position))
        return True

    def _sync_maxwell_widgets(self) -> None:
        """Synchronize slider values, ranges, and active-branch visibility."""
        panel_visible = self.data_is_simulated
        parameter_button = getattr(
            self,
            "parameter_window_button",
            None,
        )
        if parameter_button is not None:
            parameter_button.SetProcessEvents(panel_visible)
            parameter_button.SetEnabled(panel_visible)
        if self.maxwell_branch_slider is None:
            return
        if self.parameter_plotter._closed:
            return
        model = self._simulation_model
        self._maxwell_widgets_updating = True
        try:
            assert self.maxwell_modulus_slider is not None
            assert self.maxwell_equilibrium_slider is not None
            self._set_slider_representation(
                self.maxwell_branch_slider,
                minimum=3.0,
                maximum=6.0,
                value=float(model.branch_count),
                title="Maxwell branches N",
            )
            self._set_slider_representation(
                self.maxwell_modulus_slider,
                minimum=0.5 * model.estimated_instantaneous_modulus,
                maximum=2.0 * model.estimated_instantaneous_modulus,
                value=(
                    model.estimated_instantaneous_modulus
                    * model.modulus_scale
                ),
                title=model.modulus_kind,
            )
            self._set_slider_representation(
                self.maxwell_equilibrium_slider,
                minimum=0.001,
                maximum=0.999,
                value=model.equilibrium_ratio,
                title="r_inf",
            )
            status = self.parameter_plotter.add_text(
                (
                    "Adjust a parameter to recompute the simulated response."
                    if panel_visible
                    else "Switch on simulation results to edit parameters."
                ),
                position=(0.5, 0.035),
                font_size=_UI_DETAIL_FONT_SIZE,
                color=("lightgray" if panel_visible else "darkorange"),
                name="maxwell_window_status",
                viewport=True,
                render=False,
            )
            status.GetTextProperty().SetJustificationToCentered()
            for slider in (
                self.maxwell_branch_slider,
                self.maxwell_modulus_slider,
                self.maxwell_equilibrium_slider,
            ):
                self._set_slider_visible(slider, panel_visible)

            g_minimum, g_maximum = self._maxwell_g_limits(model)
            log_taus = np.log10(model.relaxation_times)
            global_minimum, global_maximum = self._maxwell_log10_tau_bounds
            for index, (g_slider, tau_slider) in enumerate(
                zip(
                    self.maxwell_g_sliders,
                    self.maxwell_tau_sliders,
                    strict=True,
                )
            ):
                active = index < model.branch_count
                g_value = (
                    model.branch_fractions[index] if active else g_minimum
                )
                self._set_slider_representation(
                    g_slider,
                    minimum=g_minimum,
                    maximum=g_maximum,
                    value=g_value,
                    title=f"g{index + 1}",
                )
                if active:
                    tau_minimum = (
                        global_minimum
                        if index == 0
                        else float(log_taus[index - 1])
                        + _MAXWELL_LOG10_TAU_MIN_GAP
                    )
                    tau_maximum = (
                        global_maximum
                        if index == model.branch_count - 1
                        else float(log_taus[index + 1])
                        - _MAXWELL_LOG10_TAU_MIN_GAP
                    )
                    tau_value = float(log_taus[index])
                else:
                    tau_minimum, tau_maximum = (
                        global_minimum,
                        global_maximum,
                    )
                    tau_value = global_minimum
                self._set_slider_representation(
                    tau_slider,
                    minimum=tau_minimum,
                    maximum=tau_maximum,
                    value=tau_value,
                    title=f"log10 tau{index + 1} (s)",
                )
                self._set_slider_visible(
                    g_slider, panel_visible and active
                )
                self._set_slider_visible(
                    tau_slider, panel_visible and active
                )
        finally:
            self._maxwell_widgets_updating = False
        self._update_parameter_window_layout(force=True)

    def _set_slider_width(self, *, split_view: bool) -> None:
        x_start, x_end = (
            _SPLIT_SLIDER_X_RANGE
            if split_view
            else _FULL_SLIDER_X_RANGE
        )
        sliders = (
            (self.time_slider, 0.92),
            (self.threshold_slider, 0.84),
        )
        for slider, y_position in sliders:
            if slider is None:
                continue
            representation = slider.GetRepresentation()
            representation.GetPoint1Coordinate().SetValue(
                x_start,
                y_position,
            )
            representation.GetPoint2Coordinate().SetValue(
                x_end,
                y_position,
            )
            representation.Modified()

    def _build_widgets(self) -> None:
        final_frame = self.times.size - 1
        first_column, second_column, third_column = _UI_CONTROL_COLUMNS
        top_row, middle_row, bottom_row = _UI_CONTROL_ROWS
        label_x_offset, label_y_offset = _UI_LABEL_OFFSET

        self.time_slider = self.plotter.add_slider_widget(
            callback=self._on_time_slider,
            rng=(0, max(final_frame, 1)),
            value=0,
            title=(
                "Time frame (1 total)"
                if final_frame == 0
                else f"Time frame ({self.times.size} total)"
            ),
            pointa=(0.35, 0.92),
            pointb=(0.90, 0.92),
            fmt="1" if final_frame == 0 else "%.0f",
            interaction_event="end",
        )
        self._style_slider_text(self.time_slider)
        if final_frame == 0:
            # Keep the one-frame bar visible while preventing it from moving
            # to a frame that does not exist.
            self.time_slider.SetProcessEvents(False)

        self.threshold_slider = self.plotter.add_slider_widget(
            callback=self._on_threshold_slider,
            rng=self.global_clim,
            value=self.state.threshold,
            title="Hotspot threshold",
            pointa=(0.35, 0.84),
            pointb=(0.90, 0.84),
            fmt="%.3f",
            interaction_event="end",
        )
        self._style_slider_text(self.threshold_slider)
        self._build_maxwell_widgets()
        self._build_result_widgets()

        self.hotspot_button = self.plotter.add_checkbox_button_widget(
            callback=self._on_hotspot_toggle,
            value=self.state.show_hotspots,
            position=(first_column, bottom_row),
            size=35,
        )

        self.plotter.add_text(
            "Hotspots",
            position=(
                first_column + label_x_offset,
                bottom_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="hotspot_label",
        )

        self.clear_selection_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_clear_selection_button,
                value=False,
                position=(second_column, bottom_row),
                size=35,
                color_on="orange",
                color_off="grey",
            )
        )

        self.plotter.add_text(
            "Clear selection",
            position=(
                second_column + label_x_offset,
                bottom_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="clear_selection_label",
        )

        self.drag_select_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_drag_select_toggle,
                value=self.state.drag_select,
                position=(first_column, middle_row),
                size=35,
                color_on="dodgerblue",
                color_off="grey",
            )
        )

        self.plotter.add_text(
            "Drag select cells",
            position=(
                first_column + label_x_offset,
                middle_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="drag_select_label",
        )
        self._update_cell_id_input(render=False)

        self.open_file_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_open_file_button,
                value=False,
                position=(first_column, top_row),
                size=35,
                color_on="mediumseagreen",
                color_off="grey",
            )
        )

        self.plotter.add_text(
            "Open local mesh",
            position=(
                first_column + label_x_offset,
                top_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="open_file_label",
        )

        self.result_output_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_result_output_button,
                value=False,
                position=(second_column, top_row),
                size=35,
                color_on="mediumseagreen",
                color_off="grey",
            )
        )
        self.plotter.add_text(
            "Open result output",
            position=(
                second_column + label_x_offset,
                top_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="result_output_label",
        )

        self.slices_button = self.plotter.add_checkbox_button_widget(
            callback=self._on_slices_toggle,
            value=self.state.show_slices,
            position=(second_column, middle_row),
            size=35,
            color_on="deepskyblue",
            color_off="grey",
        )

        self.plotter.add_text(
            "Show slices",
            position=(
                second_column + label_x_offset,
                middle_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="slices_label",
        )

        self.parts_button = self.plotter.add_checkbox_button_widget(
            callback=self._on_parts_toggle,
            value=self.state.show_parts,
            position=(third_column, top_row),
            size=35,
            color_on="gold",
            color_off="grey",
        )

        self.plotter.add_text(
            "Show parts",
            position=(
                third_column + label_x_offset,
                top_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="parts_label",
        )

        self.simulation_button = self.plotter.add_checkbox_button_widget(
            callback=self._on_simulation_toggle,
            value=self.state.show_simulation_results,
            position=(third_column, middle_row),
            size=35,
            color_on="darkorange",
            color_off="grey",
        )

        self.plotter.add_text(
            "Show simulation results",
            position=(
                third_column + label_x_offset,
                middle_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="simulation_label",
        )

        self.parameter_window_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_parameter_window_button,
                value=False,
                position=(third_column, bottom_row),
                size=35,
                color_on="mediumpurple",
                color_off="grey",
            )
        )
        self.plotter.add_text(
            "Open simulation parameters",
            position=(
                third_column + label_x_offset,
                bottom_row + label_y_offset,
            ),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="simulation_parameters_label",
        )
        self.parameter_window_button.SetProcessEvents(self.data_is_simulated)
        self.parameter_window_button.SetEnabled(self.data_is_simulated)
        if not self.has_real_results:
            # Simulated data is the only available source, so keep the visible
            # switch selected and prevent it from implying a real alternative.
            self.simulation_button.SetProcessEvents(False)

        self._update_data_source_badge(render=False)

        self.plotter.add_key_event(
            "p",
            self.go_to_global_peak,
        )

        self.plotter.add_key_event(
            "s",
            self.save_screenshot,
        )

        self.plotter.add_key_event(
            "e",
            self.show_result_window,
        )

        self.plotter.add_key_event(
            "v",
            self.reset_camera,
        )

        self.plotter.add_key_event(
            "c",
            self.clear_selected_cells,
        )

        self.plotter.add_key_event(
            "plus",
            self.zoom_in,
        )

        self.plotter.add_key_event(
            "equal",
            self.zoom_in,
        )

        self.plotter.add_key_event(
            "minus",
            self.zoom_out,
        )

        self.plotter.add_key_event(
            "o",
            self._on_open_file_shortcut,
        )

        self.plotter.add_key_event(
            "l",
            self.toggle_slices,
        )

        self.plotter.add_key_event(
            "b",
            self.toggle_parts,
        )

        self.plotter.add_key_event(
            "r",
            self.toggle_drag_selection,
        )

        for digit in _CELL_ID_DIGITS:
            self.plotter.add_key_event(
                digit,
                lambda digit=digit: self._append_cell_id_digit(digit),
            )

        self.plotter.add_key_event(
            "BackSpace",
            self._remove_cell_id_digit,
        )

        self.plotter.add_key_event(
            "Return",
            self._submit_cell_id_input,
        )

        self.plotter.add_key_event(
            "KP_Enter",
            self._submit_cell_id_input,
        )

        self.plotter.add_key_event(
            "Escape",
            self._clear_cell_id_input,
        )

    def _enable_picking(self) -> None:
        self.visualizer.configure_pick_handler(
            self._on_cells_selected,
            additive=True,
        )
        self.plotter.enable_rectangle_picking(
            callback=self._on_rectangle_selected,
            start=self.state.drag_select,
            show_message=False,
            show_frustum=False,
        )
        interactor = self.plotter.iren.interactor
        self._cell_press_observer = interactor.AddObserver(
            "LeftButtonPressEvent",
            self._on_cell_pointer_press,
            1.0,
        )
        self._cell_release_observer = interactor.AddObserver(
            "LeftButtonReleaseEvent",
            self._on_cell_pointer_release,
            -1.0,
        )

    def _on_rectangle_selected(self, selection: Any) -> None:
        """Extract original cells intersected by a dragged screen rectangle."""
        extractor = vtkExtractGeometry()
        extractor.SetInputData(self.visualizer.mesh)
        extractor.SetImplicitFunction(selection.frustum)
        extractor.ExtractInsideOn()
        extractor.ExtractBoundaryCellsOn()
        extractor.Update()
        picked = pv.wrap(extractor.GetOutput())
        if picked.n_cells:
            self.visualizer.handle_picked_cells(picked)

    def _on_cell_pointer_press(
        self,
        interactor: Any,
        _event: str,
    ) -> None:
        if not self.state.drag_select:
            self._cell_click_start = None
            return
        x_position, y_position = interactor.GetEventPosition()
        self._cell_click_start = (int(x_position), int(y_position))

    def _on_cell_pointer_release(
        self,
        interactor: Any,
        _event: str,
    ) -> None:
        start = self._cell_click_start
        self._cell_click_start = None
        if not self.state.drag_select or start is None:
            return

        x_position, y_position = map(int, interactor.GetEventPosition())
        distance_squared = (
            (x_position - start[0]) ** 2
            + (y_position - start[1]) ** 2
        )
        if distance_squared > _CELL_CLICK_MAX_DRAG_SQUARED:
            return
        self._pick_cell_at_display_position(x_position, y_position)

    def _pick_cell_at_display_position(
        self,
        x_position: int,
        y_position: int,
    ) -> int | None:
        """Select the original mesh cell under a display coordinate."""
        renderer = self.plotter.iren.get_poked_renderer(
            x_position,
            y_position,
        )
        if renderer is not self.plotter.renderers[0]:
            return None
        if not self._cell_click_picker.Pick(
            x_position,
            y_position,
            0.0,
            renderer,
        ):
            return None

        actor = self._cell_click_picker.GetActor()
        local_cell_id = int(self._cell_click_picker.GetCellId())
        if actor is None or local_cell_id < 0:
            return None
        mapper = actor.GetMapper()
        if mapper is None:
            return None
        dataset = pv.wrap(mapper.GetInputDataObject(0, 0))
        if not isinstance(dataset, pv.DataSet):
            return None
        if local_cell_id >= dataset.n_cells:
            return None

        picked = dataset.extract_cells([local_cell_id])
        original_ids = self.visualizer.picked_original_ids(picked)
        if not original_ids.size:
            return None
        self.visualizer.handle_picked_cells(picked)
        return int(original_ids[0])

    # --------------------------------------------------
    # Slider/button callbacks
    # --------------------------------------------------

    def _apply_maxwell_model(self, model: GeneralizedMaxwellModel) -> None:
        """Store a slider-edited model and regenerate active simulation data."""
        previous_model = self._simulation_model
        previous_index = self.state.time_index
        self._simulation_model = model
        self._sync_maxwell_widgets()
        self._simulated_times = None
        self._simulated_scalar_series = None
        if not self.data_is_simulated:
            self._update_status_text()
            self._render()
            return

        try:
            generated_times, generated_series = (
                simulate_generalized_maxwell_strain(
                    self.mesh,
                    times=self.times,
                    model=model,
                    impact_mode=self._simulation_impact_mode,
                    target_mean_maximum_shear_strain=(
                        self._simulation_target_mean_mss
                    ),
                    rotation_axis=self._simulation_rotation_axis,
                )
            )
        except Exception:
            self._simulation_model = previous_model
            self._sync_maxwell_widgets()
            raise

        self._simulated_times = generated_times
        self._simulated_scalar_series = generated_series
        self._activate_results(
            generated_times,
            generated_series,
            simulated=True,
        )
        restored_index = min(previous_index, self.times.size - 1)
        if restored_index:
            self.state.time_index = restored_index
            self.time_slider.GetRepresentation().SetValue(
                float(restored_index)
            )
            self._refresh_scene()

    def _commit_maxwell_model(self, model: GeneralizedMaxwellModel) -> None:
        try:
            self._apply_maxwell_model(model)
        except (TypeError, ValueError, VisualisationError) as exc:
            self._sync_maxwell_widgets()
            self._update_file_status(str(exc), error=True)

    def _on_maxwell_branch_slider(self, value: float) -> None:
        if not self._callbacks_active or self._maxwell_widgets_updating:
            return
        model = self._simulation_model
        branch_count = int(np.clip(round(value), 3, 6))
        if branch_count == model.branch_count:
            self._sync_maxwell_widgets()
            return
        log_minimum = float(np.log10(model.relaxation_times[0]))
        log_maximum = float(np.log10(model.relaxation_times[-1]))
        relaxation_times = np.logspace(
            log_minimum,
            log_maximum,
            branch_count,
        )
        relaxation_times[0] = model.relaxation_times[0]
        relaxation_times[-1] = model.relaxation_times[-1]
        self._commit_maxwell_model(
            replace(
                model,
                branch_fractions=default_branch_fractions(
                    branch_count, model.equilibrium_ratio
                ),
                relaxation_times=tuple(
                    float(item) for item in relaxation_times
                ),
            )
        )

    def _on_maxwell_modulus_slider(self, value: float) -> None:
        if not self._callbacks_active or self._maxwell_widgets_updating:
            return
        model = self._simulation_model
        scale = float(value) / model.estimated_instantaneous_modulus
        scale = float(np.clip(scale, 0.5, 2.0))
        self._commit_maxwell_model(replace(model, modulus_scale=scale))

    def _on_maxwell_equilibrium_slider(self, value: float) -> None:
        if not self._callbacks_active or self._maxwell_widgets_updating:
            return
        model = self._simulation_model
        ratio = float(np.clip(value, 0.001, 0.999))
        transient = 1.0 - ratio
        current_transient = 1.0 - model.equilibrium_ratio
        branch_fractions = tuple(
            fraction * transient / current_transient
            for fraction in model.branch_fractions
        )
        self._commit_maxwell_model(
            replace(
                model,
                equilibrium_ratio=ratio,
                branch_fractions=branch_fractions,
            )
        )

    def _on_maxwell_g_slider(self, index: int, value: float) -> None:
        if not self._callbacks_active or self._maxwell_widgets_updating:
            return
        model = self._simulation_model
        if index < 0 or index >= model.branch_count:
            self._sync_maxwell_widgets()
            return
        minimum, maximum = self._maxwell_g_limits(model)
        selected = float(np.clip(value, minimum, maximum))
        transient = 1.0 - model.equilibrium_ratio
        remaining = transient - selected
        other_indices = [
            item for item in range(model.branch_count) if item != index
        ]
        old = np.asarray(model.branch_fractions, dtype=np.float64)
        other_weights = old[other_indices]
        free = remaining - minimum * len(other_indices)
        redistributed = (
            np.full(len(other_indices), minimum, dtype=np.float64)
            + free * other_weights / np.sum(other_weights)
        )
        updated = old.copy()
        updated[index] = selected
        updated[other_indices] = redistributed
        # Remove the final floating-point residual while preserving positivity.
        updated[other_indices[-1]] += transient - float(np.sum(updated))
        self._commit_maxwell_model(
            replace(
                model,
                branch_fractions=tuple(float(item) for item in updated),
            )
        )

    def _on_maxwell_tau_slider(self, index: int, value: float) -> None:
        if not self._callbacks_active or self._maxwell_widgets_updating:
            return
        model = self._simulation_model
        if index < 0 or index >= model.branch_count:
            self._sync_maxwell_widgets()
            return
        log_taus = np.log10(model.relaxation_times)
        global_minimum, global_maximum = self._maxwell_log10_tau_bounds
        minimum = (
            global_minimum
            if index == 0
            else float(log_taus[index - 1]) + _MAXWELL_LOG10_TAU_MIN_GAP
        )
        maximum = (
            global_maximum
            if index == model.branch_count - 1
            else float(log_taus[index + 1]) - _MAXWELL_LOG10_TAU_MIN_GAP
        )
        log_taus[index] = float(np.clip(value, minimum, maximum))
        self._commit_maxwell_model(
            replace(
                model,
                relaxation_times=tuple(
                    float(10.0**item) for item in log_taus
                ),
            )
        )

    def _on_time_slider(self, value: float) -> None:
        if not self._callbacks_active:
            return
        time_index = int(round(value))
        time_index = int(
            np.clip(
                time_index,
                0,
                self.times.size - 1,
            )
        )

        self.state.time_index = time_index
        self._refresh_scene()

    def _on_threshold_slider(
        self,
        value: float,
    ) -> None:
        if not self._callbacks_active:
            return
        self.state.threshold = float(value)
        self._refresh_hotspots()
        self._update_status_text()
        self._render()

    def _on_hotspot_toggle(
        self,
        enabled: bool,
    ) -> None:
        if not self._callbacks_active:
            return
        self.state.show_hotspots = bool(enabled)

        if enabled:
            self._refresh_hotspots()
        else:
            self.visualizer.hide_hotspot_cells(render=False)
            self._set_brain_opacity(1.0)

        self._render()

    def _on_simulation_toggle(self, enabled: bool) -> None:
        """Switch between real results and the explicit simulation fallback."""
        if not self._callbacks_active:
            return
        try:
            self.show_simulation_results(bool(enabled))
        except (TypeError, ValueError, VisualisationError) as exc:
            self._set_toggle_button_state(
                self.simulation_button,
                self.data_is_simulated,
            )
            self._update_file_status(str(exc), error=True)

    def _on_parameter_window_button(self, _enabled: bool) -> None:
        """Open the simulation controls as a momentary button action."""
        if not self._callbacks_active or not self.data_is_simulated:
            return
        self.show_parameter_window()
        self._set_toggle_button_state(self.parameter_window_button, False)

    def _on_result_output_button(self, _enabled: bool) -> None:
        """Open the result summary as a momentary button action."""
        if not self._callbacks_active:
            return
        self.show_result_window()
        self._set_toggle_button_state(self.result_output_button, False)

    def _on_result_export_button(self, _enabled: bool) -> None:
        """Choose a workbook destination and export the current snapshot."""
        if not self._callbacks_active:
            return
        try:
            saved = self.export_results()
            self._result_export_status = (
                "Export cancelled."
                if saved is None
                else f"Saved: {saved.name}"
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._result_export_status = f"Export failed: {exc}"
        finally:
            self._set_toggle_button_state(self.result_export_button, False)
            self._update_result_window_text()

    def _on_slices_toggle(
        self,
        enabled: bool,
    ) -> None:
        if not self._callbacks_active:
            return
        self.show_slices(bool(enabled))

    def _on_parts_toggle(
        self,
        enabled: bool,
    ) -> None:
        if not self._callbacks_active:
            return
        try:
            self.show_parts(bool(enabled))
        except (TypeError, ValueError) as exc:
            self.state.show_parts = False
            self._set_toggle_button_state(self.parts_button, False)
            self._update_file_status(str(exc), error=True, render=False)
            self._render()

    def _on_clear_selection_button(
        self,
        _enabled: bool,
    ) -> None:
        if not self._callbacks_active:
            return
        self.clear_selected_cells()
        # Treat the checkbox widget as a momentary push button.
        representation = self.clear_selection_button.GetRepresentation()
        if representation is not None:
            representation.SetState(0)
            representation.Modified()
        self._render()

    def _on_drag_select_toggle(
        self,
        enabled: bool,
    ) -> None:
        if not self._callbacks_active:
            return
        self.set_drag_selection_mode(bool(enabled))

    def _update_cell_id_input(
        self,
        *,
        render: bool = True,
    ) -> None:
        value = self._cell_id_input or "_"
        text = f"Cell index [0..{self.mesh.n_cells - 1}]: {value}"
        if self._cell_id_input_error is not None:
            text += f"\nError: {self._cell_id_input_error}"
        self.plotter.add_text(
            text,
            position=_UI_CELL_INPUT_POSITION,
            font_size=_UI_DETAIL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="cell_id_input",
            render=False,
        )
        if render:
            self._render()

    def _append_cell_id_digit(self, digit: str) -> None:
        if not self._callbacks_active:
            return
        self._cell_id_input += digit
        self._cell_id_input_error = None
        self._update_cell_id_input()

    def _remove_cell_id_digit(self) -> None:
        if not self._callbacks_active:
            return
        self._cell_id_input = self._cell_id_input[:-1]
        self._cell_id_input_error = None
        self._update_cell_id_input()

    def _clear_cell_id_input(self) -> None:
        if not self._callbacks_active:
            return
        self._cell_id_input = ""
        self._cell_id_input_error = None
        self._update_cell_id_input()

    def _submit_cell_id_input(self) -> None:
        if not self._callbacks_active:
            return
        try:
            self.select_cell_by_index(self._cell_id_input)
        except (TypeError, ValueError, IndexError) as exc:
            self._cell_id_input_error = str(exc)
            self._update_cell_id_input()

    def _on_cells_selected(
        self,
        cell_ids: np.ndarray,
    ) -> None:
        newly_selected_ids = np.unique(
            np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        )
        newly_selected_ids = newly_selected_ids[
            (newly_selected_ids >= 0)
            & (newly_selected_ids < self.mesh.n_cells)
        ]
        if newly_selected_ids.size == 0:
            return

        previous_ids = np.asarray(
            self.state.selected_cell_ids,
            dtype=np.int64,
        )
        selected_ids = np.unique(
            np.concatenate((previous_ids, newly_selected_ids))
        )
        cell_id = int(newly_selected_ids[0])
        self.state.selected_cell_id = cell_id
        self.state.selected_cell_ids = tuple(map(int, selected_ids))
        self._cell_id_input = str(cell_id)
        self._cell_id_input_error = None
        self._update_cell_id_input(render=False)

        self.visualizer.highlight_cells(selected_ids, render=False)
        self._update_selection_text(render=False)
        self._update_status_text()
        self._render()

    def _update_selection_text(self, *, render: bool = True) -> None:
        """Display count, identity, and values for the active chosen cell."""
        cell_index = self.state.selected_cell_id
        if cell_index is None:
            return

        history = extract_element_history(
            self.scalar_series,
            cell_index,
        )

        peak_index = int(np.nanargmax(history))
        peak_value = float(history[peak_index])
        current_value = float(
            history[self.state.time_index]
        )

        self.plotter.add_text(
            (
                f"Selected cells: {len(self.state.selected_cell_ids)}\n"
                f"Chosen cell index: {cell_index}\n"
                f"Source element ID: "
                f"{self._source_element_id_text(cell_index)}\n"
                f"Current value: {current_value:.4f}\n"
                f"Chosen cell peak value: {peak_value:.4f}\n"
                f"Cell peak time: "
                f"{self.times[peak_index]:.3f}"
            ),
            position="upper_right",
            font_size=_UI_STATUS_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="selection_text",
            render=False,
        )

        if render:
            self._render()

    # --------------------------------------------------
    # Scene updates
    # --------------------------------------------------

    @staticmethod
    def _find_element_id_array_name(mesh: pv.DataSet) -> str | None:
        """Find the source element-number array, when the format preserves it."""
        names = list(mesh.cell_data.keys())
        for preferred in ("element_id", "Element ID", "eid", "EID"):
            if preferred in names:
                return preferred
        return next(
            (
                name
                for name in names
                if name.casefold().replace("_", "").replace(" ", "")
                in {"elementid", "eid"}
            ),
            None,
        )

    def _read_element_ids(
        self,
        mesh: pv.DataSet,
    ) -> npt.NDArray[np.int64] | None:
        """Return source element numbers aligned with PyVista cell indices."""
        if self.element_id_array_name is None:
            return None
        raw = np.asarray(mesh.cell_data[self.element_id_array_name])
        if raw.ndim != 1 or raw.size != mesh.n_cells:
            raise ValueError(
                f"Cell array {self.element_id_array_name!r} must contain "
                "one element ID per cell"
            )
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Element IDs must be numeric") from exc
        if (
            not np.isfinite(numeric).all()
            or np.any(numeric != np.floor(numeric))
        ):
            raise ValueError("Element IDs must be finite integers")
        return numeric.astype(np.int64)

    def _source_element_id_text(self, cell_index: int | None) -> str:
        """Format a source element ID, or ``NaN`` when none was preserved."""
        if cell_index is None or not 0 <= cell_index < self.mesh.n_cells:
            return "NaN"
        if self.element_ids is not None:
            return str(int(self.element_ids[cell_index]))
        return "NaN"

    @staticmethod
    def _find_part_array_name(mesh: pv.DataSet) -> str | None:
        """Return the preferred cell-data array containing part IDs."""
        names = list(mesh.cell_data.keys())
        for preferred in ("part_id", "Part ID", "pid", "PID"):
            if preferred in names:
                return preferred
        return next(
            (
                name
                for name in names
                if name.casefold().replace("_", "").replace(" ", "")
                in {"partid", "pid"}
            ),
            None,
        )

    def _read_part_ids(self, mesh: pv.DataSet) -> npt.NDArray[np.int64]:
        """Validate and return the sorted part IDs attached to a mesh."""
        if self.part_array_name is None:
            return np.empty(0, dtype=np.int64)
        raw = np.asarray(mesh.cell_data[self.part_array_name])
        if raw.ndim != 1 or raw.size != mesh.n_cells:
            raise ValueError(
                f"Cell array {self.part_array_name!r} must contain one "
                "part ID per cell"
            )
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Part IDs must be numeric") from exc
        if (
            not np.isfinite(numeric).all()
            or np.any(numeric != np.floor(numeric))
        ):
            raise ValueError("Part IDs must be finite integers")
        return np.unique(numeric.astype(np.int64))

    def _part_mesh_options(
        self,
        dataset: pv.DataSet,
        *,
        show_scalar_bar: bool,
    ) -> dict[str, Any]:
        """Attach categorical indices and return discrete part-color options."""
        if self.part_array_name is None or not self.part_ids.size:
            raise ValueError(
                "This mesh has no part_id, Part ID, or PID cell array"
            )
        if self.part_array_name not in dataset.cell_data:
            raise ValueError(
                f"Part array {self.part_array_name!r} was not preserved "
                "in the displayed mesh"
            )

        values = np.asarray(
            dataset.cell_data[self.part_array_name],
            dtype=np.int64,
        )
        part_indices = np.searchsorted(self.part_ids, values)
        valid = part_indices < self.part_ids.size
        if not valid.all():
            raise ValueError("Displayed mesh contains unknown part IDs")
        if not np.array_equal(
            self.part_ids[part_indices],
            values,
        ):
            raise ValueError("Displayed mesh contains unknown part IDs")

        other_index = len(_PART_GROUPS) - 1
        category_ids = np.full(values.shape, other_index, dtype=np.int32)
        for group_index, (_, _, part_ids) in enumerate(
            _PART_GROUPS[:-1]
        ):
            category_ids[np.isin(values, tuple(part_ids))] = group_index
        dataset.cell_data[_PART_CATEGORY_ARRAY] = category_ids

        annotations = {
            float(index): label
            for index, (label, _, _) in enumerate(_PART_GROUPS)
        }
        return {
            "scalars": _PART_CATEGORY_ARRAY,
            "preference": "cell",
            "categories": False,
            "n_colors": len(_PART_GROUPS),
            "cmap": [color for _, color, _ in _PART_GROUPS],
            "clim": (-0.5, float(len(_PART_GROUPS)) - 0.5),
            "interpolate_before_map": False,
            "annotations": annotations,
            "show_scalar_bar": show_scalar_bar,
            "scalar_bar_args": {
                "title": _PARTS_SCALAR_BAR,
                "n_labels": 0,
            },
        }

    def _remove_scalar_bar(self, title: str) -> None:
        try:
            self.plotter.remove_scalar_bar(title, render=False)
        except KeyError:
            pass

    def _remove_parts_actor(self) -> None:
        if self._parts_actor is not None:
            self.plotter.remove_actor(
                self._parts_actor,
                reset_camera=False,
                render=False,
            )
            self._parts_actor = None

    def _remove_slice_panel_actors(self) -> None:
        for index, renderer in enumerate(self._slice_panel_renderers):
            for name in (
                f"{_SLICE_PANEL_ACTOR_PREFIX}{index}",
                f"{_SLICE_PANEL_TITLE_PREFIX}{index}",
            ):
                if name in renderer.actors:
                    renderer.remove_actor(
                        name,
                        reset_camera=False,
                        render=False,
                    )

    def _set_slice_panel_layout(self, visible: bool) -> None:
        main_viewport = (
            (0.0, 0.0, _SLICE_PANEL_SPLIT, 1.0)
            if visible
            else (0.0, 0.0, 1.0, 1.0)
        )
        self.plotter.renderers[0].viewport = main_viewport
        for renderer, viewport in zip(
            self._slice_panel_renderers,
            self._slice_panel_viewports,
            strict=True,
        ):
            renderer.viewport = viewport
            renderer.SetDraw(visible)
        self.plotter.subplot(*_MAIN_RENDERER_LOCATION)
        self._set_slider_width(split_view=visible)

    def _update_slice_panels(
        self,
        slices: list[pv.DataSet],
    ) -> None:
        """Render orthographic, colour-mapped slice views beside the model."""
        self._remove_slice_panel_actors()
        self._set_slice_panel_layout(True)

        for index, (sliced, title, view_method) in enumerate(
            zip(
                slices,
                _SLICE_PANEL_TITLES,
                _SLICE_PANEL_VIEWS,
                strict=True,
            )
        ):
            self.plotter.subplot(*_SLICE_PANEL_LOCATIONS[index])
            if self.state.show_parts:
                mesh_options = self._part_mesh_options(
                    sliced,
                    show_scalar_bar=False,
                )
            else:
                mesh_options = self.visualizer.mesh_options()
                mesh_options.pop("show_edges", None)
                mesh_options["show_scalar_bar"] = False

            self.plotter.add_mesh(
                sliced,
                name=f"{_SLICE_PANEL_ACTOR_PREFIX}{index}",
                show_edges=False,
                pickable=False,
                reset_camera=False,
                render=False,
                **mesh_options,
            )
            self.plotter.add_text(
                title,
                position="upper_left",
                font_size=_UI_DETAIL_FONT_SIZE,
                color=_UI_FONT_COLOR,
                name=f"{_SLICE_PANEL_TITLE_PREFIX}{index}",
                viewport=True,
                render=False,
            )
            self.plotter.set_background(
                _SLICE_PANEL_BACKGROUND,
                all_renderers=False,
            )
            getattr(self.plotter, view_method)(render=False)
            self.plotter.enable_parallel_projection()
            self.plotter.reset_camera(render=False)

        self.plotter.subplot(*_MAIN_RENDERER_LOCATION)

    def _hide_slice_panels(self) -> None:
        self._remove_slice_panel_actors()
        self._set_slice_panel_layout(False)

    def _refresh_parts_actor(self) -> Any:
        self._remove_parts_actor()
        self._remove_scalar_bar(self.field_name)
        self._remove_scalar_bar(_PARTS_SCALAR_BAR)
        self._parts_actor = self.plotter.add_mesh(
            self.visualizer.mesh,
            name=_PARTS_ACTOR,
            show_edges=False,
            pickable=True,
            reset_camera=False,
            render=False,
            **self._part_mesh_options(
                self.visualizer.mesh,
                show_scalar_bar=True,
            ),
        )
        main_actor = self.visualizer.main_actor
        if main_actor is not None:
            main_actor.SetVisibility(False)
            main_actor.SetPickable(False)
        return self._parts_actor

    def _recolor_slices_for_parts(
        self,
        slices: list[pv.DataSet],
    ) -> list[pv.DataSet]:
        """Replace strain-colored slice actors with categorical part colors."""
        settings = self.visualizer.slice_settings
        actors: list[Any] = []
        for index, sliced in enumerate(slices):
            actor = self.plotter.add_mesh(
                sliced,
                name=f"ui-part-slice-{index}",
                opacity=float(settings.get("opacity", 1.0)),
                show_edges=bool(settings.get("show_edges", True)),
                line_width=float(settings.get("line_width", 1.0)),
                pickable=False,
                reset_camera=False,
                render=False,
                **self._part_mesh_options(
                    sliced,
                    show_scalar_bar=False,
                ),
            )
            actors.append(actor)
        self.visualizer.replace_slice_actors(actors)
        self._remove_scalar_bar(self.field_name)
        return slices

    def _refresh_part_slices(self) -> list[pv.DataSet]:
        settings = self.visualizer.slice_settings
        slices = self.visualizer.show_slices(
            render=False,
            **settings,
        )
        slices = self._recolor_slices_for_parts(slices)
        self._update_slice_panels(slices)
        return slices

    def _refresh_scene(self) -> None:
        self.visualizer.update_scalar_frame(
            self.state.time_index,
            render=False,
        )

        if self.state.show_parts:
            self._refresh_parts_actor()
        if self.state.show_slices:
            if self.state.show_parts:
                self._refresh_part_slices()
            else:
                self._update_slice_panels(self.visualizer.slices)

        if self.state.show_hotspots:
            self._refresh_hotspots()

        if self.state.selected_cell_ids:
            selected_ids = np.asarray(
                self.state.selected_cell_ids,
                dtype=np.int64,
            )
            selected_ids = selected_ids[
                selected_ids < self.visualizer.mesh.n_cells
            ]
            self.visualizer.highlight_cells(
                selected_ids,
                render=False,
            )
            self._update_selection_text(render=False)

        self._update_status_text()
        self._render()

    def _refresh_hotspots(self) -> None:
        if not self.state.show_hotspots:
            return

        frame = self.scalar_series[
            self.state.time_index
        ]

        mask = (
            np.isfinite(frame)
            & (frame >= self.state.threshold)
        )

        hotspot_cell_ids = np.flatnonzero(mask)

        if hotspot_cell_ids.size == 0:
            self.visualizer.hide_hotspot_cells(render=False)
            self._set_brain_opacity(1.0)
            return

        self.visualizer.show_hotspot_cells(
            threshold=self.state.threshold,
            max_hotspots=None,
            min_distance=0.0,
            render=False,
        )
        self._set_brain_opacity(0.15)

    def _update_status_text(self) -> None:
        frame = self.scalar_series[
            self.state.time_index
        ]

        current_max = float(np.nanmax(frame))
        hotspot_count = int(
            np.count_nonzero(
                frame >= self.state.threshold
            )
        )

        view = (
            f"View: tissue-coloured parts ({self.part_ids.size} IDs)\n"
            if self.state.show_parts
            else ""
        )
        selected_cell_index = (
            str(self.state.selected_cell_id)
            if self.state.selected_cell_id is not None
            else "NaN"
        )
        source_element_id = self._source_element_id_text(
            self.state.selected_cell_id
        )
        peak_cell_index = "NaN"
        if (
            np.isfinite(self.peak.value)
            and 0 <= self.peak.element_index < self.mesh.n_cells
        ):
            peak_cell_index = str(self.peak.element_index)
        text = (
            view
            + (
                "Data source: SIMULATED (generalized Maxwell, reduced order)\n"
                if self.data_is_simulated
                else "Data source: REAL results\n"
            )
            + (
                f"N={self._simulation_model.branch_count}, "
                f"G0={self._simulation_model.instantaneous_modulus:.4g}, "
                f"r_inf={self._simulation_model.equilibrium_ratio:.4g}\n"
                if self.data_is_simulated
                else ""
            )
            + f"{self.field_name}\n"
            f"Frame: {self.state.time_index}\n"
            f"Time: "
            f"{self.times[self.state.time_index]:.3f}\n"
            f"Selected cells: {len(self.state.selected_cell_ids)}\n"
            f"Chosen cell index: {selected_cell_index}\n"
            f"Source element ID: {source_element_id}\n"
            f"Frame maximum: {current_max:.4f}\n"
            f"Global peak value: {self.peak.value:.4f}\n"
            f"Global peak cell index: {peak_cell_index}\n"
            f"Threshold: {self.state.threshold:.4f}\n"
            f"Cells above threshold: {hotspot_count}"
        )

        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=_UI_STATUS_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="status_text",
            render=False,
        )
        if getattr(self, "_result_window_visible", False):
            self._update_result_window_text(render=False)

    def _set_brain_opacity(
        self,
        opacity: float,
    ) -> None:
        actors = (
            self.visualizer.main_actor,
            self._parts_actor,
        )
        for actor in actors:
            if actor is not None:
                actor.prop.opacity = opacity

    def _render(self) -> None:
        if self._render_enabled:
            self.plotter.render()
            parameter_plotter = getattr(self, "parameter_plotter", None)
            if (
                getattr(self, "_parameter_window_visible", False)
                and parameter_plotter is not None
                and not parameter_plotter._closed
            ):
                parameter_plotter.render()
            result_plotter = getattr(self, "result_plotter", None)
            if (
                getattr(self, "_result_window_visible", False)
                and result_plotter is not None
                and not result_plotter._closed
            ):
                result_plotter.render()

    # --------------------------------------------------
    # Commands
    # --------------------------------------------------

    def show_slices(
        self,
        enabled: bool = True,
    ) -> list[pv.DataSet]:
        """Show or hide three orthogonal slices through the active frame."""
        self.state.show_slices = bool(enabled)
        self._set_toggle_button_state(
            self.slices_button,
            self.state.show_slices,
        )
        if self.state.show_slices:
            slices = self.visualizer.show_slices(render=False)
            if self.state.show_parts:
                slices = self._recolor_slices_for_parts(slices)
            self._update_slice_panels(slices)
        else:
            self.visualizer.hide_slices(render=False)
            self._hide_slice_panels()
            slices = []
        self._render()
        return slices

    def toggle_slices(self) -> None:
        """Toggle the orthogonal slice actors."""
        self.show_slices(not self.state.show_slices)

    def show_parts(
        self,
        enabled: bool = True,
    ) -> npt.NDArray[np.int64]:
        """Color cells by part ID, or restore the active strain colors."""
        requested = bool(enabled)
        if requested and (
            self.part_array_name is None or not self.part_ids.size
        ):
            raise ValueError(
                "This mesh has no part_id, Part ID, or PID cell array"
            )

        self.state.show_parts = requested
        self._set_toggle_button_state(
            self.parts_button,
            self.state.show_parts,
        )

        if self.state.show_parts:
            self._refresh_parts_actor()
            if self.state.show_slices:
                self._refresh_part_slices()
        else:
            self._remove_parts_actor()
            self._remove_scalar_bar(_PARTS_SCALAR_BAR)
            self.visualizer.replace_main_actor()
            if self.state.show_slices:
                settings = self.visualizer.slice_settings
                slices = self.visualizer.show_slices(
                    render=False,
                    **settings,
                )
                self._update_slice_panels(slices)

        opacity = (
            0.15
            if (
                self.state.show_hotspots
                and self.visualizer.hotspot_cell_ids.size
            )
            else 1.0
        )
        self._set_brain_opacity(opacity)
        self._update_status_text()
        self._render()
        return self.part_ids.copy()

    def toggle_parts(self) -> None:
        """Toggle categorical coloring by part ID."""
        try:
            self.show_parts(not self.state.show_parts)
        except (TypeError, ValueError) as exc:
            self.state.show_parts = False
            self._set_toggle_button_state(self.parts_button, False)
            self._update_file_status(str(exc), error=True)

    def set_drag_selection_mode(
        self,
        enabled: bool = True,
    ) -> bool:
        """Enable rectangle cell selection or restore camera rotation."""
        requested = bool(enabled)
        style = self.plotter.iren._style_class
        method_name = "StartSelect" if requested else "StopSelect"
        method = getattr(style, method_name, None)
        if method is None:
            raise RuntimeError(
                "The rectangle cell-selection tool is not available"
            )

        method()
        self.state.drag_select = requested
        self._set_toggle_button_state(
            self.drag_select_button,
            self.state.drag_select,
        )
        self._render()
        return self.state.drag_select

    def toggle_drag_selection(self) -> None:
        """Toggle between rectangle cell selection and camera rotation."""
        if not self._callbacks_active:
            return
        self.set_drag_selection_mode(not self.state.drag_select)

    def _choose_result_output_file(self) -> str:
        """Open the operating system's Excel-workbook save dialog."""
        return choose_save_file(
            self._last_result_directory,
            title="Export result workbook",
            default_name="brain_strain_results.xlsx",
            extension=".xlsx",
            file_type_name="Excel workbook",
        )

    def select_cell_by_index(
        self,
        cell_index: int | str,
    ) -> int:
        """Select a cell by its zero-based PyVista/scalar-array index."""
        if isinstance(cell_index, bool):
            raise TypeError("cell index must be an integer")
        if isinstance(cell_index, str):
            value = cell_index.strip()
            if not value:
                raise ValueError("enter a cell index")
            if not value.isdecimal():
                raise ValueError("cell index must contain digits only")
            selected_index = int(value)
        elif isinstance(cell_index, (int, np.integer)):
            selected_index = int(cell_index)
        else:
            raise TypeError("cell index must be an integer")
        if selected_index < 0 or selected_index >= self.mesh.n_cells:
            raise IndexError(
                f"cell index must be in 0..{self.mesh.n_cells - 1}"
            )

        indices = np.asarray([selected_index], dtype=np.int64)
        self._on_cells_selected(indices)
        return selected_index

    def select_cell_by_number(self, cell_id: int | str) -> int:
        """Compatibility alias; the argument is a zero-based cell index."""
        return self.select_cell_by_index(cell_id)

    def clear_selected_cells(self) -> None:
        """Clear the selected cell, highlight actor, and selection details."""
        self.state.selected_cell_id = None
        self.state.selected_cell_ids = ()
        self._cell_id_input = ""
        self._cell_id_input_error = None
        self.visualizer.clear_highlighted_cells(render=False)
        self.plotter.remove_actor(
            "selection_text",
            reset_camera=False,
            render=False,
        )
        self._update_cell_id_input(render=False)
        self._update_status_text()
        self._render()

    def go_to_global_peak(self) -> None:
        self.state.time_index = self.peak.time_index

        self.visualizer.highlight_cells(
            np.array(
                [self.peak.element_index],
                dtype=int,
            ),
            render=False,
        )

        self._refresh_scene()

    def reset_camera(self) -> None:
        self.plotter.reset_camera(render=False)
        self._render()

    def zoom_in(self) -> None:
        """Zoom the camera in by one step."""
        self.plotter.camera.zoom(1.2)
        self._render()

    def zoom_out(self) -> None:
        """Zoom the camera out by one step."""
        self.plotter.camera.zoom(1.0 / 1.2)
        self._render()

    def _result_export_parameters(self) -> tuple[ResultParameter, ...]:
        """Return selected inputs in a stable, extendable tabular schema."""
        model = self._simulation_model
        parameters = [
            ResultParameter(
                "Simulation",
                "Method",
                _RESULT_EXPORT_METHOD,
                description=(
                    "Reduced-order demonstration; not a finite-element result."
                ),
            ),
            ResultParameter(
                "Simulation",
                "Impact mode",
                self._simulation_impact_mode,
            ),
            ResultParameter(
                "Simulation",
                "Requested duration",
                self._simulation_duration,
                "s",
            ),
            ResultParameter(
                "Simulation",
                "Target mean maximum shear strain",
                self._simulation_target_mean_mss,
                "1",
                "Blank means the impact-mode reference amplitude was used.",
            ),
            ResultParameter(
                "Loading",
                "Rotation axis X",
                float(self._simulation_rotation_axis[0]),
            ),
            ResultParameter(
                "Loading",
                "Rotation axis Y",
                float(self._simulation_rotation_axis[1]),
            ),
            ResultParameter(
                "Loading",
                "Rotation axis Z",
                float(self._simulation_rotation_axis[2]),
            ),
            ResultParameter(
                "Material",
                "Input modulus kind",
                model.modulus_kind,
            ),
            ResultParameter(
                "Material",
                "Estimated instantaneous modulus",
                model.estimated_instantaneous_modulus,
                "consistent stress unit",
            ),
            ResultParameter(
                "Material",
                "Modulus scale",
                model.modulus_scale,
            ),
            ResultParameter(
                "Material",
                "Poisson ratio",
                model.poisson_ratio,
            ),
            ResultParameter(
                "Material",
                "Instantaneous shear modulus G0",
                model.instantaneous_modulus,
                "consistent stress unit",
            ),
            ResultParameter(
                "Material",
                "Equilibrium ratio r_inf",
                model.equilibrium_ratio,
            ),
            ResultParameter(
                "Material",
                "Equilibrium shear modulus G_inf",
                model.equilibrium_modulus,
                "consistent stress unit",
            ),
            ResultParameter(
                "Material",
                "Maxwell branch count",
                model.branch_count,
            ),
            ResultParameter(
                "Interface",
                "Active field",
                self.field_name,
            ),
            ResultParameter(
                "Interface",
                "Active source",
                "Simulated data" if self.data_is_simulated else "Real data",
            ),
            ResultParameter(
                "Interface",
                "Active frame index",
                self.state.time_index,
            ),
            ResultParameter(
                "Interface",
                "Active time",
                float(self.times[self.state.time_index]),
                "s",
            ),
            ResultParameter(
                "Interface",
                "Hotspot threshold",
                self.state.threshold,
            ),
            ResultParameter(
                "Interface",
                "Selected cell count",
                len(self.state.selected_cell_ids),
            ),
        ]
        for index, (fraction, relaxation_time, branch_modulus) in enumerate(
            zip(
                model.branch_fractions,
                model.relaxation_times,
                model.branch_moduli,
                strict=True,
            ),
            start=1,
        ):
            parameters.extend(
                (
                    ResultParameter(
                        "Maxwell branches",
                        f"Branch {index} fraction g_{index}",
                        fraction,
                    ),
                    ResultParameter(
                        "Maxwell branches",
                        f"Branch {index} relaxation time tau_{index}",
                        relaxation_time,
                        "s",
                    ),
                    ResultParameter(
                        "Maxwell branches",
                        f"Branch {index} modulus G_{index}",
                        branch_modulus,
                        "consistent stress unit",
                    ),
                )
            )
        return tuple(parameters)

    def _build_result_export_data(self) -> ResultExportData:
        """Snapshot real, simulated, parameter, and selection state."""
        simulated_times, simulated_values = self._ensure_simulated_results()
        series: list[ResultSeries] = []
        if self._real_times is not None and self._real_scalar_series is not None:
            series.append(
                ResultSeries(
                    source_type="Real",
                    source_name="Real result data",
                    field_name=self._real_field_name,
                    times=self._real_times,
                    values=self._real_scalar_series,
                )
            )
        series.append(
            ResultSeries(
                source_type="Simulated",
                source_name="Generalized Maxwell calculation",
                field_name="Maximum shear strain",
                times=simulated_times,
                values=simulated_values,
                unit="1",
            )
        )
        source_element_ids = (
            {
                cell_index: int(self.element_ids[cell_index])
                for cell_index in self.state.selected_cell_ids
            }
            if self.element_ids is not None
            else {}
        )
        return ResultExportData(
            simulation_method=_RESULT_EXPORT_METHOD,
            active_source=(
                "Generalized Maxwell calculation"
                if self.data_is_simulated
                else "Real result data"
            ),
            active_frame_index=self.state.time_index,
            active_time=float(self.times[self.state.time_index]),
            parameters=self._result_export_parameters(),
            series=tuple(series),
            selected_cell_ids=self.state.selected_cell_ids,
            source_element_ids=source_element_ids,
        )

    def export_results(
        self,
        path: str | Path | None = None,
    ) -> Path | None:
        """Write the current result snapshot to an analysis-ready XLSX file."""
        selected = self._choose_result_output_file() if path is None else str(path)
        if not selected:
            return None
        workbook = build_result_workbook(self._build_result_export_data())
        saved = workbook.save(selected)
        self._last_result_directory = saved.parent
        self._result_export_status = f"Saved: {saved.name}"
        self._update_result_window_text(render=False)
        print(f"Saved result workbook: {saved}")
        return saved

    def save_screenshot(
        self,
        path: str | Path | None = None,
    ) -> Path:
        filename = Path(path) if path is not None else Path(
            f"brain_{self.field_name}_"
            f"frame_{self.state.time_index:04d}.png"
        )

        saved = self.visualizer.save_screenshot(filename)

        print(f"Saved screenshot: {saved}")
        return saved

    def run(self) -> None:
        if self.plotter.off_screen:
            self.visualizer.show()
            self.parameter_plotter.close()
            self.result_plotter.close()
            return

        # ``interactive_update`` lets one event loop service the main window
        # and both optional control/output windows.
        self.plotter.show(
            title="Brain strain visualisation",
            interactive_update=True,
            auto_close=False,
        )
        try:
            while self._window_is_open(self.plotter):
                assert self.plotter.iren is not None
                self.plotter.iren.process_events()
                if self._parameter_window_visible:
                    if self._window_is_open(self.parameter_plotter):
                        layout_changed = (
                            self._update_parameter_window_layout()
                        )
                        assert self.parameter_plotter.iren is not None
                        self.parameter_plotter.iren.process_events()
                        if layout_changed:
                            self.parameter_plotter.render()
                    else:
                        self._close_parameter_window()
                if self._result_window_visible:
                    if self._window_is_open(self.result_plotter):
                        assert self.result_plotter.iren is not None
                        self.result_plotter.iren.process_events()
                    else:
                        self._close_result_window()
                time.sleep(0.01)
        finally:
            self.close()

    def show_parameter_window(self) -> None:
        """Open the Maxwell controls after an explicit button click."""
        if not self.data_is_simulated:
            return
        if self._parameter_window_visible and self._window_is_open(
            self.parameter_plotter
        ):
            return
        if self.parameter_plotter._closed:
            self.parameter_plotter = self._create_parameter_plotter()
            callbacks_were_active = self._callbacks_active
            self._callbacks_active = False
            try:
                self._build_maxwell_widgets()
            finally:
                self._callbacks_active = callbacks_were_active

        if not self._parameter_window_off_screen:
            self._position_interactive_windows()
            self.parameter_plotter.show(
                title="Simulation parameters",
                interactive_update=True,
                auto_close=False,
            )
        self._parameter_window_visible = True

    def show_result_window(self) -> None:
        """Open the current result summary and Excel export section."""
        if self._result_window_visible and self._window_is_open(
            self.result_plotter
        ):
            self._update_result_window_text()
            return
        if self.result_plotter._closed:
            self.result_plotter = self._create_result_plotter()
            callbacks_were_active = self._callbacks_active
            self._callbacks_active = False
            try:
                self._build_result_widgets()
            finally:
                self._callbacks_active = callbacks_were_active

        self._update_result_window_text(render=False)
        if not self._result_window_off_screen:
            self._position_result_window()
            self.result_plotter.show(
                title="Result output",
                interactive_update=True,
                auto_close=False,
            )
        self._result_window_visible = True

    def _close_parameter_window(self) -> None:
        """Close and mark the optional parameter window as hidden."""
        self._parameter_window_visible = False
        if not self.parameter_plotter._closed:
            self.parameter_plotter.close()

    def _close_result_window(self) -> None:
        """Close and mark the optional result window as hidden."""
        self._result_window_visible = False
        if not self.result_plotter._closed:
            self.result_plotter.close()

    def _position_interactive_windows(self) -> None:
        """Place the parameter window beside the model when space permits."""
        main_window = self.plotter.render_window
        parameter_window = self.parameter_plotter.render_window
        if main_window is None or parameter_window is None:
            return

        margin = 20
        gap = 20
        screen_width, screen_height = map(int, main_window.GetScreenSize())
        main_width, main_height = map(int, main_window.GetSize())
        parameter_width, parameter_height = map(
            int, parameter_window.GetSize()
        )
        if main_width + gap + parameter_width + 2 * margin <= screen_width:
            main_window.SetPosition(margin, margin)
            parameter_window.SetPosition(
                margin + main_width + gap,
                margin,
            )
        elif (
            main_height + gap + parameter_height + 2 * margin
            <= screen_height
        ):
            main_window.SetPosition(margin, margin)
            parameter_window.SetPosition(
                margin,
                margin + main_height + gap,
            )

    def _position_result_window(self) -> None:
        """Place the result window beside or below the main viewer."""
        main_window = self.plotter.render_window
        result_window = self.result_plotter.render_window
        if main_window is None or result_window is None:
            return

        margin = 20
        gap = 20
        screen_width, screen_height = map(int, main_window.GetScreenSize())
        main_width, main_height = map(int, main_window.GetSize())
        result_width, result_height = map(int, result_window.GetSize())
        if main_width + gap + result_width + 2 * margin <= screen_width:
            main_window.SetPosition(margin, margin)
            result_window.SetPosition(margin + main_width + gap, margin)
        elif main_height + gap + result_height + 2 * margin <= screen_height:
            main_window.SetPosition(margin, margin)
            result_window.SetPosition(margin, margin + main_height + gap)

    @staticmethod
    def _window_is_open(plotter: pv.Plotter) -> bool:
        """Return whether a plotter still has a live interactive window."""
        if plotter._closed or plotter.iren is None:
            return False
        return not bool(plotter.iren.interactor.GetDone())

    def close(self) -> None:
        self._close_parameter_window()
        self._close_result_window()
        self.visualizer.close()


class BrainLauncherUI(LocalMeshOpeningUI):
    """Empty startup window that loads nothing until a file is selected."""

    def __init__(
        self,
        *,
        field_name: str = "MPS",
        off_screen: bool = False,
        window_size: tuple[int, int] = (1200, 850),
        render: bool = True,
    ) -> None:
        self.field_name = field_name
        self._render_enabled = render
        self._callbacks_active = False
        self._last_open_directory = _initial_open_directory()
        self.plotter = pv.Plotter(
            off_screen=off_screen,
            window_size=window_size,
        )
        self.plotter.set_background("#202124")
        self.plotter.theme.font.color = _UI_FONT_COLOR
        self.plotter.theme.font.size = _UI_LABEL_FONT_SIZE

        button_x = max(20, int(window_size[0]) // 2 - 120)
        button_y = max(20, int(window_size[1]) // 2 - 25)
        self.open_file_button = (
            self.plotter.add_checkbox_button_widget(
                callback=self._on_open_file_button,
                value=False,
                position=(button_x, button_y),
                size=50,
                color_on="mediumseagreen",
                color_off="grey",
            )
        )
        self.plotter.add_text(
            "Open local mesh",
            position=(button_x + 65, button_y + 10),
            font_size=_UI_LABEL_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="open_file_label",
            render=False,
        )
        self.plotter.add_text(
            "No model loaded",
            position="upper_left",
            font_size=_UI_STATUS_FONT_SIZE,
            color=_UI_FONT_COLOR,
            name="empty_status",
            render=False,
        )
        self.plotter.add_key_event(
            "o",
            self._on_open_file_shortcut,
        )
        self._callbacks_active = True
        self._render()

    def _render(self) -> None:
        if self._render_enabled:
            self.plotter.render()

    def run(self) -> None:
        self.plotter.show(title="Brain strain visualisation")

    def close(self) -> None:
        self.plotter.close()


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Open the interactive brain-strain viewer. Frame count is read "
            "from the selected scalar series or model time data."
        )
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=None,
        help="mesh readable by PyVista; omit to start with no model loaded",
    )
    parser.add_argument("--field", default="MPS")
    parser.add_argument(
        "--simulation-case",
        type=Path,
        help="ObservationCase JSON/TOML metadata for the positional mesh",
    )
    parser.add_argument(
        "--scalar-series",
        type=Path,
        help="NPY/NPZ array with shape (n_times, n_cells)",
    )
    parser.add_argument("--series-key")
    parser.add_argument("--time", type=Path)
    parser.add_argument("--time-key", default="time")
    parser.add_argument(
        "--observation-mesh",
        type=Path,
        help=(
            "second mesh to show as the observation case; enables the "
            "side-by-side comparison UI"
        ),
    )
    parser.add_argument(
        "--observation-field",
        help="observation cell field; defaults to --field",
    )
    parser.add_argument(
        "--observation-scalar-series",
        type=Path,
        help="observation NPY/NPZ array with shape (n_times, n_cells)",
    )
    parser.add_argument("--observation-series-key")
    parser.add_argument("--observation-time", type=Path)
    parser.add_argument("--observation-time-key", default="time")
    parser.add_argument(
        "--observation-case",
        type=Path,
        help="ObservationCase JSON/TOML metadata for --observation-mesh",
    )
    parser.add_argument(
        "--show-difference",
        action="store_true",
        help=(
            "initially show the normalized visual-difference panel when "
            "both meshes share identical cell geometry"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "use this many simulated frames for the fallback or simulation "
            "toggle; otherwise real-data frame count or the 10-frame "
            "reference sampling is used"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=REFERENCE_DURATION_SECONDS,
        help=(
            "simulated event duration in seconds (default: 0.162, matching "
            "ten 18-ms reference frames)"
        ),
    )
    parser.add_argument(
        "--impact-mode",
        choices=("neck-rotation", "neck-extension"),
        default="neck-rotation",
        help="select the referenced strain-amplitude scale",
    )
    parser.add_argument(
        "--branches",
        dest="maxwell_branches",
        type=int,
        default=DEFAULT_MAXWELL_MODEL.branch_count,
        metavar="N",
        help="number of Maxwell branches, from 3 to 6",
    )
    parser.add_argument(
        "--modulus-kind",
        choices=("G0", "E0"),
        default=DEFAULT_MAXWELL_MODEL.modulus_kind,
        help="interpret --estimated-modulus as shear G0 or Young's E0",
    )
    parser.add_argument(
        "--estimated-modulus",
        type=float,
        default=DEFAULT_MAXWELL_MODEL.estimated_instantaneous_modulus,
        help="estimated instantaneous E0 or G0 in a consistent stress unit",
    )
    parser.add_argument(
        "--modulus-scale",
        type=float,
        default=DEFAULT_MAXWELL_MODEL.modulus_scale,
        help="multiplier on the estimated modulus, constrained to 0.5..2.0",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=DEFAULT_MAXWELL_MODEL.poisson_ratio,
        help="Poisson ratio used only to convert E0 to shear modulus",
    )
    parser.add_argument(
        "--r-infinity",
        dest="equilibrium_ratio",
        type=float,
        default=DEFAULT_MAXWELL_MODEL.equilibrium_ratio,
        help="equilibrium ratio E_inf/E0 or G_inf/G0, from 0.001 to 0.999",
    )
    parser.add_argument(
        "--gi",
        nargs="+",
        type=float,
        help="N Prony fractions; they must be positive and sum to 1-r_inf",
    )
    tau_group = parser.add_mutually_exclusive_group()
    tau_group.add_argument(
        "--log10-tau",
        nargs="+",
        type=float,
        metavar="LOG10_SECONDS",
        help="N ordered log10 relaxation times",
    )
    tau_group.add_argument(
        "--relaxation-times",
        nargs="+",
        type=float,
        metavar="SECONDS",
        help="N ordered positive relaxation times",
    )
    parser.add_argument(
        "--tau-range",
        nargs=2,
        type=float,
        default=(DEFAULT_TAU_MIN_SECONDS, DEFAULT_TAU_MAX_SECONDS),
        metavar=("MIN_SECONDS", "MAX_SECONDS"),
        help="limits for automatically log-spaced tau_i values",
    )
    parser.add_argument(
        "--target-mean-mss",
        type=float,
        help=(
            "override the referenced peak spatial-mean maximum shear strain"
        ),
    )
    parser.add_argument(
        "--rotation-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="rotation axis in the mesh coordinate frame",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="save one view and exit instead of opening a window",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1200, 850),
    )
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--no-picking", action="store_true")
    return parser


def create_ui_from_args(
    args: Any,
    *,
    render: bool = True,
) -> BrainUI | BrainLauncherUI | BrainComparisonUI:
    """Create a configured UI from parsed command-line arguments."""
    if args.mesh is None:
        if args.screenshot is not None:
            raise ValueError("A mesh path is required with --screenshot")
        if args.scalar_series is not None:
            raise ValueError("A mesh path is required with --scalar-series")
        if args.time is not None:
            raise ValueError("A mesh path is required with --time")
        if getattr(args, "observation_mesh", None) is not None:
            raise ValueError(
                "A positional simulation mesh is required with "
                "--observation-mesh"
            )
        return BrainLauncherUI(
            field_name=args.field,
            off_screen=args.off_screen,
            window_size=tuple(args.window_size),
            render=render,
        )

    branch_count = int(args.maxwell_branches)
    branch_fractions = (
        default_branch_fractions(branch_count, args.equilibrium_ratio)
        if args.gi is None
        else tuple(args.gi)
    )
    if args.log10_tau is not None:
        relaxation_times = tuple(
            float(10.0**value) for value in args.log10_tau
        )
    elif args.relaxation_times is not None:
        relaxation_times = tuple(args.relaxation_times)
    else:
        relaxation_times = logarithmic_relaxation_times(
            branch_count,
            minimum=args.tau_range[0],
            maximum=args.tau_range[1],
        )
    if len(branch_fractions) != branch_count:
        raise ValueError(
            f"--gi requires {branch_count} value(s), received "
            f"{len(branch_fractions)}"
        )
    if len(relaxation_times) != branch_count:
        raise ValueError(
            f"relaxation times require {branch_count} value(s), received "
            f"{len(relaxation_times)}"
        )
    simulation_model = GeneralizedMaxwellModel(
        estimated_instantaneous_modulus=args.estimated_modulus,
        modulus_scale=args.modulus_scale,
        equilibrium_ratio=args.equilibrium_ratio,
        branch_fractions=branch_fractions,
        relaxation_times=relaxation_times,
        modulus_kind=args.modulus_kind,
        poisson_ratio=args.poisson_ratio,
    )

    loaded = load_data(
        args.mesh,
        time_path=args.time,
        metadata_path=getattr(args, "simulation_case", None),
        time_key=args.time_key,
    )
    prepared = prepare_display_data(
        loaded,
        args.field,
        scalar_series_path=args.scalar_series,
        series_key=args.series_key,
        role="Model",
    )
    mesh = prepared.mesh
    field_name = prepared.field_name
    real_series = prepared.scalar_series
    data_is_simulated = real_series is None
    series = real_series
    generated_times: npt.NDArray[np.float64] | None = None
    if data_is_simulated:
        frame_count = (
            int(args.frames)
            if args.frames is not None
            else (
                int(prepared.source_times.size)
                if prepared.source_times is not None
                else DEFAULT_SIMULATION_FRAME_COUNT
            )
        )
        generated_times, series = simulate_generalized_maxwell_strain(
            mesh,
            frame_count=frame_count,
            duration=args.duration,
            times=prepared.source_times,
            model=simulation_model,
            impact_mode=args.impact_mode,
            target_mean_maximum_shear_strain=args.target_mean_mss,
            rotation_axis=args.rotation_axis,
        )
        field_name = "Maximum shear strain"

    assert series is not None

    if generated_times is not None:
        times = generated_times
    else:
        assert prepared.times is not None
        times = prepared.times

    observation_mesh_path = getattr(args, "observation_mesh", None)
    if observation_mesh_path is not None:
        observation_loaded = load_data(
            observation_mesh_path,
            time_path=getattr(args, "observation_time", None),
            metadata_path=getattr(args, "observation_case", None),
            time_key=getattr(args, "observation_time_key", "time"),
        )
        observation_prepared = prepare_display_data(
            observation_loaded,
            getattr(args, "observation_field", None) or field_name,
            scalar_series_path=getattr(args, "observation_scalar_series", None),
            series_key=getattr(args, "observation_series_key", None),
            role="Observation",
            require_results=True,
        )
        observation_series = observation_prepared.scalar_series
        observation_times = observation_prepared.times
        assert observation_series is not None
        assert observation_times is not None

        return BrainComparisonUI(
            CaseDisplayData(
                role="observation",
                mesh=observation_prepared.mesh,
                times=observation_times,
                scalar_series=observation_series,
                field_name=observation_prepared.field_name,
                metadata=observation_prepared.metadata or None,
            ),
            CaseDisplayData(
                role="simulation",
                mesh=mesh,
                times=times,
                scalar_series=series,
                field_name=field_name,
                metadata=prepared.metadata or None,
            ),
            show_difference=bool(getattr(args, "show_difference", False)),
            off_screen=args.off_screen or args.screenshot is not None,
            window_size=tuple(args.window_size),
            render=render,
        )

    return BrainUI(
        mesh,
        times,
        series,
        mesh_frames=prepared.frames,
        field_name=field_name,
        data_is_simulated=data_is_simulated,
        simulation_frame_count=args.frames,
        simulation_duration=args.duration,
        simulation_model=simulation_model,
        simulation_impact_mode=args.impact_mode,
        simulation_target_mean_mss=args.target_mean_mss,
        simulation_rotation_axis=args.rotation_axis,
        initial_threshold=args.threshold,
        off_screen=args.off_screen or args.screenshot is not None,
        window_size=tuple(args.window_size),
        enable_picking=not args.no_picking and args.screenshot is None,
        render=render,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    try:
        ui = create_ui_from_args(args)
        if args.screenshot is not None:
            ui.save_screenshot(args.screenshot)
            ui.close()
        else:
            ui.run()
    except (
        DataLoadError,
        VisualisationError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"ui: error: {exc}") from exc
    return 0


__all__ = [
    "BrainComparisonUI",
    "BrainLauncherUI",
    "BrainUI",
    "CaseDisplayData",
    "UIState",
    "create_ui_from_args",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
