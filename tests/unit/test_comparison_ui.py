"""Tests for the side-by-side observation/simulation UI."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.comparison.view import (
    BrainComparisonUI,
    CaseDisplayData,
    meshes_share_cell_geometry,
    normalize_visual_series,
)


class ComparisonUITests(unittest.TestCase):
    @staticmethod
    def _mesh() -> pv.ImageData:
        return pv.ImageData(dimensions=(3, 2, 2))

    def test_normalization_is_unit_free_and_preserves_missing_values(self) -> None:
        values = np.array([[10.0, 20.0], [30.0, np.nan]])

        normalized = normalize_visual_series(values)

        np.testing.assert_allclose(
            normalized[np.isfinite(normalized)], [0.0, 0.5, 1.0]
        )
        self.assertTrue(np.isnan(normalized[1, 1]))

    def test_geometry_check_rejects_an_unregistered_mesh(self) -> None:
        observation = self._mesh()
        simulation = observation.copy(deep=True)
        simulation.translate((1.0, 0.0, 0.0), inplace=True)

        self.assertFalse(
            meshes_share_cell_geometry(observation, simulation)
        )

    def test_difference_panel_is_optional_and_updates_both_timelines(self) -> None:
        mesh = self._mesh()
        observation = CaseDisplayData(
            role="observation",
            mesh=mesh,
            times=(0.0, 1.0),
            scalar_series=((0.0, 1.0), (1.0, 0.0)),
            field_name="observed",
        )
        simulation = CaseDisplayData(
            role="simulation",
            mesh=mesh.copy(deep=True),
            times=(0.0, 0.5, 1.0),
            scalar_series=(
                (0.0, 1.0),
                (0.25, 0.75),
                (0.75, 0.25),
            ),
            field_name="simulated",
        )
        ui = BrainComparisonUI(
            observation,
            simulation,
            off_screen=True,
            render=False,
        )
        self.addCleanup(ui.close)

        self.assertTrue(ui.difference_available)
        self.assertFalse(ui.state.show_difference)
        self.assertTrue(ui.show_difference())
        self.assertEqual(ui.set_comparison_frame(2), 2)
        np.testing.assert_allclose(
            ui.current_difference,
            np.array((0.25, -0.25)),
        )

    def test_difference_control_stays_disabled_for_different_grids(self) -> None:
        observation_mesh = self._mesh()
        simulation_mesh = pv.ImageData(dimensions=(2, 2, 2))
        observation = CaseDisplayData(
            "observation",
            observation_mesh,
            (0.0,),
            ((1.0, 2.0),),
            "observed",
        )
        simulation = CaseDisplayData(
            "simulation",
            simulation_mesh,
            (0.0,),
            ((1.0,),),
            "simulated",
        )
        ui = BrainComparisonUI(
            observation,
            simulation,
            off_screen=True,
            render=False,
        )
        self.addCleanup(ui.close)

        self.assertFalse(ui.difference_available)
        self.assertFalse(ui.show_difference())
        self.assertIsNone(ui.current_difference)
        self.assertIn("Registration", ui.difference_unavailable_reason or "")

    def test_command_arguments_create_the_comparison_ui(self) -> None:
        from brain_strain.viewer.app import _build_argument_parser, create_ui_from_args

        with TemporaryDirectory(prefix="comparison-ui-test-") as directory:
            root = Path(directory)
            observation_mesh = self._mesh()
            observation_mesh.cell_data["observed"] = np.array((0.0, 1.0))
            simulation_mesh = self._mesh()
            simulation_mesh.cell_data["simulated"] = np.array((0.25, 0.75))
            observation_path = root / "observation.vti"
            simulation_path = root / "simulation.vti"
            observation_mesh.save(observation_path)
            simulation_mesh.save(simulation_path)

            args = _build_argument_parser().parse_args(
                (
                    str(simulation_path),
                    "--field",
                    "simulated",
                    "--observation-mesh",
                    str(observation_path),
                    "--observation-field",
                    "observed",
                    "--show-difference",
                    "--off-screen",
                )
            )
            ui = create_ui_from_args(args, render=False)
            self.addCleanup(ui.close)

        self.assertIsInstance(ui, BrainComparisonUI)
        self.assertTrue(ui.state.show_difference)


if __name__ == "__main__":
    unittest.main()
