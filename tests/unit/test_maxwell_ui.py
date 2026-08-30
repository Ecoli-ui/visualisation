"""Tests for interactive generalized-Maxwell parameter sliders."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.simulation import simulate_generalized_maxwell_strain
from brain_strain.viewer.app import BrainUI


class MaxwellSliderTests(unittest.TestCase):
    def setUp(self) -> None:
        mesh = pv.ImageData(dimensions=(4, 4, 4))
        times, strain = simulate_generalized_maxwell_strain(mesh)
        self.viewer = BrainUI(
            mesh,
            times,
            strain,
            data_is_simulated=True,
            off_screen=True,
            enable_picking=False,
            render=False,
        )
        self.addCleanup(self.viewer.close)

    def test_sliders_change_every_adjustable_parameter(self) -> None:
        viewer = self.viewer
        self.assertEqual(viewer.maxwell_branch_slider.GetEnabled(), 1)
        self.assertEqual(len(viewer.maxwell_g_sliders), 6)
        self.assertEqual(len(viewer.maxwell_tau_sliders), 6)

        viewer._on_maxwell_branch_slider(6.0)
        self.assertEqual(viewer._simulation_model.branch_count, 6)
        self.assertTrue(
            all(slider.GetEnabled() for slider in viewer.maxwell_g_sliders)
        )

        initial_peak = float(np.max(np.mean(viewer.scalar_series, axis=1)))
        viewer._on_maxwell_modulus_slider(4.0)
        stiff_peak = float(np.max(np.mean(viewer.scalar_series, axis=1)))
        self.assertEqual(viewer._simulation_model.modulus_scale, 2.0)
        self.assertAlmostEqual(initial_peak / stiff_peak, 2.0)

        viewer._on_maxwell_equilibrium_slider(0.4)
        model = viewer._simulation_model
        self.assertAlmostEqual(model.equilibrium_ratio, 0.4)
        self.assertAlmostEqual(sum(model.branch_fractions), 0.6)

        viewer._on_maxwell_g_slider(0, 0.2)
        model = viewer._simulation_model
        self.assertAlmostEqual(model.branch_fractions[0], 0.2)
        self.assertAlmostEqual(
            model.equilibrium_ratio + sum(model.branch_fractions), 1.0
        )

        viewer._on_maxwell_tau_slider(0, -4.0)
        model = viewer._simulation_model
        self.assertAlmostEqual(model.relaxation_times[0], 1e-4)
        self.assertTrue(
            all(
                right > left
                for left, right in zip(
                    model.relaxation_times,
                    model.relaxation_times[1:],
                    strict=False,
                )
            )
        )

    def test_parameter_sliders_use_a_separate_window(self) -> None:
        viewer = self.viewer
        parameter_sliders = (
            viewer.maxwell_branch_slider,
            viewer.maxwell_modulus_slider,
            viewer.maxwell_equilibrium_slider,
            *viewer.maxwell_g_sliders,
            *viewer.maxwell_tau_sliders,
        )

        self.assertIsNot(viewer.parameter_plotter, viewer.plotter)
        self.assertFalse(viewer._parameter_window_visible)
        self.assertTrue(
            all(
                slider in viewer.parameter_plotter.widgets.slider_widgets
                for slider in parameter_sliders
            )
        )
        self.assertTrue(
            all(
                slider not in viewer.plotter.widgets.slider_widgets
                for slider in parameter_sliders
            )
        )

        viewer._on_parameter_window_button(True)
        self.assertTrue(viewer._parameter_window_visible)

    def test_parameter_window_can_be_reopened(self) -> None:
        viewer = self.viewer
        original_plotter = viewer.parameter_plotter
        viewer._close_parameter_window()

        viewer.show_parameter_window()

        self.assertIsNot(viewer.parameter_plotter, original_plotter)
        self.assertTrue(viewer._parameter_window_visible)
        self.assertIn(
            viewer.maxwell_branch_slider,
            viewer.parameter_plotter.widgets.slider_widgets,
        )

    def test_parameter_layout_reflows_for_a_narrow_window(self) -> None:
        viewer = self.viewer
        branch_representation = (
            viewer.maxwell_branch_slider.GetRepresentation()
        )
        tau_representation = viewer.maxwell_tau_sliders[0].GetRepresentation()
        wide_tau_x = float(
            tau_representation.GetPoint1Coordinate().GetValue()[0]
        )
        wide_title_height = float(branch_representation.GetTitleHeight())
        title_actor = viewer.parameter_plotter.renderer.actors[
            "maxwell_window_title"
        ]
        wide_font_size = int(title_actor.GetTextProperty().GetFontSize())

        viewer.parameter_plotter.render_window.SetSize(360, 800)
        viewer.parameter_plotter.iren.interactor.InvokeEvent(
            "ConfigureEvent"
        )

        narrow_branch_x = float(
            branch_representation.GetPoint1Coordinate().GetValue()[0]
        )
        narrow_tau_x = float(
            tau_representation.GetPoint1Coordinate().GetValue()[0]
        )
        fraction_heading = viewer.parameter_plotter.renderer.actors[
            "maxwell_fraction_heading"
        ]
        self.assertAlmostEqual(narrow_branch_x, narrow_tau_x)
        self.assertNotAlmostEqual(wide_tau_x, narrow_tau_x)
        self.assertEqual(fraction_heading.GetVisibility(), 0)
        self.assertLess(
            int(title_actor.GetTextProperty().GetFontSize()),
            wide_font_size,
        )
        self.assertNotEqual(
            wide_title_height,
            float(branch_representation.GetTitleHeight()),
        )

    def test_fullscreen_resize_restores_normalized_coordinates(self) -> None:
        viewer = self.viewer
        representation = viewer.maxwell_branch_slider.GetRepresentation()
        point1 = representation.GetPoint1Coordinate()
        point1.SetCoordinateSystemToDisplay()
        point1.SetValue(1.0, 1.0)

        viewer.parameter_plotter.render_window.SetSize(1920, 1080)
        viewer.parameter_plotter.render_window.InvokeEvent(
            "WindowResizeEvent"
        )

        self.assertEqual(
            point1.GetCoordinateSystemAsString(),
            "Normalized Display",
        )
        self.assertEqual(viewer._parameter_layout_size, (1920, 1080))
        g_point1 = (
            viewer.maxwell_g_sliders[0]
            .GetRepresentation()
            .GetPoint1Coordinate()
            .GetValue()
        )
        tau_point1 = (
            viewer.maxwell_tau_sliders[0]
            .GetRepresentation()
            .GetPoint1Coordinate()
            .GetValue()
        )
        self.assertLess(point1.GetValue()[0], g_point1[0])
        self.assertLess(g_point1[0], tau_point1[0])
        primary_heading = viewer.parameter_plotter.renderer.actors[
            "maxwell_primary_heading"
        ]
        self.assertEqual(primary_heading.GetVisibility(), 1)
        computed_x, computed_y = point1.GetComputedDisplayValue(
            viewer.parameter_plotter.renderer
        )
        self.assertGreater(computed_x, 10)
        self.assertGreater(computed_y, 100)

    def test_inactive_branch_controls_are_hidden(self) -> None:
        self.assertEqual(self.viewer._simulation_model.branch_count, 3)

        self.assertTrue(
            all(
                slider.GetEnabled()
                for slider in self.viewer.maxwell_g_sliders[:3]
            )
        )
        self.assertTrue(
            all(
                not slider.GetEnabled()
                for slider in self.viewer.maxwell_g_sliders[3:]
            )
        )

    def test_result_export_uses_the_selected_maxwell_parameters(self) -> None:
        self.viewer._on_maxwell_modulus_slider(3.0)

        export_data = self.viewer._build_result_export_data()
        modulus_scale = next(
            parameter
            for parameter in export_data.parameters
            if parameter.name == "Modulus scale"
        )

        self.assertEqual(modulus_scale.value, 1.5)

    def test_result_window_exports_real_and_simulated_results(self) -> None:
        mesh = pv.ImageData(dimensions=(3, 3, 3))
        times = np.array((0.0, 0.1), dtype=float)
        real_values = np.vstack(
            (
                np.linspace(0.01, 0.08, mesh.n_cells),
                np.linspace(0.02, 0.16, mesh.n_cells),
            )
        )
        viewer = BrainUI(
            mesh,
            times,
            real_values,
            field_name="Measured MPS",
            off_screen=True,
            enable_picking=False,
            render=False,
        )
        self.addCleanup(viewer.close)
        viewer.select_cell_by_index(1)

        viewer.show_result_window()
        self.assertTrue(viewer._result_window_visible)
        original_result_plotter = viewer.result_plotter
        viewer._close_result_window()
        viewer.show_result_window()
        self.assertIsNot(viewer.result_plotter, original_result_plotter)
        self.assertIn(
            viewer.result_export_button,
            viewer.result_plotter.widgets.button_widgets,
        )
        export_data = viewer._build_result_export_data()
        self.assertEqual(
            [series.source_type for series in export_data.series],
            ["Real", "Simulated"],
        )

        with TemporaryDirectory() as directory:
            saved = viewer.export_results(Path(directory) / "comparison.xlsx")
            assert saved is not None
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
