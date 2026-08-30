"""Characterization tests for the public rendering boundary."""

import unittest

import numpy as np
import pyvista as pv

from brain_strain.viewer.rendering import MeshVisualisation


class MeshVisualisationTests(unittest.TestCase):
    def setUp(self) -> None:
        mesh = pv.ImageData(dimensions=(3, 3, 3))
        values = np.arange(mesh.n_cells, dtype=np.float64)
        self.viewer = MeshVisualisation(
            mesh,
            "strain",
            scalar_frames=np.stack((values, values + 10.0)),
            association="cell",
            times=(0.0, 1.0),
            off_screen=True,
        )
        self.viewer.initialize_user_view(show_axes=False, render=False)

    def tearDown(self) -> None:
        self.viewer.close()

    def test_public_actor_and_slice_boundary(self) -> None:
        self.assertIsNotNone(self.viewer.main_actor)
        self.assertEqual(self.viewer.mesh_options()["scalars"], "strain")
        slices = self.viewer.show_slices(render=False)
        self.assertEqual(len(slices), 3)
        self.assertEqual(self.viewer.slice_settings["normals"], ("x", "y", "z"))
        self.viewer.update_scalar_frame(1, render=False)
        self.assertEqual(self.viewer.current_frame, 1)

    def test_public_picker_boundary_reports_original_ids(self) -> None:
        selected: list[np.ndarray] = []
        self.viewer.configure_pick_handler(selected.append, additive=True)
        picked = self.viewer.mesh.extract_cells([2])
        np.testing.assert_array_equal(self.viewer.picked_original_ids(picked), [2])
        self.viewer.handle_picked_cells(picked)
        np.testing.assert_array_equal(selected[0], [2])


if __name__ == "__main__":
    unittest.main()
