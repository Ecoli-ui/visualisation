"""Characterization tests for mesh analysis primitives."""

import unittest

import numpy as np
import pyvista as pv

from brain_strain.analysis import (
    detect_hotspots,
    extract_element_history,
    find_global_peak,
)


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = pv.ImageData(dimensions=(3, 2, 2))
        self.mesh.cell_data["strain"] = np.array([0.1, 0.9])

    def test_hotspots_and_history_preserve_cell_indices(self) -> None:
        hotspots = detect_hotspots(
            self.mesh,
            "strain",
            association="cell",
            threshold=0.5,
            max_hotspots=None,
        )
        self.assertEqual([item.index for item in hotspots], [1])
        history = extract_element_history(
            np.array([[0.1, 0.9], [0.2, 1.2]]),
            1,
        )
        np.testing.assert_allclose(history, [0.9, 1.2])

    def test_global_peak_retains_time_and_position(self) -> None:
        centers = self.mesh.cell_centers().points
        peak = find_global_peak(
            np.array([[0.1, 0.9], [1.3, 1.2]]),
            np.array([0.0, 0.5]),
            centers,
        )
        self.assertEqual(peak.element_index, 0)
        self.assertEqual(peak.time_index, 1)
        self.assertAlmostEqual(peak.value, 1.3)
        np.testing.assert_allclose(peak.position, centers[0])


if __name__ == "__main__":
    unittest.main()
