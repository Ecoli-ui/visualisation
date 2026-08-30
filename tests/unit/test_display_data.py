"""Tests for shared scalar-series and display-case preparation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.io.display import load_scalar_series, prepare_display_data
from brain_strain.io.loader import LoadedData


class DisplayDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = pv.ImageData(dimensions=(3, 2, 2))
        self.mesh.cell_data["GmaxT2"] = np.array([0.1, 0.2])

    def test_adapter_default_and_time_validation_share_one_path(self) -> None:
        loaded = LoadedData(
            mesh=self.mesh,
            time=np.array([0.0]),
            metadata={"adapter": "dryad", "default_field": "GmaxT2"},
            frames=(self.mesh,),
        )
        prepared = prepare_display_data(loaded, "MPS")
        self.assertEqual(prepared.field_name, "GmaxT2")
        self.assertEqual(prepared.scalar_series.shape, (1, 2))
        np.testing.assert_allclose(prepared.times, [0.0])

    def test_npz_requires_a_key_when_multiple_arrays_exist(self) -> None:
        with TemporaryDirectory(prefix="display-data-test-") as directory:
            source = Path(directory) / "series.npz"
            np.savez(source, first=np.ones((1, 2)), second=np.zeros((1, 2)))
            with self.assertRaisesRegex(ValueError, "multiple arrays"):
                load_scalar_series(source)
            selected = load_scalar_series(source, key="SECOND")
        np.testing.assert_allclose(selected, np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
