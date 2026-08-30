"""Generated-fixture tests for Dryad discovery, loading, and viewer use."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.adapters.dryad import (
    DRYAD_CELL_SCALARS,
    DRYAD_DEFAULT_FIELD,
    DRYAD_FRAME_INTERVAL_SECONDS,
    DryadAdapterError,
    discover_dryad_frames,
    load_dryad_sequence,
)
from brain_strain.io.loader import LoadedData, load_data


def _write_dryad_frame(path: Path, *, shift: float, value: float) -> None:
    points = np.array(
        (
            (0.0 + shift, 0.0, 0.0),
            (1.0 + shift, 0.0, 0.0),
            (0.0 + shift, 1.0, 0.0),
            (0.0 + shift, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    mesh = pv.UnstructuredGrid(
        np.array((4, 0, 1, 2, 3)),
        np.array((pv.CellType.TETRA,), dtype=np.uint8),
        points,
    )
    mesh.point_data["T1"] = np.full(mesh.n_points, 0.5, dtype=np.float32)
    mesh.point_data["T1_std"] = np.full(mesh.n_points, 0.1, dtype=np.float32)
    mesh.point_data["disp_std"] = np.full(
        mesh.n_points, abs(shift), dtype=np.float32
    )
    mesh.point_data["disp"] = np.tile(
        np.array((shift, 0.0, 0.0), dtype=np.float32),
        (mesh.n_points, 1),
    )
    for index, name in enumerate(DRYAD_CELL_SCALARS):
        mesh.cell_data[name] = np.array((value + index,), dtype=np.float32)
    mesh.cell_data["V1"] = np.array(((value, 0.0, 0.0),), dtype=np.float32)
    mesh.save(path, binary=False)


class DryadAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory(prefix="dryad-adapter-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        _write_dryad_frame(
            self.root / "HR_MESH_1.vtk", shift=0.0, value=0.1
        )
        _write_dryad_frame(
            self.root / "HR_MESH_2.vtk", shift=0.25, value=0.2
        )
        _write_dryad_frame(
            self.root / "NE_MESH_1.vtk", shift=1.0, value=0.3
        )

    def test_discovery_is_condition_specific_and_naturally_ordered(self) -> None:
        condition, frames = discover_dryad_frames(
            self.root / "HR_MESH_2.vtk"
        )

        self.assertEqual(condition, "HR")
        self.assertEqual([item.index for item in frames], [1, 2])
        self.assertEqual(
            [item.path.name for item in frames],
            ["HR_MESH_1.vtk", "HR_MESH_2.vtk"],
        )

    def test_incomplete_sequence_is_rejected(self) -> None:
        (self.root / "HR_MESH_2.vtk").rename(self.root / "HR_MESH_3.vtk")

        with self.assertRaisesRegex(DryadAdapterError, "missing frame.*2"):
            discover_dryad_frames(self.root / "HR_MESH_1.vtk")

    def test_load_preserves_changing_geometry_fields_and_time(self) -> None:
        sequence = load_dryad_sequence(self.root / "HR_MESH_1.vtk")

        self.assertEqual(sequence.condition, "HR")
        self.assertEqual(len(sequence.frames), 2)
        np.testing.assert_allclose(
            sequence.times,
            (0.0, DRYAD_FRAME_INTERVAL_SECONDS),
        )
        np.testing.assert_allclose(
            sequence.frames[1].points[:, 0] - sequence.frames[0].points[:, 0],
            0.25,
        )
        np.testing.assert_allclose(
            sequence.scalar_series(DRYAD_DEFAULT_FIELD),
            ((1.1,), (1.2,)),
        )
        self.assertEqual(
            sequence.frames[0].active_scalars_name, DRYAD_DEFAULT_FIELD
        )
        self.assertEqual(sequence.metadata["condition_name"], "head rotation")

        generic = sequence.as_loaded_data()
        self.assertIsInstance(generic, LoadedData)
        self.assertEqual(len(generic.frames), 2)

    def test_generic_loader_automatically_uses_the_adapter(self) -> None:
        loaded = load_data(self.root / "HR_MESH_2.vtk")

        self.assertEqual(len(loaded.frames), 2)
        np.testing.assert_allclose(loaded.time, (0.0, 0.018))
        self.assertEqual(loaded.metadata["adapter"], "dryad")
        self.assertEqual(loaded.metadata["default_field"], DRYAD_DEFAULT_FIELD)

    def test_viewer_defaults_to_real_gmaxt2_and_updates_geometry(self) -> None:
        from brain_strain.viewer.app import (
            BrainUI,
            _build_argument_parser,
            create_ui_from_args,
        )

        args = _build_argument_parser().parse_args(
            (
                str(self.root / "HR_MESH_1.vtk"),
                "--off-screen",
                "--no-picking",
            )
        )
        viewer = create_ui_from_args(args, render=False)
        self.addCleanup(viewer.close)

        self.assertIsInstance(viewer, BrainUI)
        self.assertEqual(viewer.field_name, DRYAD_DEFAULT_FIELD)
        self.assertFalse(viewer.data_is_simulated)
        self.assertEqual(viewer.maxwell_branch_slider.GetEnabled(), 0)
        self.assertEqual(viewer.scalar_series.shape, (2, 1))
        baseline_points = viewer.visualizer.mesh.points.copy()

        viewer._on_time_slider(1.0)

        np.testing.assert_allclose(
            viewer.visualizer.mesh.points[:, 0] - baseline_points[:, 0],
            0.25,
        )
        self.assertAlmostEqual(viewer.times[1], 0.018)


if __name__ == "__main__":
    unittest.main()
