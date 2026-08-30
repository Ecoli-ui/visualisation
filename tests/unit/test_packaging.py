"""Tests for behavior that differs between source and frozen runtimes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from brain_strain import paths
from brain_strain.viewer import app


class PackagingRuntimeTests(unittest.TestCase):
    def test_frozen_resource_root_uses_pyinstaller_bundle(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch.object(paths.sys, "frozen", True, create=True),
                patch.object(paths.sys, "_MEIPASS", directory, create=True),
            ):
                self.assertEqual(paths._checkout_root(), Path(directory).resolve())

    def test_source_viewer_relaunches_the_module(self) -> None:
        with (
            patch.object(app.sys, "executable", "/python"),
            patch.object(app.sys, "frozen", False, create=True),
        ):
            command = app._viewer_launch_command(Path("model.vtk"), "MPS")
        self.assertEqual(
            command,
            ["/python", "-m", "brain_strain", "model.vtk", "--field", "MPS"],
        )

    def test_frozen_viewer_relaunches_the_executable_directly(self) -> None:
        with (
            patch.object(app.sys, "executable", "BrainStrain.exe"),
            patch.object(app.sys, "frozen", True, create=True),
        ):
            command = app._viewer_launch_command(Path("model.vtk"), "strain")
        self.assertEqual(
            command,
            ["BrainStrain.exe", "model.vtk", "--field", "strain"],
        )

    def test_missing_default_mesh_directory_falls_back_to_home(self) -> None:
        with TemporaryDirectory() as directory:
            missing_mesh = Path(directory) / "missing" / "model.vtk"
            with patch.object(app, "DEFAULT_MESH", missing_mesh):
                self.assertEqual(app._initial_open_directory(), Path.home())


if __name__ == "__main__":
    unittest.main()
