"""Tests for the platform-independent file-dialog interface."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from brain_strain.viewer.dialogs import choose_open_file, choose_save_file


class FileDialogTests(unittest.TestCase):
    @patch("brain_strain.viewer.dialogs.sys.platform", "darwin")
    @patch("brain_strain.viewer.dialogs.subprocess.run")
    def test_open_dialog_returns_the_native_selection(self, run) -> None:
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="/tmp/model.vtk\n",
            stderr="",
        )
        selected = choose_open_file(Path.cwd(), title="Open", pattern="*.vtk")
        self.assertEqual(selected, "/tmp/model.vtk")

    @patch("brain_strain.viewer.dialogs.sys.platform", "darwin")
    @patch("brain_strain.viewer.dialogs.subprocess.run")
    def test_save_dialog_reports_native_errors(self, run) -> None:
        run.return_value = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="permission denied",
        )
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            choose_save_file(
                Path.cwd(),
                title="Export",
                default_name="result.xlsx",
                extension=".xlsx",
                file_type_name="Excel workbook",
            )


if __name__ == "__main__":
    unittest.main()
