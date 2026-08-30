"""Tests for the analysis-ready brain-strain XLSX export."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from brain_strain.io.export import (
    ResultExportData,
    ResultParameter,
    ResultSeries,
    ResultWorkbook,
    build_result_workbook,
)

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


class ResultExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.real = ResultSeries(
            "Real",
            "Real result data",
            "Measured MPS",
            (0.0, 0.1),
            ((0.10, 0.20), (0.30, 0.40)),
        )
        self.simulated = ResultSeries(
            "Simulated",
            "Generalized Maxwell calculation",
            "Maximum shear strain",
            (0.0, 0.1),
            ((0.00, 0.01), (0.02, 0.03)),
            "1",
        )
        self.data = ResultExportData(
            simulation_method="Generalized Maxwell (reduced order)",
            active_source="Real result data",
            active_frame_index=1,
            active_time=0.1,
            parameters=(
                ResultParameter("Material", "Modulus scale", 1.5),
                ResultParameter("Interface", "Threshold", 0.2),
            ),
            series=(self.real, self.simulated),
            selected_cell_ids=(1,),
            source_element_ids={1: 42},
        )

    def test_standard_sections_include_real_simulated_and_selected_values(
        self,
    ) -> None:
        workbook = build_result_workbook(self.data)

        self.assertEqual(
            [sheet.name for sheet in workbook.sheets],
            [
                "Results Summary",
                "Parameters",
                "Frame Results",
                "Selected Cell Results",
                "Data Dictionary",
            ],
        )
        summary = workbook.sheets[0].rows
        real_maximum = next(
            row for row in summary if row[0] == "Actual real-data global maximum"
        )
        simulated_maximum = next(
            row for row in summary if row[0] == "Calculated maximum strain"
        )
        self.assertEqual(real_maximum[1], 0.4)
        self.assertEqual(simulated_maximum[1], 0.03)

        selected_rows = workbook.sheets[3].rows
        self.assertEqual(len(selected_rows), 2)
        self.assertEqual(selected_rows[0][3], 1)
        self.assertEqual(selected_rows[0][4], 42)
        self.assertEqual(selected_rows[0][7], 0.4)

    def test_saved_package_contains_parseable_xml_and_numeric_cells(self) -> None:
        with TemporaryDirectory() as directory:
            path = build_result_workbook(self.data).save(
                Path(directory) / "results"
            )

            self.assertEqual(path.suffix, ".xlsx")
            with ZipFile(path) as archive:
                expected = {
                    "[Content_Types].xml",
                    "_rels/.rels",
                    "xl/workbook.xml",
                    "xl/styles.xml",
                    "xl/worksheets/sheet1.xml",
                    "xl/worksheets/sheet5.xml",
                }
                self.assertTrue(expected.issubset(archive.namelist()))
                for name in archive.namelist():
                    if name.endswith(".xml") or name.endswith(".rels"):
                        ET.fromstring(archive.read(name))

                summary_xml = ET.fromstring(
                    archive.read("xl/worksheets/sheet1.xml")
                )
                numeric_cells = summary_xml.findall(
                    f".//{{{_MAIN_NS}}}c[{{{_MAIN_NS}}}v]"
                )
                self.assertTrue(numeric_cells)

    def test_workbook_accepts_future_custom_sections(self) -> None:
        workbook = ResultWorkbook()

        sheet = workbook.add_sheet(
            "Validation Metrics",
            ("Metric", "Value"),
            (("RMSE", 0.01),),
        )

        self.assertEqual(sheet.rows[0], ("RMSE", 0.01))
        with self.assertRaisesRegex(ValueError, "duplicate worksheet"):
            workbook.add_sheet("validation metrics", ("Metric",), ())


if __name__ == "__main__":
    unittest.main()
