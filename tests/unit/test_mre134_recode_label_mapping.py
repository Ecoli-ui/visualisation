"""Tests for the MRE134-to-ReCoDE anatomical label crosswalk."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.adapters.lsdyna import add_pid_attributes, analyze_auxiliary_files
from brain_strain.adapters.mre134_labels import (
    MRE134_RECODE_LABEL_MAPPINGS,
    map_mre134_label,
    mapping_document,
    mappings_for_recode_part,
    observation_case_label_metadata,
    unified_label_for_recode_part,
    write_mapping,
)
from brain_strain.paths import DEFAULT_RECODE_ROOT

RECODE_PART_LIST = (
    DEFAULT_RECODE_ROOT / "src" / "dependencies" / "simulation" / "part_list_full.k"
)


class MRE134ReCoDELabelMappingTests(unittest.TestCase):
    def _recode_metadata(self) -> dict[str, object]:
        if not RECODE_PART_LIST.is_file():
            self.skipTest("ReCoDE reference part list is not installed")
        return analyze_auxiliary_files([RECODE_PART_LIST])

    def test_all_publication_roi_names_are_represented(self) -> None:
        categories: dict[str, int] = {}
        for mapping in MRE134_RECODE_LABEL_MAPPINGS:
            categories[mapping.category] = categories.get(mapping.category, 0) + 1

        self.assertEqual(len(MRE134_RECODE_LABEL_MAPPINGS), 34)
        self.assertEqual(
            categories,
            {
                "global": 4,
                "subcortical-gray-matter": 6,
                "white-matter-tract": 12,
                "cortical-gray-matter": 12,
            },
        )

    def test_full_names_abbreviations_and_spelling_variants_resolve(self) -> None:
        self.assertEqual(map_mre134_label("AM").mre134_label, "Amygdala")
        self.assertEqual(map_mre134_label("amygdala").unified_label, "Amygdala")
        self.assertEqual(
            map_mre134_label("subcortical grey matter").abbreviation,
            "SGM",
        )
        self.assertEqual(map_mre134_label("corpus-callosum").abbreviation, "CC")
        self.assertEqual(map_mre134_label("FSG").abbreviation, "FSM")
        self.assertEqual(
            map_mre134_label("inferior fronto-occipital fasciculus").abbreviation,
            "IFOF",
        )

    def test_direct_matches_combine_recode_left_and_right_parts(self) -> None:
        amygdala = map_mre134_label("Amygdala")
        self.assertEqual(amygdala.match_kind, "exact-bilateral")
        self.assertEqual(
            [(part.part_id, part.label) for part in amygdala.recode_parts],
            [(18, "Left-Amygdala"), (54, "Right-Amygdala")],
        )

        thalamus = map_mre134_label("TH")
        self.assertEqual([part.part_id for part in thalamus.recode_parts], [10, 49])
        self.assertTrue(
            all("Thalamus-Proper" in part.label for part in thalamus.recode_parts)
        )

    def test_granularity_loss_is_explicit(self) -> None:
        self.assertEqual(map_mre134_label("CC").match_kind, "partitioned")
        self.assertEqual(
            [part.part_id for part in map_mre134_label("CC").recode_parts],
            [251, 252, 253, 254, 255],
        )
        self.assertEqual(map_mre134_label("CST").match_kind, "broader-recode")
        self.assertEqual(map_mre134_label("SFC").unified_label, "Cerebral cortex")
        self.assertEqual(map_mre134_label("Global").match_kind, "no-direct-match")

    def test_recode_parts_receive_one_most_specific_unified_label(self) -> None:
        self.assertEqual(unified_label_for_recode_part(18), "Amygdala")
        self.assertEqual(unified_label_for_recode_part(251), "Corpus callosum")
        self.assertEqual(unified_label_for_recode_part(2), "Cerebral white matter")
        self.assertEqual(unified_label_for_recode_part(3), "Cerebral cortex")
        self.assertIsNone(unified_label_for_recode_part(260))
        self.assertIn(
            "White matter",
            {mapping.mre134_label for mapping in mappings_for_recode_part(2)},
        )

    def test_every_target_id_and_name_exists_in_active_recode_part_list(self) -> None:
        parsed_parts = self._recode_metadata()["parts"]
        for mapping in MRE134_RECODE_LABEL_MAPPINGS:
            for part in mapping.recode_parts:
                self.assertIn(part.part_id, parsed_parts)
                self.assertEqual(parsed_parts[part.part_id]["name"], part.label)

    def test_recode_converter_attaches_unified_label_cell_data(self) -> None:
        metadata = self._recode_metadata()
        grid = pv.ImageData(dimensions=(5, 2, 2))
        grid.cell_data["part_id"] = np.array([18, 251, 3, 260], dtype=np.int32)

        add_pid_attributes(grid, metadata)

        self.assertEqual(
            grid.cell_data["mre134_unified_label"].tolist(),
            ["Amygdala", "Corpus callosum", "Cerebral cortex", "Unmapped MRE134 ROI"],
        )
        embedded = json.loads(
            str(grid.field_data["mre134_recode_label_mapping_json"][0])
        )
        self.assertEqual(embedded["summary"]["mre134_labels"], 34)

    def test_json_and_observation_metadata_are_machine_readable(self) -> None:
        document = mapping_document()
        self.assertEqual(document["summary"]["mre134_labels"], 34)  # type: ignore[index]
        metadata = observation_case_label_metadata()
        self.assertFalse(metadata["label_volume_in_release"])
        self.assertEqual(len(metadata["labels"]), 34)  # type: ignore[arg-type]

        with TemporaryDirectory(prefix="mre134-label-map-test-") as directory:
            output = write_mapping(Path(directory) / "mapping.json")
            restored = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(restored, document)


if __name__ == "__main__":
    unittest.main()
