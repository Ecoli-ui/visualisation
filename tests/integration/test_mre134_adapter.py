"""Integration tests for the MRE134 PyVista and ObservationCase adapter."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv

from brain_strain.adapters.mre134 import (
    CONTRIBUTOR_COUNT_ARRAY,
    DAMPING_ARRAY,
    STIFFNESS_ARRAY,
    VALID_MASK_ARRAY,
    build_mre134_observation_case,
    compare_mre134_with_case,
    load_mre134,
    load_mre134_demographics,
)
from brain_strain.io.loader import LoadedData, load_observation_case
from brain_strain.observation_case import CaseSource, ObservationCase, ValueAssociation
from brain_strain.paths import DEFAULT_MRE134_ROOT, EXAMPLE_CASES_ROOT

MRE_ROOT = DEFAULT_MRE134_ROOT


class MRE134AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MRE_ROOT.is_dir():
            raise unittest.SkipTest("MRE134 research release is not installed")
        cls.loaded = load_mre134(MRE_ROOT, include_contributor_count=True)

    def test_demographics_align_with_release(self) -> None:
        participants = load_mre134_demographics(MRE_ROOT)

        self.assertEqual(len(participants), 134)
        self.assertEqual(len({item.subject_id for item in participants}), 134)
        self.assertEqual(
            sum(item.sex == "F" for item in participants),
            78,
        )
        self.assertEqual(
            (
                min(item.age_years for item in participants),
                max(item.age_years for item in participants),
            ),
            (18.0, 35.0),
        )

    def test_loaded_grid_preserves_voxel_centres_and_cell_semantics(self) -> None:
        grid = self.loaded.dataset

        self.assertEqual(grid.dimensions, (92, 110, 92))
        self.assertEqual(grid.n_cells, 91 * 109 * 91)
        self.assertEqual(grid.spacing, (2.0, 2.0, 2.0))
        self.assertEqual(grid.origin, (91.0, -127.0, -73.0))
        np.testing.assert_allclose(
            grid.direction_matrix,
            np.diag((-1.0, 1.0, 1.0)),
        )
        centres = grid.cell_centers().points
        np.testing.assert_allclose(centres[0], (90.0, -126.0, -72.0))
        np.testing.assert_allclose(centres[-1], (-90.0, 90.0, 108.0))

    def test_loaded_arrays_have_expected_units_support_and_means(self) -> None:
        grid = self.loaded.dataset
        self.assertEqual(
            set(grid.cell_data.keys()),
            {
                STIFFNESS_ARRAY,
                DAMPING_ARRAY,
                VALID_MASK_ARRAY,
                CONTRIBUTOR_COUNT_ARRAY,
            },
        )
        self.assertEqual(grid.active_scalars_name, STIFFNESS_ARRAY)
        valid = np.asarray(grid.cell_data[VALID_MASK_ARRAY], dtype=bool)

        self.assertEqual(int(valid.sum()), 212_217)
        self.assertAlmostEqual(
            float(np.mean(grid.cell_data[STIFFNESS_ARRAY][valid])),
            2.612625,
            places=5,
        )
        self.assertAlmostEqual(
            float(np.mean(grid.cell_data[DAMPING_ARRAY][valid])),
            0.208470,
            places=5,
        )

        contributors = np.asarray(grid.cell_data[CONTRIBUTOR_COUNT_ARRAY])
        self.assertEqual(int(np.min(contributors[valid])), 1)
        self.assertEqual(int(np.median(contributors[valid])), 130)
        self.assertEqual(int(np.max(contributors[valid])), 134)
        self.assertAlmostEqual(
            float(np.mean(contributors[valid] == 134)), 0.387801, places=5
        )

    def test_observation_case_is_typed_resolvable_and_round_trips(self) -> None:
        case = build_mre134_observation_case(MRE_ROOT)

        self.assertEqual(case.source, CaseSource.DERIVED)
        self.assertEqual(case.validate_references(), ())
        self.assertEqual(len(case.assets), 12)
        self.assertEqual(case.units.pressure, "kPa")
        self.assertEqual(case.anatomy.mesh_asset_id, "mre134-adapted-vti")  # type: ignore[union-attr]
        self.assertEqual(
            len(case.anatomy.segmentation_labels["labels"]),  # type: ignore[union-attr,arg-type]
            34,
        )
        self.assertFalse(
            case.anatomy.segmentation_labels["label_volume_in_release"]  # type: ignore[union-attr]
        )
        mapping_asset = next(
            item
            for item in case.assets
            if item.asset_id == "mre134-recode-label-mapping"
        )
        self.assertEqual(mapping_asset.association, ValueAssociation.REGION)
        stiffness_asset = next(
            item
            for item in case.assets
            if item.asset_id == "mre134-stiffness-kpa-runtime"
        )
        self.assertEqual(stiffness_asset.association, ValueAssociation.VOXEL)
        self.assertEqual(stiffness_asset.units, "kPa")
        self.assertEqual(ObservationCase.from_json(case.to_json()), case)

    def test_comparison_refuses_raw_maxwell_response_value_comparison(self) -> None:
        simulation_case = load_observation_case(
            EXAMPLE_CASES_ROOT / "observation_case.json"
        )
        result = compare_mre134_with_case(self.loaded.case, simulation_case)

        self.assertTrue(result.metadata_comparison_valid)
        self.assertTrue(result.normalized_distribution_comparison_valid)
        self.assertFalse(result.spatial_comparison_ready)
        self.assertFalse(result.raw_value_comparison_valid)
        self.assertEqual(len(result.field_pairs), 2)
        self.assertFalse(any(pair.same_quantity for pair in result.field_pairs))
        self.assertFalse(result.raw_value_comparison_valid)

    def test_adapter_can_feed_generic_loader_and_vti_round_trip(self) -> None:
        generic = self.loaded.as_loaded_data()
        self.assertIsInstance(generic, LoadedData)
        self.assertIs(generic.mesh, self.loaded.dataset)
        self.assertEqual(generic.metadata["case_id"], self.loaded.case.case_id)

        with TemporaryDirectory(prefix="mre134-adapter-test-") as directory:
            output = self.loaded.write_vti(Path(directory) / "atlas.vti")
            restored = pv.read(output)

        self.assertEqual(restored.n_cells, self.loaded.dataset.n_cells)
        self.assertEqual(
            set(restored.cell_data.keys()), set(self.loaded.dataset.cell_data.keys())
        )
        self.assertEqual(restored.origin, self.loaded.dataset.origin)
        np.testing.assert_allclose(
            restored.direction_matrix,
            self.loaded.dataset.direction_matrix,
        )


if __name__ == "__main__":
    unittest.main()
