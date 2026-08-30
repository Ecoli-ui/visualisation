"""Tests for visualisation-scoped ObservationCase comparability decisions."""

import unittest
from types import SimpleNamespace

from brain_strain.comparison.policy import decide_observation_case_comparability
from brain_strain.io.loader import load_metadata, load_observation_case
from brain_strain.observation_case import (
    CaseSource,
    DataAsset,
    ObservationCase,
    StrainMeasure,
    StrainObservation,
    ValueAssociation,
)
from brain_strain.paths import EXAMPLE_CASES_ROOT


class ObservationCaseComparabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mre_case = load_observation_case(
            EXAMPLE_CASES_ROOT / "mre134_observation_case.json"
        )
        cls.simulation_case = load_observation_case(
            EXAMPLE_CASES_ROOT / "observation_case.json"
        )

    def test_mre_and_demo_are_visualisable_but_not_physically_comparable(self) -> None:
        result = decide_observation_case_comparability(
            self.mre_case, self.simulation_case
        )

        self.assertTrue(result.is_comparable)
        self.assertTrue(result.visualisation_comparable)
        self.assertFalse(result.physical_calculation_applicable)
        self.assertFalse(result.physical_comparison_valid)
        self.assertEqual(result.comparison_scope, "visualisation-only")
        self.assertEqual(result.anatomy_compatibility, "compatible")
        self.assertEqual(len(result.field_pairs), 2)
        self.assertTrue(
            all(pair.same_display_association for pair in result.field_pairs)
        )
        self.assertFalse(any(pair.spatial_overlay_ready for pair in result.field_pairs))
        self.assertFalse(
            any(pair.raw_value_semantically_compatible for pair in result.field_pairs)
        )

    def test_decision_accepts_loaded_data_metadata_and_plain_mapping(self) -> None:
        loaded_like = SimpleNamespace(
            metadata=load_metadata(EXAMPLE_CASES_ROOT / "mre134_observation_case.json")
        )
        simulation_mapping = load_metadata(EXAMPLE_CASES_ROOT / "observation_case.json")

        result = decide_observation_case_comparability(loaded_like, simulation_mapping)

        self.assertEqual(result.current_case_id, self.mre_case.case_id)
        self.assertEqual(result.simulation_case_id, self.simulation_case.case_id)
        self.assertTrue(result.metadata_comparable)

    def test_missing_current_case_returns_structured_negative_decision(self) -> None:
        result = decide_observation_case_comparability(None, self.simulation_case)

        self.assertFalse(result.is_comparable)
        self.assertFalse(result.metadata_comparable)
        self.assertIn("No current observation case", " ".join(result.issues))

    def test_same_metadata_can_be_semantically_aligned_but_physics_stays_disabled(
        self,
    ) -> None:
        result = decide_observation_case_comparability(
            self.simulation_case, self.simulation_case
        )

        self.assertTrue(result.is_comparable)
        self.assertTrue(
            any(pair.raw_value_semantically_compatible for pair in result.field_pairs)
        )
        self.assertFalse(result.physical_calculation_applicable)
        self.assertFalse(result.physical_comparison_valid)

    def test_non_scalar_or_region_only_fields_are_not_viewer_comparable(self) -> None:
        region_asset = DataAsset(
            asset_id="region-result",
            role="region result",
            location="generated:region",
            field_names=("region_value",),
            association=ValueAssociation.REGION,
            units="1",
        )
        current = ObservationCase(
            case_id="region-case",
            title="Region-only case",
            source=CaseSource.DERIVED,
            assets=(region_asset,),
            strain_observations=(
                StrainObservation(
                    observation_id="region-observation",
                    name="region_value",
                    measure=StrainMeasure.OTHER,
                    asset_id=region_asset.asset_id,
                    association=ValueAssociation.REGION,
                    tensor_order=0,
                ),
            ),
        )

        result = decide_observation_case_comparability(current, self.simulation_case)

        self.assertFalse(result.is_comparable)
        self.assertFalse(result.visualisation_comparable)
        self.assertEqual(result.field_pairs[0].current.display_association, "region")
        self.assertIn("cell-scalar", " ".join(result.field_pairs[0].limitations))


if __name__ == "__main__":
    unittest.main()
