"""Tests for the strain-research observation metadata model."""

import unittest

from brain_strain.io.loader import DataLoadError, load_observation_case
from brain_strain.observation_case import (
    CaseSource,
    DataAsset,
    ObservationCase,
    StrainMeasure,
    StrainObservation,
    ValueAssociation,
)
from brain_strain.paths import EXAMPLE_CASES_ROOT, PROJECT_ROOT


class ObservationCaseTests(unittest.TestCase):
    def test_repository_example_is_complete_and_resolvable(self) -> None:
        case = load_observation_case(EXAMPLE_CASES_ROOT / "observation_case.json")

        self.assertEqual(case.source, CaseSource.SIMULATION)
        self.assertEqual(case.anatomy.node_count, 18617)  # type: ignore[union-attr]
        self.assertEqual(case.anatomy.element_count, 17030)  # type: ignore[union-attr]
        self.assertEqual(case.validate_references(), ())

    def test_json_round_trip_preserves_typed_values_and_tuples(self) -> None:
        original = load_observation_case(
            EXAMPLE_CASES_ROOT / "observation_case.json"
        )

        restored = ObservationCase.from_json(original.to_json())

        self.assertEqual(restored, original)
        self.assertIsInstance(restored.assets[1].shape, tuple)
        self.assertEqual(
            restored.strain_observations[0].measure, StrainMeasure.SHEAR
        )

    def test_unknown_properties_are_kept_as_extensions(self) -> None:
        case = ObservationCase.from_dict(
            {
                "case_id": "case-1",
                "title": "Future schema case",
                "source": "experiment",
                "future_study_property": {"value": 3},
                "subject": {
                    "subject_id": "subject-1",
                    "future_subject_property": True,
                },
            }
        )

        self.assertEqual(
            case.extensions["future_study_property"], {"value": 3}
        )
        subject = case.subject
        self.assertIsNotNone(subject)
        assert subject is not None
        self.assertTrue(subject.extensions["future_subject_property"])

    def test_duplicate_ids_are_rejected(self) -> None:
        asset = DataAsset("mesh", "mesh", "mesh.vtu")
        with self.assertRaisesRegex(ValueError, "Duplicate asset_id"):
            ObservationCase(
                case_id="case-1",
                title="Duplicate assets",
                source=CaseSource.SIMULATION,
                assets=(asset, asset),
            )

    def test_unresolved_references_are_reported(self) -> None:
        case = ObservationCase(
            case_id="case-1",
            title="Partial legacy case",
            source="simulation",  # type: ignore[arg-type]
            strain_observations=(
                StrainObservation(
                    observation_id="mps",
                    name="MPS",
                    measure="maximum-principal",  # type: ignore[arg-type]
                    asset_id="missing-result",
                    association="cell",  # type: ignore[arg-type]
                ),
            ),
        )

        self.assertEqual(case.source, CaseSource.SIMULATION)
        self.assertEqual(
            case.strain_observations[0].association, ValueAssociation.CELL
        )
        self.assertIn("unknown asset 'missing-result'", case.validate_references()[0])

    def test_invalid_typed_metadata_is_wrapped_as_load_error(self) -> None:
        with self.assertRaises(DataLoadError):
            load_observation_case(PROJECT_ROOT / "README.md")


if __name__ == "__main__":
    unittest.main()
