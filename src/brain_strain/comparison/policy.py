"""Decide whether an observation case can be compared with a simulation case.

The decision in this module is intentionally scoped to this repository's
visualisation capabilities.  It records whether scalar fields can be rendered
and compared side by side, with normalized colours, or by shared labels.  It
does not promote visual similarity to a precise physical calculation.

Inputs may be typed :class:`ObservationCase` instances, their dictionary
representations, or a ``LoadedData``-like object exposing a ``metadata``
dictionary.  This lets the same function operate on the currently loaded case
without weakening the structured result.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..observation_case import (
    CaseSource,
    DataAsset,
    ObservationCase,
    StrainMeasure,
    StrainObservation,
)


class LoadedCaseMetadata(Protocol):
    """Structural type for ``LoadedData`` without creating an import cycle."""

    metadata: Mapping[str, Any]


CaseInput = ObservationCase | Mapping[str, Any] | LoadedCaseMetadata | None

_FIELD_ROLE_TOKENS = (
    "result",
    "strain",
    "stress",
    "stiffness",
    "damping",
    "displacement",
    "pressure",
)
_SUPPORTED_DISPLAY_ASSOCIATIONS = frozenset({"cell"})


@dataclass(frozen=True, slots=True)
class ComparisonField:
    """Structured display and scientific semantics for one case field."""

    case_id: str
    field_name: str
    asset_id: str
    source_record: str
    quantity: str
    measure: str
    units: str
    association: str
    display_association: str
    coordinate_system: str | None
    scalar: bool
    temporal: bool
    frame_count: int | None
    is_simulated: bool


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """Pairwise visual and semantic compatibility of two fields."""

    current: ComparisonField
    simulation: ComparisonField
    visualisation_comparable: bool
    normalized_visualisation_comparable: bool
    spatial_overlay_ready: bool
    same_quantity: bool
    same_units: bool
    same_association: bool
    same_display_association: bool
    same_coordinate_system: bool
    same_temporal_semantics: bool
    raw_value_semantically_compatible: bool
    physical_calculation_applicable: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationCaseComparability:
    """Machine-readable decision for a current case and simulation case."""

    current_case_id: str | None
    simulation_case_id: str | None
    comparison_scope: str
    is_comparable: bool
    metadata_comparable: bool
    visualisation_comparable: bool
    physical_calculation_applicable: bool
    physical_comparison_valid: bool
    anatomy_compatibility: str
    label_mapping_declared: bool
    label_mapping_applicable: bool
    field_pairs: tuple[FieldComparison, ...]
    allowed_operations: tuple[str, ...]
    prohibited_interpretations: tuple[str, ...]
    issues: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


def _coerce_case(value: CaseInput, role: str) -> ObservationCase | None:
    if value is None:
        return None
    if isinstance(value, ObservationCase):
        return value
    candidate: Any = value
    if not isinstance(candidate, Mapping) and hasattr(candidate, "metadata"):
        candidate = candidate.metadata
    if isinstance(candidate, Mapping):
        if not candidate:
            return None
        try:
            return ObservationCase.from_dict(candidate)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {role} observation-case metadata") from exc
    raise TypeError(
        f"{role} case must be ObservationCase, metadata mapping, "
        "LoadedData-like object, or None"
    )


def _display_association(asset: DataAsset | None, association: str) -> str:
    if asset is not None:
        declared = asset.extensions.get("pyvista_association")
        if declared:
            return str(declared).casefold()
        if (asset.format or "").casefold() == "pyvista-cell-array":
            return "cell"
    return association.casefold()


def _observation_quantity(observation: StrainObservation) -> str:
    declared = observation.extensions.get("quantity_kind")
    if declared:
        return str(declared)
    measure = observation.measure
    if measure not in {StrainMeasure.OTHER, StrainMeasure.UNKNOWN}:
        return f"{measure.value}-strain"
    if "strain" in observation.name.casefold():
        return "strain-like"
    return measure.value


def _observation_field(
    case: ObservationCase,
    observation: StrainObservation,
    assets: Mapping[str, DataAsset],
) -> ComparisonField:
    asset = assets.get(observation.asset_id)
    association = observation.association.value
    frame_count = observation.frame_count
    temporal = bool(frame_count is not None and frame_count > 1)
    if observation.time_asset_id or observation.time_start is not None:
        temporal = True
    simulated = observation.is_simulated
    if simulated is None:
        simulated = case.source in {CaseSource.SIMULATION, CaseSource.SYNTHETIC}
    return ComparisonField(
        case_id=case.case_id,
        field_name=observation.name,
        asset_id=observation.asset_id,
        source_record="strain_observation",
        quantity=_observation_quantity(observation),
        measure=observation.measure.value,
        units=observation.units or (asset.units if asset else "unknown") or "unknown",
        association=association,
        display_association=_display_association(asset, association),
        coordinate_system=(
            observation.coordinate_system
            or (asset.coordinate_system if asset is not None else None)
        ),
        scalar=observation.tensor_order in (None, 0),
        temporal=temporal,
        frame_count=frame_count,
        is_simulated=bool(simulated),
    )


def _asset_field(case: ObservationCase, asset: DataAsset) -> ComparisonField:
    association = (
        asset.association.value if asset.association is not None else "unknown"
    )
    frame_count_value = asset.extensions.get("frame_count")
    frame_count = int(frame_count_value) if frame_count_value is not None else None
    temporal = bool(asset.extensions.get("temporal")) or bool(
        frame_count is not None and frame_count > 1
    )
    quantity = str(asset.extensions.get("quantity_kind") or asset.role)
    return ComparisonField(
        case_id=case.case_id,
        field_name=asset.field_names[0] if asset.field_names else asset.asset_id,
        asset_id=asset.asset_id,
        source_record="data_asset",
        quantity=quantity,
        measure=str(asset.extensions.get("measure") or "not-declared"),
        units=asset.units or "unknown",
        association=association,
        display_association=_display_association(asset, association),
        coordinate_system=asset.coordinate_system,
        scalar=bool(asset.extensions.get("scalar", True)),
        temporal=temporal,
        frame_count=frame_count,
        is_simulated=case.source in {CaseSource.SIMULATION, CaseSource.SYNTHETIC},
    )


def _case_fields(case: ObservationCase) -> tuple[ComparisonField, ...]:
    assets = {asset.asset_id: asset for asset in case.assets}
    fields = [
        _observation_field(case, observation, assets)
        for observation in case.strain_observations
    ]
    observed_assets = {field.asset_id for field in fields}
    fields.extend(
        _asset_field(case, asset)
        for asset in case.assets
        if asset.asset_id not in observed_assets
        and asset.extensions.get("comparison_role") == "field"
        and asset.field_names
    )
    if fields:
        return tuple(fields)

    # Legacy cases may have no StrainObservation records.  Retain a narrow,
    # explicit fallback for assets whose role clearly identifies result data.
    return tuple(
        _asset_field(case, asset)
        for asset in case.assets
        if asset.field_names
        and asset.association is not None
        and any(token in asset.role.casefold() for token in _FIELD_ROLE_TOKENS)
    )


def _normalise_anatomy(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _anatomy_compatibility(
    current: ObservationCase,
    simulation: ObservationCase,
) -> tuple[str, dict[str, str | None]]:
    current_name = current.anatomy.anatomy if current.anatomy else None
    simulation_name = simulation.anatomy.anatomy if simulation.anatomy else None
    detail = {"current": current_name, "simulation": simulation_name}
    if not current_name or not simulation_name:
        return "unknown", detail
    current_tokens = _normalise_anatomy(current_name)
    simulation_tokens = _normalise_anatomy(simulation_name)
    if current_tokens == simulation_tokens or (
        "brain" in current_tokens and "brain" in simulation_tokens
    ):
        return "compatible", detail
    return "incompatible", detail


def _case_identity_text(case: ObservationCase) -> str:
    values = [case.case_id, case.title, case.description or ""]
    values.extend(asset.location for asset in case.assets)
    values.extend(case.citations)
    return " ".join(values).casefold()


def _label_mapping_details(
    current: ObservationCase,
    simulation: ObservationCase,
) -> tuple[bool, bool, dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for role, case, other in (
        ("current", current, simulation),
        ("simulation", simulation, current),
    ):
        labels = case.anatomy.segmentation_labels if case.anatomy else {}
        mapping_file = labels.get("mapping_file")
        mapping_target = labels.get("mapping_target")
        if not mapping_file and not mapping_target:
            continue
        target_text = str(mapping_target or "").casefold()
        other_identity = _case_identity_text(other)
        target_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", target_text)
            if len(token) >= 4 and token not in frozenset({"part", "list", "active"})
        }
        applicable = bool(
            target_tokens
            and target_tokens & set(re.findall(r"[a-z0-9]+", other_identity))
        )
        declarations.append(
            {
                "declared_by": role,
                "mapping_file": mapping_file,
                "mapping_target": mapping_target,
                "label_count": len(labels.get("labels") or {}),
                "applicable_to_other_case": applicable,
            }
        )
    declared = bool(declarations)
    applicable = any(item["applicable_to_other_case"] for item in declarations)
    return declared, applicable, {"declarations": declarations}


def _compare_fields(
    current: ComparisonField,
    simulation: ComparisonField,
    anatomy_compatibility: str,
) -> FieldComparison:
    same_quantity = current.quantity.casefold() == simulation.quantity.casefold()
    same_units = current.units.casefold() == simulation.units.casefold()
    same_association = current.association == simulation.association
    same_display = current.display_association == simulation.display_association
    same_coordinates = bool(
        current.coordinate_system
        and current.coordinate_system == simulation.coordinate_system
    )
    same_temporal = current.temporal == simulation.temporal
    current_displayable = (
        current.scalar
        and current.display_association in _SUPPORTED_DISPLAY_ASSOCIATIONS
    )
    simulation_displayable = (
        simulation.scalar
        and simulation.display_association in _SUPPORTED_DISPLAY_ASSOCIATIONS
    )
    visualisation_comparable = (
        current_displayable
        and simulation_displayable
        and anatomy_compatibility != "incompatible"
    )
    spatial_overlay_ready = visualisation_comparable and same_coordinates
    semantic_raw_compatibility = all(
        (
            same_quantity,
            same_units,
            same_association,
            same_coordinates,
            same_temporal,
        )
    )

    limitations: list[str] = []
    if not same_quantity:
        limitations.append("different physical quantities")
    if not same_units:
        limitations.append("different or undeclared units")
    if not same_association:
        limitations.append("different scientific value associations")
    if not same_coordinates:
        limitations.append("no shared coordinate system or declared transform")
    if not same_temporal:
        limitations.append("different temporal semantics")
    if current.is_simulated or simulation.is_simulated:
        limitations.append("at least one field is simulated or synthetic")
    if not current_displayable or not simulation_displayable:
        limitations.append("at least one field is not a supported cell-scalar display")

    return FieldComparison(
        current=current,
        simulation=simulation,
        visualisation_comparable=visualisation_comparable,
        normalized_visualisation_comparable=visualisation_comparable,
        spatial_overlay_ready=spatial_overlay_ready,
        same_quantity=same_quantity,
        same_units=same_units,
        same_association=same_association,
        same_display_association=same_display,
        same_coordinate_system=same_coordinates,
        same_temporal_semantics=same_temporal,
        raw_value_semantically_compatible=semantic_raw_compatibility,
        physical_calculation_applicable=False,
        limitations=tuple(limitations),
    )


def decide_observation_case_comparability(
    current_case: CaseInput,
    simulation_case: CaseInput,
) -> ObservationCaseComparability:
    """Decide visualisation comparability while retaining physical details.

    ``is_comparable`` deliberately means *comparable in this visualisation
    tool*.  It is true when at least one scalar field pair can use the viewer's
    cell-data rendering path and the anatomies are not explicitly
    incompatible.  Coordinate, quantity, unit, temporal, label, and source
    details remain in the result and never make precise physical calculation
    applicable here.
    """
    current = _coerce_case(current_case, "current")
    simulation = _coerce_case(simulation_case, "simulation")
    missing: list[str] = []
    if current is None:
        missing.append("No current observation case is loaded.")
    if simulation is None:
        missing.append("No simulation observation case was supplied.")
    if missing:
        return ObservationCaseComparability(
            current_case_id=current.case_id if current else None,
            simulation_case_id=simulation.case_id if simulation else None,
            comparison_scope="visualisation-only",
            is_comparable=False,
            metadata_comparable=False,
            visualisation_comparable=False,
            physical_calculation_applicable=False,
            physical_comparison_valid=False,
            anatomy_compatibility="unknown",
            label_mapping_declared=False,
            label_mapping_applicable=False,
            field_pairs=(),
            allowed_operations=(),
            prohibited_interpretations=(
                "precise physical calculation or constitutive inference",
            ),
            issues=tuple(missing),
            details={"current_case_loaded": current is not None},
        )

    anatomy_status, anatomy_details = _anatomy_compatibility(current, simulation)
    current_fields = _case_fields(current)
    simulation_fields = _case_fields(simulation)
    pairs = tuple(
        _compare_fields(current_field, simulation_field, anatomy_status)
        for current_field in current_fields
        for simulation_field in simulation_fields
    )
    visualisation_comparable = any(pair.visualisation_comparable for pair in pairs)
    label_declared, label_applicable, label_details = _label_mapping_details(
        current, simulation
    )

    allowed = ["metadata and provenance comparison"]
    if visualisation_comparable:
        allowed.extend(
            (
                "side-by-side scalar rendering with independently labelled units",
                "within-field normalized or percentile colour comparison",
                "visual hotspot and spatial-concentration comparison",
            )
        )
    if any(pair.spatial_overlay_ready for pair in pairs):
        allowed.append("spatial overlay in the shared coordinate system")
    if label_applicable:
        allowed.append("region grouping through the declared label mapping")

    prohibited = [
        "precise physical calculation or constitutive inference",
        "clinical, injury-risk, or validation conclusions from visual similarity",
        "direct comparison of raw values unless separately validated "
        "outside this viewer",
    ]
    if not any(pair.spatial_overlay_ready for pair in pairs):
        prohibited.append("voxel-to-mesh overlay without registration and resampling")

    issues = [
        "Comparison is limited to visualisation; precise physical calculation "
        "is outside the tool's scope."
    ]
    if not current_fields:
        issues.append("The current case declares no visualisable result field.")
    if not simulation_fields:
        issues.append("The simulation case declares no visualisable result field.")
    if anatomy_status == "incompatible":
        issues.append("The cases declare incompatible anatomies.")
    elif anatomy_status == "unknown":
        issues.append(
            "Anatomy compatibility is unknown because metadata is incomplete."
        )
    if pairs and not any(pair.same_coordinate_system for pair in pairs):
        issues.append(
            "The fields have no shared coordinate system or declared transform."
        )
    if pairs and not any(pair.same_quantity for pair in pairs):
        issues.append("The fields represent different physical quantities.")
    if pairs and not any(pair.same_units for pair in pairs):
        issues.append("The field units differ or are undeclared.")
    if pairs and not any(pair.same_temporal_semantics for pair in pairs):
        issues.append("Static and time-varying field semantics differ.")
    if simulation.source is CaseSource.SYNTHETIC or any(
        field.is_simulated for field in simulation_fields
    ):
        issues.append(
            "The simulation field is synthetic or simulated and is for display only."
        )
    if label_declared and not label_applicable:
        issues.append(
            "A label mapping is declared, but its target is not identified "
            "as the simulation case."
        )
    if not visualisation_comparable and pairs:
        issues.append(
            "No field pair satisfies the viewer's cell-scalar display requirements."
        )

    current_reference_issues = current.validate_references()
    simulation_reference_issues = simulation.validate_references()
    if current_reference_issues:
        issues.append("The current case contains unresolved internal references.")
    if simulation_reference_issues:
        issues.append("The simulation case contains unresolved internal references.")

    return ObservationCaseComparability(
        current_case_id=current.case_id,
        simulation_case_id=simulation.case_id,
        comparison_scope="visualisation-only",
        is_comparable=visualisation_comparable,
        metadata_comparable=True,
        visualisation_comparable=visualisation_comparable,
        physical_calculation_applicable=False,
        physical_comparison_valid=False,
        anatomy_compatibility=anatomy_status,
        label_mapping_declared=label_declared,
        label_mapping_applicable=label_applicable,
        field_pairs=pairs,
        allowed_operations=tuple(allowed),
        prohibited_interpretations=tuple(prohibited),
        issues=tuple(issues),
        details={
            "decision_meaning": (
                "is_comparable reports viewer compatibility, not scientific "
                "equivalence or physical validation"
            ),
            "current_case": {
                "title": current.title,
                "source": current.source.value,
                "reference_issues": list(current_reference_issues),
                "field_count": len(current_fields),
            },
            "simulation_case": {
                "title": simulation.title,
                "source": simulation.source.value,
                "reference_issues": list(simulation_reference_issues),
                "field_count": len(simulation_fields),
            },
            "anatomy": anatomy_details,
            "label_mapping": label_details,
        },
    )


__all__ = [
    "ComparisonField",
    "FieldComparison",
    "ObservationCaseComparability",
    "decide_observation_case_comparability",
]
