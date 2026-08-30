"""Research metadata model for one strain observation case.

The numerical arrays used by this project can be very large, so an
``ObservationCase`` records *what* was observed and points to the files that
contain the mesh, images, kinematics, and result arrays.  It can describe a
physical experiment, a clinical/in-vivo measurement, a finite-element
simulation, synthetic demonstration data, or a hybrid validation case.

Every section except the case identity and source type is optional.  This is
intentional: older datasets are often incomplete, while ``extensions`` on
the case and its component records preserve information that is not yet part
of the common vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    ClassVar,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

JsonObject = dict[str, Any]


class CaseSource(StrEnum):
    """Origin of the observation values."""

    EXPERIMENT = "experiment"
    IN_VIVO = "in-vivo"
    SIMULATION = "simulation"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class StrainMeasure(StrEnum):
    """Common strain measures; ``OTHER`` is available for extensions."""

    INFINITESIMAL = "infinitesimal"
    ENGINEERING = "engineering"
    GREEN_LAGRANGE = "green-lagrange"
    ALMANSI = "almansi"
    LOGARITHMIC = "logarithmic"
    DEFORMATION_GRADIENT = "deformation-gradient"
    PRINCIPAL = "principal"
    MAXIMUM_PRINCIPAL = "maximum-principal"
    MINIMUM_PRINCIPAL = "minimum-principal"
    SHEAR = "shear"
    OCTAHEDRAL_SHEAR = "octahedral-shear"
    EQUIVALENT = "equivalent"
    VOLUMETRIC = "volumetric"
    FIBRE = "fibre"
    STRAIN_RATE = "strain-rate"
    OTHER = "other"
    UNKNOWN = "unknown"


class ValueAssociation(StrEnum):
    """Spatial entity on which a result is defined."""

    POINT = "point"
    CELL = "cell"
    FIELD = "field"
    VOXEL = "voxel"
    REGION = "region"
    SENSOR = "sensor"
    SPECIMEN = "specimen"
    GLOBAL = "global"
    UNKNOWN = "unknown"


def _extension_field() -> Any:
    return field(default_factory=dict)


@dataclass(slots=True)
class PersonOrOrganisation:
    """Contributor, institution, laboratory, or data custodian."""

    name: str
    role: str | None = None
    affiliation: str | None = None
    identifier: str | None = None
    contact: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Subject:
    """Human, animal, cadaver, phantom, or computational subject."""

    subject_id: str
    subject_type: str = "human"
    cohort: str | None = None
    species: str | None = None
    strain_or_breed: str | None = None
    sex: str | None = None
    gender: str | None = None
    age: float | None = None
    age_unit: str | None = None
    mass: float | None = None
    mass_unit: str | None = None
    height: float | None = None
    height_unit: str | None = None
    anthropometry: JsonObject = _extension_field()
    clinical_history: JsonObject = _extension_field()
    inclusion_criteria: tuple[str, ...] = ()
    exclusion_criteria: tuple[str, ...] = ()
    deidentified: bool | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Specimen:
    """Tissue/sample information for ex-vivo or material testing."""

    specimen_id: str
    tissue: str | None = None
    anatomical_region: str | None = None
    laterality: str | None = None
    orientation: str | None = None
    dimensions: JsonObject = _extension_field()
    collection_method: str | None = None
    post_mortem_interval: float | None = None
    post_mortem_interval_unit: str | None = None
    storage: str | None = None
    preparation: str | None = None
    temperature: float | None = None
    temperature_unit: str | None = None
    hydration: str | None = None
    preconditioning: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class UnitSystem:
    """Units needed to interpret geometry, time, loading, and mechanics."""

    name: str | None = None
    length: str | None = None
    time: str | None = None
    mass: str | None = None
    force: str | None = None
    pressure: str | None = None
    acceleration: str | None = None
    angular_velocity: str | None = None
    angular_acceleration: str | None = None
    strain: str = "1"
    strain_rate: str | None = None
    temperature: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class CoordinateSystem:
    """Spatial frame and conventions used by geometry and result tensors."""

    name: str
    convention: str | None = None
    origin: tuple[float, float, float] | None = None
    axes: JsonObject = _extension_field()
    reference: str | None = None
    transform_to_reference: tuple[tuple[float, ...], ...] | None = None
    handedness: str | None = None
    moving: bool | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class DataAsset:
    """Reference to one file, URI, embedded mesh array, or database object."""

    asset_id: str
    role: str
    location: str
    format: str | None = None
    media_type: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None
    byte_size: int | None = None
    field_names: tuple[str, ...] = ()
    association: ValueAssociation | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    units: str | None = None
    coordinate_system: str | None = None
    description: str | None = None
    generated_by: str | None = None
    extensions: JsonObject = _extension_field()

    def __post_init__(self) -> None:
        if self.association is not None:
            self.association = _enum_value(ValueAssociation, self.association)


@dataclass(slots=True)
class AnatomyModel:
    """Imaging, segmentation, geometry, and anatomical-region provenance."""

    anatomy: str = "brain/head"
    imaging_modality: str | None = None
    imaging_protocol: JsonObject = _extension_field()
    image_asset_ids: tuple[str, ...] = ()
    segmentation_method: str | None = None
    segmentation_software: str | None = None
    segmentation_labels: JsonObject = _extension_field()
    mesh_asset_id: str | None = None
    mesh_stage: str | None = None
    mesh_type: str | None = None
    element_types: tuple[str, ...] = ()
    node_count: int | None = None
    element_count: int | None = None
    regions: JsonObject = _extension_field()
    sets: JsonObject = _extension_field()
    mesh_quality: JsonObject = _extension_field()
    centre_of_gravity: tuple[float, float, float] | None = None
    coordinate_system: str | None = None
    registration: JsonObject = _extension_field()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class LoadingCondition:
    """Impact, motion, or material-test input applied to the case."""

    event_type: str | None = None
    description: str | None = None
    date_time: str | None = None
    loading_mode: str | None = None
    direction: tuple[float, float, float] | None = None
    location: tuple[float, float, float] | None = None
    reference_point: tuple[float, float, float] | None = None
    coordinate_system: str | None = None
    duration: float | None = None
    time_unit: str | None = None
    impactor: JsonObject = _extension_field()
    boundary_conditions: JsonObject = _extension_field()
    initial_conditions: JsonObject = _extension_field()
    linear_kinematics_asset_ids: tuple[str, ...] = ()
    angular_kinematics_asset_ids: tuple[str, ...] = ()
    force_asset_ids: tuple[str, ...] = ()
    displacement_asset_ids: tuple[str, ...] = ()
    loading_rate: float | None = None
    loading_rate_unit: str | None = None
    peak_values: JsonObject = _extension_field()
    filtering: JsonObject = _extension_field()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Instrument:
    """Sensor, scanner, test machine, or reconstruction system."""

    instrument_id: str
    kind: str
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    location: str | None = None
    orientation: str | None = None
    coordinate_system: str | None = None
    sampling_rate: float | None = None
    sampling_rate_unit: str | None = None
    calibration: JsonObject = _extension_field()
    accuracy: JsonObject = _extension_field()
    channels: JsonObject = _extension_field()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Acquisition:
    """Experimental or clinical acquisition and synchronisation details."""

    method: str
    protocol: str | None = None
    laboratory: str | None = None
    operators: tuple[str, ...] = ()
    instruments: tuple[Instrument, ...] = ()
    sampling_rate: float | None = None
    sampling_rate_unit: str | None = None
    trigger: str | None = None
    synchronisation: str | None = None
    calibration: JsonObject = _extension_field()
    filtering: JsonObject = _extension_field()
    environmental_conditions: JsonObject = _extension_field()
    repetitions: int | None = None
    raw_asset_ids: tuple[str, ...] = ()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class MaterialModel:
    """Constitutive assignment for one tissue, part, or region."""

    material_id: str
    name: str
    region_ids: tuple[str, ...] = ()
    constitutive_model: str | None = None
    parameters: JsonObject = _extension_field()
    density: float | None = None
    density_unit: str | None = None
    anisotropy: JsonObject = _extension_field()
    viscoelasticity: JsonObject = _extension_field()
    source: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Simulation:
    """Finite-element or other computational model configuration."""

    solver: str
    solver_version: str | None = None
    model_asset_ids: tuple[str, ...] = ()
    run_asset_id: str | None = None
    materials: tuple[MaterialModel, ...] = ()
    sections: JsonObject = _extension_field()
    contacts: JsonObject = _extension_field()
    constraints: JsonObject = _extension_field()
    controls: JsonObject = _extension_field()
    output_requests: JsonObject = _extension_field()
    element_formulation: JsonObject = _extension_field()
    timestep: float | None = None
    termination_time: float | None = None
    damping: JsonObject = _extension_field()
    mass_scaling: JsonObject = _extension_field()
    hardware: JsonObject = _extension_field()
    status: str | None = None
    warnings: tuple[str, ...] = ()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class StrainObservation:
    """Definition and storage of one strain or strain-rate field."""

    observation_id: str
    name: str
    measure: StrainMeasure
    asset_id: str
    association: ValueAssociation
    units: str = "1"
    components: tuple[str, ...] = ()
    tensor_order: int | None = None
    coordinate_system: str | None = None
    reference_configuration: str | None = None
    direction: str | None = None
    fibre_definition: str | None = None
    spatial_location: str | None = None
    time_asset_id: str | None = None
    time_start: float | None = None
    time_end: float | None = None
    frame_count: int | None = None
    calculation_method: str | None = None
    software: str | None = None
    filtering: JsonObject = _extension_field()
    missing_value: float | str | None = None
    valid_range: tuple[float, float] | None = None
    is_simulated: bool | None = None
    extensions: JsonObject = _extension_field()

    def __post_init__(self) -> None:
        self.measure = _enum_value(StrainMeasure, self.measure)
        self.association = _enum_value(ValueAssociation, self.association)


@dataclass(slots=True)
class ProcessingStep:
    """One reproducible transformation in the data provenance chain."""

    step_id: str
    name: str
    description: str | None = None
    software: str | None = None
    software_version: str | None = None
    code_reference: str | None = None
    parameters: JsonObject = _extension_field()
    input_asset_ids: tuple[str, ...] = ()
    output_asset_ids: tuple[str, ...] = ()
    started_at: str | None = None
    ended_at: str | None = None
    performed_by: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class DerivedResult:
    """Summary metric, hotspot, history, injury predictor, or statistic."""

    result_id: str
    name: str
    value: Any = None
    units: str | None = None
    method: str | None = None
    source_observation_ids: tuple[str, ...] = ()
    asset_id: str | None = None
    time: float | None = None
    frame_index: int | None = None
    element_id: int | str | None = None
    region: str | None = None
    position: tuple[float, float, float] | None = None
    coordinate_system: str | None = None
    threshold: float | None = None
    uncertainty: JsonObject = _extension_field()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Outcome:
    """Independent clinical, behavioural, histological, or injury outcome."""

    outcome_id: str
    category: str
    name: str
    value: Any = None
    units: str | None = None
    scale: str | None = None
    definition: str | None = None
    assessed_at: str | None = None
    time_from_event: float | None = None
    time_unit: str | None = None
    region: str | None = None
    assessor: str | None = None
    blinded: bool | None = None
    asset_id: str | None = None
    uncertainty: JsonObject = _extension_field()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class QualityAssessment:
    """Quality control, validation, uncertainty, and limitations."""

    status: str | None = None
    mesh_checks: JsonObject = _extension_field()
    signal_checks: JsonObject = _extension_field()
    convergence: JsonObject = _extension_field()
    validation: JsonObject = _extension_field()
    uncertainty: JsonObject = _extension_field()
    sensitivity: JsonObject = _extension_field()
    missing_data: JsonObject = _extension_field()
    exclusions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    reviewed_by: tuple[str, ...] = ()
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class Governance:
    """Ethics, consent, privacy, access, licensing, and retention."""

    ethics_approval: str | None = None
    consent: str | None = None
    privacy_classification: str | None = None
    deidentification: str | None = None
    access_conditions: str | None = None
    licence: str | None = None
    data_use_agreement: str | None = None
    retention_policy: str | None = None
    extensions: JsonObject = _extension_field()


@dataclass(slots=True)
class ObservationCase:
    """Complete, extensible description of one strain-research case.

    Large or format-specific data belong in :class:`DataAsset` entries.  The
    remaining fields capture the scientific context needed to interpret,
    reproduce, compare, and govern those data.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    case_id: str
    title: str
    source: CaseSource
    schema_version: str = SCHEMA_VERSION
    description: str | None = None
    study_id: str | None = None
    research_question: str | None = None
    hypothesis: str | None = None
    created_at: str | None = None
    observed_at: str | None = None
    contributors: tuple[PersonOrOrganisation, ...] = ()
    subject: Subject | None = None
    specimen: Specimen | None = None
    units: UnitSystem = field(default_factory=UnitSystem)
    coordinate_systems: tuple[CoordinateSystem, ...] = ()
    anatomy: AnatomyModel | None = None
    loading: LoadingCondition | None = None
    acquisition: Acquisition | None = None
    simulation: Simulation | None = None
    strain_observations: tuple[StrainObservation, ...] = ()
    derived_results: tuple[DerivedResult, ...] = ()
    outcomes: tuple[Outcome, ...] = ()
    assets: tuple[DataAsset, ...] = ()
    processing: tuple[ProcessingStep, ...] = ()
    quality: QualityAssessment = field(default_factory=QualityAssessment)
    governance: Governance = field(default_factory=Governance)
    publications: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    related_case_ids: tuple[str, ...] = ()
    extensions: JsonObject = _extension_field()

    def __post_init__(self) -> None:
        self.source = _enum_value(CaseSource, self.source)
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.title.strip():
            raise ValueError("title cannot be empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version cannot be empty")

        self._require_unique((item.asset_id for item in self.assets), "asset_id")
        self._require_unique(
            (item.observation_id for item in self.strain_observations),
            "observation_id",
        )
        self._require_unique(
            (item.result_id for item in self.derived_results), "result_id"
        )
        self._require_unique((item.outcome_id for item in self.outcomes), "outcome_id")
        self._require_unique((item.step_id for item in self.processing), "step_id")

    @staticmethod
    def _require_unique(values: Any, description: str) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            duplicates.add(value) if value in seen else seen.add(value)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate {description} values: {names}")

    def validate_references(self) -> tuple[str, ...]:
        """Return unresolved internal references without rejecting partial data."""
        assets = {item.asset_id for item in self.assets}
        coordinate_systems = {item.name for item in self.coordinate_systems}
        processing_steps = {item.step_id for item in self.processing}
        observations = {item.observation_id for item in self.strain_observations}
        issues: list[str] = []

        def check(asset_id: str | None, owner: str) -> None:
            if asset_id and asset_id not in assets:
                issues.append(f"{owner} references unknown asset {asset_id!r}")

        def check_coordinates(name: str | None, owner: str) -> None:
            if name and name not in coordinate_systems:
                issues.append(f"{owner} references unknown coordinate system {name!r}")

        for item in self.assets:
            check_coordinates(item.coordinate_system, f"assets[{item.asset_id}]")
            if item.generated_by and item.generated_by not in processing_steps:
                issues.append(
                    f"assets[{item.asset_id}] references unknown processing "
                    f"step {item.generated_by!r}"
                )

        if self.anatomy:
            check(self.anatomy.mesh_asset_id, "anatomy.mesh_asset_id")
            check_coordinates(
                self.anatomy.coordinate_system, "anatomy.coordinate_system"
            )
            for value in self.anatomy.image_asset_ids:
                check(value, "anatomy.image_asset_ids")
        if self.loading:
            check_coordinates(
                self.loading.coordinate_system, "loading.coordinate_system"
            )
            groups = (
                self.loading.linear_kinematics_asset_ids,
                self.loading.angular_kinematics_asset_ids,
                self.loading.force_asset_ids,
                self.loading.displacement_asset_ids,
            )
            for group in groups:
                for value in group:
                    check(value, "loading")
        if self.acquisition:
            for value in self.acquisition.raw_asset_ids:
                check(value, "acquisition.raw_asset_ids")
            for item in self.acquisition.instruments:
                check_coordinates(
                    item.coordinate_system,
                    f"acquisition.instruments[{item.instrument_id}]",
                )
        if self.simulation:
            check(self.simulation.run_asset_id, "simulation.run_asset_id")
            for value in self.simulation.model_asset_ids:
                check(value, "simulation.model_asset_ids")
        for item in self.strain_observations:
            check(item.asset_id, f"strain_observations[{item.observation_id}]")
            check(item.time_asset_id, f"strain_observations[{item.observation_id}]")
            check_coordinates(
                item.coordinate_system,
                f"strain_observations[{item.observation_id}]",
            )
        for item in self.derived_results:
            check(item.asset_id, f"derived_results[{item.result_id}]")
            check_coordinates(
                item.coordinate_system, f"derived_results[{item.result_id}]"
            )
            for value in item.source_observation_ids:
                if value not in observations:
                    issues.append(
                        f"derived_results[{item.result_id}] references unknown "
                        f"observation {value!r}"
                    )
        for item in self.outcomes:
            check(item.asset_id, f"outcomes[{item.outcome_id}]")
        for item in self.processing:
            for value in (*item.input_asset_ids, *item.output_asset_ids):
                check(value, f"processing[{item.step_id}]")
        return tuple(issues)

    def to_dict(self, *, omit_none: bool = False) -> JsonObject:
        """Return a JSON-compatible nested dictionary."""
        value = _to_json_value(self)
        if not isinstance(value, dict):  # pragma: no cover - defensive guard
            raise TypeError("ObservationCase did not serialise to an object")
        return _remove_none(value) if omit_none else value

    def to_json(self, *, indent: int = 2, omit_none: bool = False) -> str:
        """Serialise the case as JSON."""
        return json.dumps(
            self.to_dict(omit_none=omit_none),
            indent=indent,
            ensure_ascii=False,
        )

    def write_json(
        self, path: str | Path, *, indent: int = 2, omit_none: bool = False
    ) -> Path:
        """Write the case to a UTF-8 JSON file and return its path."""
        output = Path(path)
        output.write_text(
            self.to_json(indent=indent, omit_none=omit_none) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObservationCase:
        """Construct a case from its JSON-compatible representation.

        Unknown top-level keys are preserved under ``extensions`` so newer
        metadata can still be opened by this version of the repository.
        """
        data = dict(value)
        known = {item.name for item in fields(cls)}
        extensions = dict(data.get("extensions") or {})
        extensions.update(
            {key: data.pop(key) for key in tuple(data) if key not in known}
        )
        data["extensions"] = extensions

        data["source"] = _enum_value(CaseSource, data.get("source", "unknown"))
        data["contributors"] = _records(PersonOrOrganisation, data.get("contributors"))
        data["subject"] = _record(Subject, data.get("subject"))
        data["specimen"] = _record(Specimen, data.get("specimen"))
        data["units"] = _record(UnitSystem, data.get("units")) or UnitSystem()
        data["coordinate_systems"] = _records(
            CoordinateSystem, data.get("coordinate_systems")
        )
        data["anatomy"] = _record(AnatomyModel, data.get("anatomy"))
        data["loading"] = _record(LoadingCondition, data.get("loading"))
        data["acquisition"] = _acquisition(data.get("acquisition"))
        data["simulation"] = _simulation(data.get("simulation"))
        data["strain_observations"] = _strain_observations(
            data.get("strain_observations")
        )
        data["derived_results"] = _records(DerivedResult, data.get("derived_results"))
        data["outcomes"] = _records(Outcome, data.get("outcomes"))
        data["assets"] = _data_assets(data.get("assets"))
        data["processing"] = _records(ProcessingStep, data.get("processing"))
        data["quality"] = (
            _record(QualityAssessment, data.get("quality")) or QualityAssessment()
        )
        data["governance"] = _record(Governance, data.get("governance")) or Governance()

        for name in (
            "publications",
            "citations",
            "keywords",
            "related_case_ids",
        ):
            data[name] = tuple(data.get(name) or ())
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> ObservationCase:
        """Construct a case from a JSON string."""
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("ObservationCase JSON must contain a top-level object")
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> ObservationCase:
        """Read an observation case from a UTF-8 JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


T = TypeVar("T")


def _enum_value(enum_type: type[T], value: Any) -> T:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _record(record_type: type[T], value: Any) -> T | None:
    if value is None or isinstance(value, record_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{record_type.__name__} must be an object")
    data = dict(value)
    allowed = {item.name for item in fields(record_type)}
    if "extensions" in allowed:
        extensions = dict(data.get("extensions") or {})
        extensions.update(
            {key: data.pop(key) for key in tuple(data) if key not in allowed}
        )
        data["extensions"] = extensions
    hints = get_type_hints(record_type)
    for name, annotation in hints.items():
        if name in data:
            data[name] = _coerce_tuples(data[name], annotation)
    return record_type(**data)


def _records(record_type: type[T], values: Any) -> tuple[T, ...]:
    result: list[T] = []
    for value in values or ():
        item = _record(record_type, value)
        if item is None:
            raise TypeError(f"{record_type.__name__} array cannot contain null")
        result.append(item)
    return tuple(result)


def _coerce_tuples(value: Any, annotation: Any) -> Any:
    """Restore tuples that JSON necessarily represented as arrays."""
    if value is None:
        return None
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        tuple_annotation = next(
            (item for item in arguments if get_origin(item) is tuple), None
        )
        return (
            _coerce_tuples(value, tuple_annotation)
            if tuple_annotation is not None
            else value
        )
    if origin is not tuple:
        return value
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(_coerce_tuples(item, arguments[0]) for item in value)
    return tuple(
        _coerce_tuples(item, arguments[index]) if index < len(arguments) else item
        for index, item in enumerate(value)
    )


def _data_assets(values: Any) -> tuple[DataAsset, ...]:
    result: list[DataAsset] = []
    for value in values or ():
        item = value if isinstance(value, DataAsset) else dict(value)
        if isinstance(item, dict) and item.get("association") is not None:
            item["association"] = _enum_value(ValueAssociation, item["association"])
        record = item if isinstance(item, DataAsset) else _record(DataAsset, item)
        if record is None:  # pragma: no cover - guarded by the mapping conversion
            raise TypeError("DataAsset array cannot contain null")
        result.append(record)
    return tuple(result)


def _strain_observations(values: Any) -> tuple[StrainObservation, ...]:
    result: list[StrainObservation] = []
    for value in values or ():
        if isinstance(value, StrainObservation):
            result.append(value)
            continue
        item = dict(value)
        item["measure"] = _enum_value(StrainMeasure, item.get("measure", "unknown"))
        item["association"] = _enum_value(
            ValueAssociation, item.get("association", "unknown")
        )
        result.append(_record(StrainObservation, item))  # type: ignore[arg-type]
    return tuple(result)


def _acquisition(value: Any) -> Acquisition | None:
    if value is None or isinstance(value, Acquisition):
        return value
    item = dict(value)
    item["instruments"] = _records(Instrument, item.get("instruments"))
    return _record(Acquisition, item)


def _simulation(value: Any) -> Simulation | None:
    if value is None or isinstance(value, Simulation):
        return value
    item = dict(value)
    item["materials"] = _records(MaterialModel, item.get("materials"))
    return _record(Simulation, item)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _to_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    return value


def _remove_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_remove_none(item) for item in value]
    return value


__all__ = [
    "Acquisition",
    "AnatomyModel",
    "CaseSource",
    "CoordinateSystem",
    "DataAsset",
    "DerivedResult",
    "Governance",
    "Instrument",
    "LoadingCondition",
    "MaterialModel",
    "ObservationCase",
    "Outcome",
    "PersonOrOrganisation",
    "ProcessingStep",
    "QualityAssessment",
    "Simulation",
    "Specimen",
    "StrainMeasure",
    "StrainObservation",
    "Subject",
    "UnitSystem",
    "ValueAssociation",
]
