"""Adapter for loading MRE134 as a PyVista data set and ObservationCase.

The source stiffness archive stores values near 2600 even though the study
reports kPa. The adapter exposes stiffness in kPa after dividing by 1000,
keeps damping ratio dimensionless, preserves the MNI affine, and represents
NIfTI samples as PyVista cells whose centres coincide with voxel centres.

MRE material properties and the repository's generalized Maxwell response
are not like-for-like response fields. :func:`compare_mre134_with_case`
reports which structural comparisons are safe and which transforms or
physical results are still required.
"""

from __future__ import annotations

import json
import math
import re
import struct
import zipfile
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import numpy.typing as npt
import pyvista as pv

from ..comparison.policy import decide_observation_case_comparability
from ..observation_case import (
    Acquisition,
    AnatomyModel,
    CaseSource,
    CoordinateSystem,
    DataAsset,
    DerivedResult,
    Governance,
    Instrument,
    LoadingCondition,
    ObservationCase,
    ProcessingStep,
    QualityAssessment,
    UnitSystem,
    ValueAssociation,
)
from ..paths import (
    DEFAULT_MRE134_ROOT,
    DEFAULT_SIMULATION_CASE,
    EXAMPLE_CASES_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
)
from .mre134_labels import (
    DEFAULT_MAPPING_OUTPUT,
    MRE134_RECODE_LABEL_MAPPINGS,
    observation_case_label_metadata,
    write_mapping,
)

STIFFNESS_ARRAY = "MRE134_Stiffness_kPa"
DAMPING_ARRAY = "MRE134_DampingRatio"
VALID_MASK_ARRAY = "MRE134_ValidMask"
CONTRIBUTOR_COUNT_ARRAY = "MRE134_ContributorCount"
ACTUATION_FREQUENCY_HZ = 50.0
STIFFNESS_RAW_TO_KPA = 1.0 / 1000.0
XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class MRE134AdapterError(ValueError):
    """Raised when the MRE134 bundle is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class MRE134Participant:
    """One deidentified row from ``MRE134_Demographics.xlsx``."""

    subject_id: str
    age_years: float
    sex: str
    original_study: str


@dataclass(slots=True)
class LoadedMRE134:
    """Loaded MRE134 atlas, typed metadata, and cohort demographics."""

    dataset: pv.ImageData
    case: ObservationCase
    participants: tuple[MRE134Participant, ...]
    source_root: Path

    def write_vti(self, path: str | Path) -> Path:
        """Write the adapted cell-data atlas to a compressed VTI file."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.save(output, binary=True)
        return output

    def as_loaded_data(self) -> Any:
        """Return the repository's generic ``LoadedData`` wrapper lazily."""
        from ..io.loader import LoadedData

        return LoadedData(
            mesh=self.dataset,
            time=None,
            metadata=self.case.to_dict(omit_none=True),
            frames=(self.dataset,),
        )


@dataclass(frozen=True, slots=True)
class FieldCompatibility:
    """Compatibility of one MRE property with one comparison-case field."""

    mre_field: str
    other_field: str
    mre_quantity: str
    other_quantity: str
    mre_units: str
    other_units: str
    same_quantity: bool
    same_units: bool
    same_coordinate_system: bool
    same_association: bool
    same_temporal_semantics: bool
    raw_value_comparison_valid: bool


@dataclass(frozen=True, slots=True)
class MRE134CaseComparison:
    """Machine-readable compatibility report for two observation cases."""

    mre_case_id: str
    other_case_id: str
    metadata_comparison_valid: bool
    normalized_distribution_comparison_valid: bool
    spatial_comparison_ready: bool
    raw_value_comparison_valid: bool
    field_pairs: tuple[FieldCompatibility, ...]
    issues: tuple[str, ...]
    allowed_comparisons: tuple[str, ...]
    required_for_physical_comparison: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


@dataclass(frozen=True, slots=True)
class _NiftiHeader:
    endian: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    voxel_offset: int
    slope: float
    intercept: float
    voxel_size: tuple[float, ...]
    affine: npt.NDArray[np.float64]
    qform_code: int
    sform_code: int


def _parse_nifti_header(header: bytes) -> _NiftiHeader:
    if len(header) < 352:
        raise MRE134AdapterError("NIfTI header is truncated")
    if struct.unpack_from("<i", header, 0)[0] == 348:
        endian = "<"
    elif struct.unpack_from(">i", header, 0)[0] == 348:
        endian = ">"
    else:
        raise MRE134AdapterError("Input is not a NIfTI-1 single-file image")

    dimensions = struct.unpack_from(endian + "8h", header, 40)
    rank = int(dimensions[0])
    shape = tuple(int(value) for value in dimensions[1 : rank + 1])
    datatype = int(struct.unpack_from(endian + "h", header, 70)[0])
    dtype_codes = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8"}
    if datatype not in dtype_codes:
        raise MRE134AdapterError(f"Unsupported NIfTI datatype code {datatype}")
    pixdim = struct.unpack_from(endian + "8f", header, 76)
    voxel_size = tuple(abs(float(value)) for value in pixdim[1 : rank + 1])
    voxel_offset = int(struct.unpack_from(endian + "f", header, 108)[0])
    slope, intercept = struct.unpack_from(endian + "ff", header, 112)
    qform_code, sform_code = struct.unpack_from(endian + "hh", header, 252)
    affine = np.array(
        [
            struct.unpack_from(endian + "4f", header, 280),
            struct.unpack_from(endian + "4f", header, 296),
            struct.unpack_from(endian + "4f", header, 312),
            (0.0, 0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    if not sform_code or not np.any(affine[:3, :3]):
        affine = _qform_affine(header, endian, voxel_size)
    return _NiftiHeader(
        endian=endian,
        shape=shape,
        dtype=np.dtype(endian + dtype_codes[datatype]),
        voxel_offset=voxel_offset,
        slope=float(slope),
        intercept=float(intercept),
        voxel_size=voxel_size,
        affine=affine,
        qform_code=int(qform_code),
        sform_code=int(sform_code),
    )


def _qform_affine(
    header: bytes, endian: str, voxel_size: tuple[float, ...]
) -> npt.NDArray[np.float64]:
    b, c, d = struct.unpack_from(endian + "3f", header, 256)
    x, y, z = struct.unpack_from(endian + "3f", header, 268)
    pixdim = struct.unpack_from(endian + "8f", header, 76)
    a_squared = max(0.0, 1.0 - b * b - c * c - d * d)
    a = math.sqrt(a_squared)
    rotation = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
        ],
        dtype=np.float64,
    )
    scales = np.array(voxel_size[:3], dtype=np.float64)
    if pixdim[0] < 0:
        scales[2] *= -1.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ np.diag(scales)
    affine[:3, 3] = (x, y, z)
    return affine


def _read_zipped_nifti(path: Path, member: str) -> tuple[np.ndarray, _NiftiHeader]:
    try:
        with zipfile.ZipFile(path) as archive:
            blob = archive.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise MRE134AdapterError(f"Could not read {member!r} from {path}") from exc
    info = _parse_nifti_header(blob[:352])
    count = math.prod(info.shape)
    values = np.frombuffer(
        blob,
        dtype=info.dtype,
        count=count,
        offset=info.voxel_offset,
    ).copy()
    if info.slope not in (0.0, 1.0):
        values *= info.slope
    if info.intercept:
        values += info.intercept
    return values, info


def _read_nifti_presence(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            header_bytes = handle.read(352)
    except OSError as exc:
        raise MRE134AdapterError(f"Could not read individual map {path}") from exc
    info = _parse_nifti_header(header_bytes)
    if info.shape != expected_shape:
        raise MRE134AdapterError(
            f"Grid mismatch in {path.name}: {info.shape} != {expected_shape}"
        )
    if info.slope not in (0.0, 1.0) or info.intercept:
        raise MRE134AdapterError(
            f"Scaled individual map is unsupported for contributor count: {path}"
        )
    values = np.memmap(
        path,
        dtype=info.dtype,
        mode="r",
        offset=info.voxel_offset,
        shape=(math.prod(info.shape),),
    )
    return np.isfinite(values) & (values > 0)


def _shared_string_values(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{XML_NS['m']}}}t"))
        for item in root.findall("m:si", XML_NS)
    ]


def _column_name(reference: str) -> str:
    match = re.match(r"[A-Z]+", reference)
    if not match:
        raise MRE134AdapterError(f"Invalid XLSX cell reference {reference!r}")
    return match.group()


def load_mre134_demographics(
    root: str | Path = DEFAULT_MRE134_ROOT,
) -> tuple[MRE134Participant, ...]:
    """Read and validate the 134-row demographic workbook."""
    bundle_root = Path(root).expanduser().resolve()
    workbook = bundle_root / "MRE134_Demographics.xlsx"
    try:
        with zipfile.ZipFile(workbook) as archive:
            shared = _shared_string_values(archive)
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise MRE134AdapterError(
            f"Could not read demographics workbook {workbook}"
        ) from exc

    rows: list[dict[str, str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", XML_NS):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", XML_NS):
            column = _column_name(cell.attrib["r"])
            value_node = cell.find("m:v", XML_NS)
            if value_node is None:
                value = ""
            elif cell.attrib.get("t") == "s":
                value = shared[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[column] = value
        rows.append(values)

    participants = tuple(
        MRE134Participant(
            subject_id=row["A"],
            age_years=float(row["B"]),
            sex=row["C"].strip(),
            original_study=row["D"].strip(),
        )
        for row in rows[1:]
    )
    ids = [item.subject_id for item in participants]
    if len(participants) != 134 or len(ids) != len(set(ids)):
        raise MRE134AdapterError(
            f"Expected 134 unique demographic rows; found {len(participants)}"
        )
    return participants


def _bundle_paths(root: Path) -> dict[str, Path]:
    return {
        "stiffness_zip": root / "MRE134_Stiffness3D.nii.zip",
        "damping_zip": root / "MRE134_Damping3D.nii.zip",
        "stiffness_individual": root / "individual_stiffness_files",
        "damping_individual": root / "individual_damping ratio_files",
        "demographics": root / "MRE134_Demographics.xlsx",
        "paper": root / "Hiscox_et_al_2020_HBM.pdf",
        "readme": root / "README.md",
    }


def _validate_bundle(root: Path) -> dict[str, Path]:
    paths = _bundle_paths(root)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MRE134 bundle is incomplete: " + ", ".join(map(str, missing))
        )
    return paths


def _relative_location(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _pyvista_grid_from_affine(
    shape: tuple[int, ...], affine: np.ndarray
) -> pv.ImageData:
    if len(shape) != 3:
        raise MRE134AdapterError(f"Expected a 3D atlas; received shape {shape}")
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise MRE134AdapterError("NIfTI affine has invalid voxel-axis scales")
    direction = linear @ np.diag(1.0 / spacing)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise MRE134AdapterError(
            "NIfTI affine contains shear; PyVista ImageData requires orthogonal axes"
        )

    # NIfTI indices denote voxel centres. ImageData cells are bounded by
    # points, so shift the point-lattice origin back by half a voxel on every
    # index axis and add one point along each dimension.
    voxel_center_origin = np.asarray(affine[:3, 3], dtype=np.float64)
    point_origin = voxel_center_origin - 0.5 * np.sum(linear, axis=1)
    return pv.ImageData(
        dimensions=tuple(value + 1 for value in shape),
        spacing=tuple(float(value) for value in spacing),
        origin=tuple(float(value) for value in point_origin),
        direction_matrix=direction,
    )


def _contributor_count(root: Path, shape: tuple[int, ...]) -> np.ndarray:
    files = sorted((root / "individual_stiffness_files").glob("*Stiffness_warped.nii"))
    damping_ids = {
        path.name[:3]
        for path in (root / "individual_damping ratio_files").glob("*DR_warped.nii")
    }
    stiffness_ids = {path.name[:3] for path in files}
    if len(files) != 134 or stiffness_ids != damping_ids:
        raise MRE134AdapterError(
            "Expected 134 aligned individual stiffness and damping files"
        )
    count = np.zeros(math.prod(shape), dtype=np.uint16)
    for path in files:
        count += _read_nifti_presence(path, shape)
    return count


def _atlas_arrays(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _NiftiHeader]:
    stiffness, stiffness_info = _read_zipped_nifti(
        root / "MRE134_Stiffness3D.nii.zip", "MRE134_Stiffness3D.nii"
    )
    damping, damping_info = _read_zipped_nifti(
        root / "MRE134_Damping3D.nii.zip", "MRE134_Damping3D.nii"
    )
    if stiffness_info.shape != damping_info.shape or not np.allclose(
        stiffness_info.affine, damping_info.affine
    ):
        raise MRE134AdapterError("Stiffness and damping atlases do not share a grid")
    stiffness_kpa = np.asarray(stiffness, dtype=np.float32)
    stiffness_kpa *= STIFFNESS_RAW_TO_KPA
    damping_ratio = np.asarray(damping, dtype=np.float32)
    valid = (
        np.isfinite(stiffness_kpa)
        & np.isfinite(damping_ratio)
        & (stiffness_kpa > 0)
        & (damping_ratio != 0)
    )
    return stiffness_kpa, damping_ratio, valid, stiffness_info


def load_mre134(
    root: str | Path = DEFAULT_MRE134_ROOT,
    *,
    include_contributor_count: bool = False,
) -> LoadedMRE134:
    """Load group atlases into a cell-valued MNI ``ImageData``.

    Parameters
    ----------
    root:
        Directory containing the MRE134 release.
    include_contributor_count:
        Scan all 134 individual stiffness maps and add the number of positive
        contributors for every atlas voxel. This reads roughly 462 MB but is
        important for coverage-aware voxelwise work.
    """
    bundle_root = Path(root).expanduser().resolve()
    _validate_bundle(bundle_root)
    stiffness, damping, valid, info = _atlas_arrays(bundle_root)
    grid = _pyvista_grid_from_affine(info.shape, info.affine)
    if grid.n_cells != stiffness.size:
        raise MRE134AdapterError(
            f"Adapted grid has {grid.n_cells} cells for {stiffness.size} voxels"
        )
    grid.cell_data[STIFFNESS_ARRAY] = stiffness
    grid.cell_data[DAMPING_ARRAY] = damping
    grid.cell_data[VALID_MASK_ARRAY] = valid.astype(np.uint8)
    if include_contributor_count:
        grid.cell_data[CONTRIBUTOR_COUNT_ARRAY] = _contributor_count(
            bundle_root, info.shape
        )
    grid.field_data["MRE134_ActuationFrequency_Hz"] = np.array(
        [ACTUATION_FREQUENCY_HZ], dtype=np.float64
    )
    grid.field_data["MRE134_SubjectCount"] = np.array([134], dtype=np.int32)
    grid.field_data["MRE134_NIfTI_VoxelToMNI"] = info.affine.reshape(-1)
    grid.set_active_scalars(STIFFNESS_ARRAY, preference="cell")

    participants = load_mre134_demographics(bundle_root)
    case = build_mre134_observation_case(bundle_root)
    if case.validate_references():
        raise MRE134AdapterError(
            "Generated ObservationCase contains unresolved references: "
            + "; ".join(case.validate_references())
        )
    return LoadedMRE134(
        dataset=grid,
        case=case,
        participants=participants,
        source_root=bundle_root,
    )


def build_mre134_observation_case(
    root: str | Path = DEFAULT_MRE134_ROOT,
) -> ObservationCase:
    """Construct complete typed research metadata for the MRE134 release."""
    bundle_root = Path(root).expanduser().resolve()
    paths = _validate_bundle(bundle_root)
    stiffness, damping, valid, info = _atlas_arrays(bundle_root)
    participants = load_mre134_demographics(bundle_root)
    ages = np.array([item.age_years for item in participants], dtype=np.float64)
    sex_counts = dict(Counter(item.sex for item in participants))
    study_counts = dict(Counter(item.original_study for item in participants))
    support_voxels = int(np.count_nonzero(valid))
    stiffness_values = stiffness[valid]
    damping_values = damping[valid]
    adapted_vti = OUTPUT_ROOT / "mre134" / "adapter" / "MRE134_atlas.vti"

    transform = tuple(tuple(float(value) for value in row) for row in info.affine)
    shape = tuple(int(value) for value in info.shape)
    raw_stiffness_location = _relative_location(paths["stiffness_zip"])
    raw_damping_location = _relative_location(paths["damping_zip"])

    assets = (
        DataAsset(
            asset_id="mre134-individual-stiffness",
            role="individual normalized shear-stiffness maps",
            location=_relative_location(paths["stiffness_individual"]),
            format="directory-of-nifti-1",
            field_names=("raw_shear_stiffness",),
            association=ValueAssociation.VOXEL,
            shape=(134, *shape),
            dtype="float32",
            units="Pa-equivalent stored scale",
            coordinate_system="MNI152-2mm",
            description=(
                "134 warped individual maps; values require division by 1000 for kPa."
            ),
            extensions={"quantity_kind": "shear-stiffness", "file_count": 134},
        ),
        DataAsset(
            asset_id="mre134-individual-damping",
            role="individual normalized damping-ratio maps",
            location=_relative_location(paths["damping_individual"]),
            format="directory-of-nifti-1",
            field_names=("damping_ratio",),
            association=ValueAssociation.VOXEL,
            shape=(134, *shape),
            dtype="float32",
            units="1",
            coordinate_system="MNI152-2mm",
            description="134 warped individual dimensionless damping-ratio maps.",
            extensions={"quantity_kind": "damping-ratio", "file_count": 134},
        ),
        DataAsset(
            asset_id="mre134-stiffness-archive",
            role="published group-average shear-stiffness NIfTI archive",
            location=raw_stiffness_location,
            format="zip/nifti-1",
            media_type="application/zip",
            byte_size=paths["stiffness_zip"].stat().st_size,
            field_names=("raw_group_mean_stiffness",),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="float32",
            units="Pa-equivalent stored scale",
            coordinate_system="MNI152-2mm",
            generated_by="create-mre134-group-atlas",
            extensions={"quantity_kind": "shear-stiffness", "raw_to_kpa": 0.001},
        ),
        DataAsset(
            asset_id="mre134-damping-archive",
            role="published group-average damping-ratio NIfTI archive",
            location=raw_damping_location,
            format="zip/nifti-1",
            media_type="application/zip",
            byte_size=paths["damping_zip"].stat().st_size,
            field_names=("group_mean_damping_ratio",),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="float32",
            units="1",
            coordinate_system="MNI152-2mm",
            generated_by="create-mre134-group-atlas",
            extensions={"quantity_kind": "damping-ratio"},
        ),
        DataAsset(
            asset_id="mre134-adapted-vti",
            role="adapter-created cell-valued MNI atlas grid",
            location=_relative_location(adapted_vti),
            format="vti",
            media_type="model/vnd.vtk",
            field_names=(
                STIFFNESS_ARRAY,
                DAMPING_ARRAY,
                VALID_MASK_ARRAY,
                CONTRIBUTOR_COUNT_ARRAY,
            ),
            association=ValueAssociation.VOXEL,
            shape=shape,
            coordinate_system="MNI152-2mm",
            generated_by="load-mre134-atlas",
            description=(
                "Compressed VTI written by the adapter; NIfTI voxel values are "
                "stored as PyVista cell arrays. Contributor count is present when "
                "the CLI is run with --contributors."
            ),
            extensions={"pyvista_dataset_type": "ImageData"},
        ),
        DataAsset(
            asset_id="mre134-stiffness-kpa-runtime",
            role="adapter-converted group-average shear stiffness",
            location=f"generated:brain_strain.adapters.mre134/{STIFFNESS_ARRAY}",
            format="pyvista-cell-array",
            field_names=(STIFFNESS_ARRAY,),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="float32",
            units="kPa",
            coordinate_system="MNI152-2mm",
            generated_by="load-mre134-atlas",
            description=(
                "Cell values exposed by load_mre134 after raw / 1000 conversion."
            ),
            extensions={
                "quantity_kind": "shear-stiffness",
                "comparison_role": "field",
                "pyvista_association": "cell",
            },
        ),
        DataAsset(
            asset_id="mre134-damping-runtime",
            role="adapter-loaded group-average damping ratio",
            location=f"generated:brain_strain.adapters.mre134/{DAMPING_ARRAY}",
            format="pyvista-cell-array",
            field_names=(DAMPING_ARRAY,),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="float32",
            units="1",
            coordinate_system="MNI152-2mm",
            generated_by="load-mre134-atlas",
            extensions={
                "quantity_kind": "damping-ratio",
                "comparison_role": "field",
                "pyvista_association": "cell",
            },
        ),
        DataAsset(
            asset_id="mre134-valid-mask-runtime",
            role="joint nonzero finite atlas mask",
            location=f"generated:brain_strain.adapters.mre134/{VALID_MASK_ARRAY}",
            format="pyvista-cell-array",
            field_names=(VALID_MASK_ARRAY,),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="uint8",
            units="1",
            coordinate_system="MNI152-2mm",
            generated_by="load-mre134-atlas",
        ),
        DataAsset(
            asset_id="mre134-contributor-count-runtime",
            role="positive individual-map contributor count per atlas voxel",
            location=f"generated:brain_strain.adapters.mre134/{CONTRIBUTOR_COUNT_ARRAY}",
            format="pyvista-cell-array",
            field_names=(CONTRIBUTOR_COUNT_ARRAY,),
            association=ValueAssociation.VOXEL,
            shape=shape,
            dtype="uint16",
            units="participants",
            coordinate_system="MNI152-2mm",
            generated_by="count-mre134-contributors",
            description="Generated when load_mre134(include_contributor_count=True).",
        ),
        DataAsset(
            asset_id="mre134-demographics",
            role="deidentified cohort demographics",
            location=_relative_location(paths["demographics"]),
            format="xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            byte_size=paths["demographics"].stat().st_size,
            field_names=("SUBJECT ID", "AGE", "SEX", "ORIG STUDY"),
            association=ValueAssociation.SPECIMEN,
            shape=(134, 4),
        ),
        DataAsset(
            asset_id="mre134-recode-label-mapping",
            role="MRE134 publication ROI to ReCoDE part-label crosswalk",
            location=_relative_location(DEFAULT_MAPPING_OUTPUT),
            format="json",
            media_type="application/json",
            field_names=("mre134_label", "unified_label", "recode_parts"),
            association=ValueAssociation.REGION,
            shape=(len(MRE134_RECODE_LABEL_MAPPINGS),),
            generated_by="map-mre134-recode-labels",
            description=(
                "Semantic mapping with explicit exact, aggregate, partitioned, "
                "broader-target, and unmatched classifications."
            ),
        ),
        DataAsset(
            asset_id="mre134-publication",
            role="study publication",
            location=_relative_location(paths["paper"]),
            format="pdf",
            media_type="application/pdf",
            byte_size=paths["paper"].stat().st_size,
        ),
    )

    case = ObservationCase(
        case_id="mre134-standard-space-atlas",
        title="MRE134 standard-space atlas of human brain viscoelastic properties",
        source=CaseSource.DERIVED,
        study_id="Hiscox-et-al-2020-HBM-MRE134",
        description=(
            "Cohort-average 50-Hz shear stiffness and damping ratio from 134 healthy "
            "young adults, nonlinearly normalized to the MNI152 2-mm template."
        ),
        research_question=(
            "What are the standard-space distributions of shear stiffness and damping "
            "ratio in the healthy young adult brain?"
        ),
        units=UnitSystem(
            name="MNI millimetres, seconds, and kPa adapter representation",
            length="mm",
            time="s",
            pressure="kPa",
            strain="1",
            extensions={"damping_ratio": "1"},
        ),
        coordinate_systems=(
            CoordinateSystem(
                name="MNI152-2mm",
                convention=(
                    "NIfTI voxel centres mapped to MNI/ICBM-152 world coordinates"
                ),
                origin=tuple(float(value) for value in info.affine[:3, 3]),
                axes={
                    "voxel_i": "world x decreases by 2 mm",
                    "voxel_j": "world y increases by 2 mm",
                    "voxel_k": "world z increases by 2 mm",
                },
                reference="MNI152 nonlinear T1-weighted 2-mm atlas",
                transform_to_reference=transform,
                handedness="right-handed MNI world; voxel i axis reversed",
                moving=False,
            ),
        ),
        anatomy=AnatomyModel(
            anatomy="brain",
            imaging_modality="magnetic resonance elastography and T1-weighted MRI",
            imaging_protocol={
                "actuation_frequency_hz": ACTUATION_FREQUENCY_HZ,
                "MRE_resolution_mm": {"Study D": 2.0, "other_studies": 1.6},
                "output_resolution_mm": 2.0,
            },
            image_asset_ids=("mre134-stiffness-archive", "mre134-damping-archive"),
            segmentation_method="MNI152 whole-brain and study ROI masks",
            segmentation_labels=observation_case_label_metadata(),
            mesh_asset_id="mre134-adapted-vti",
            mesh_stage="voxel atlas",
            mesh_type="regular image grid represented as PyVista ImageData cells",
            element_types=("voxel",),
            node_count=math.prod(value + 1 for value in shape),
            element_count=math.prod(shape),
            coordinate_system="MNI152-2mm",
            registration={
                "software": "ANTs",
                "method": "rigid/affine MRE-to-T1 plus nonlinear SyN T1-to-MNI152",
                "interpolation": "linear",
            },
        ),
        loading=LoadingCondition(
            event_type="harmonic vibration",
            description="Pneumatically transmitted head vibration for brain MRE",
            loading_mode="steady-state harmonic actuation",
            extensions={"actuation_frequency_hz": ACTUATION_FREQUENCY_HZ},
        ),
        acquisition=Acquisition(
            method="3D multislab multishot spiral magnetic resonance elastography",
            protocol=(
                "three motion-encoding axes, opposite polarities, four phase offsets"
            ),
            instruments=(
                Instrument(
                    instrument_id="resoundant-actuator",
                    kind="pneumatic MRE actuator",
                    manufacturer="Resoundant",
                    extensions={"actuation_frequency_hz": ACTUATION_FREQUENCY_HZ},
                ),
            ),
            extensions={
                "sites": [
                    "University of Edinburgh",
                    "UIUC",
                    "Carle Foundation Hospital",
                ],
                "scanner_models": ["Siemens Verio", "Siemens Trio"],
                "available_processed_property_asset_ids": [
                    "mre134-individual-stiffness",
                    "mre134-individual-damping",
                ],
            },
        ),
        derived_results=(
            DerivedResult(
                result_id="mre134-global-mean-stiffness",
                name="Mean nonzero atlas shear stiffness",
                value=float(np.mean(stiffness_values)),
                units="kPa",
                method="mean over joint finite nonzero atlas support",
                asset_id="mre134-stiffness-kpa-runtime",
                coordinate_system="MNI152-2mm",
                extensions={"support_voxels": support_voxels},
            ),
            DerivedResult(
                result_id="mre134-global-mean-damping",
                name="Mean nonzero atlas damping ratio",
                value=float(np.mean(damping_values)),
                units="1",
                method="mean over joint finite nonzero atlas support",
                asset_id="mre134-damping-runtime",
                coordinate_system="MNI152-2mm",
                extensions={"support_voxels": support_voxels},
            ),
        ),
        assets=assets,
        processing=(
            ProcessingStep(
                step_id="create-mre134-group-atlas",
                name="Normalize individual maps and form group atlases",
                software="ANTs and study MRE processing pipeline",
                parameters={
                    "group_operation": (
                        "voxelwise mean of positive/nonzero contributors"
                    ),
                    "target": "MNI152 2-mm",
                    "participants": 134,
                },
                input_asset_ids=(
                    "mre134-individual-stiffness",
                    "mre134-individual-damping",
                ),
                output_asset_ids=("mre134-stiffness-archive", "mre134-damping-archive"),
            ),
            ProcessingStep(
                step_id="load-mre134-atlas",
                name="Adapt NIfTI atlases to a cell-valued PyVista MNI grid",
                software="brain_strain.adapters.mre134",
                code_reference="brain_strain.adapters.mre134.load_mre134",
                parameters={
                    "stiffness_scale": STIFFNESS_RAW_TO_KPA,
                    "voxel_representation": "PyVista ImageData cells",
                    "cell_centres_match_nifti_voxel_centres": True,
                },
                input_asset_ids=("mre134-stiffness-archive", "mre134-damping-archive"),
                output_asset_ids=(
                    "mre134-adapted-vti",
                    "mre134-stiffness-kpa-runtime",
                    "mre134-damping-runtime",
                    "mre134-valid-mask-runtime",
                ),
            ),
            ProcessingStep(
                step_id="count-mre134-contributors",
                name="Count positive individual stiffness contributors per voxel",
                software="brain_strain.adapters.mre134",
                code_reference="brain_strain.adapters.mre134._contributor_count",
                input_asset_ids=("mre134-individual-stiffness",),
                output_asset_ids=("mre134-contributor-count-runtime",),
            ),
            ProcessingStep(
                step_id="map-mre134-recode-labels",
                name="Map MRE134 publication ROI names to ReCoDE part labels",
                software="brain_strain.adapters.mre134_labels",
                code_reference=(
                    "brain_strain.adapters.mre134_labels.mapping_document"
                ),
                parameters={
                    "target_vocabulary": "ReCoDE active part_list_full.k",
                    "preserve_mapping_granularity": True,
                    "spatial_registration_implied": False,
                },
                input_asset_ids=("mre134-publication",),
                output_asset_ids=("mre134-recode-label-mapping",),
            ),
        ),
        quality=QualityAssessment(
            status="published cohort atlas with adapter validation",
            signal_checks={
                "joint_finite_nonzero_support_voxels": support_voxels,
                "converted_stiffness_mean_kpa": float(np.mean(stiffness_values)),
                "damping_mean": float(np.mean(damping_values)),
            },
            validation={
                "reference_global_stiffness_kpa": 2.62,
                "reference_global_damping_ratio": 0.208,
                "case_references_resolve": True,
            },
            missing_data={
                "background": "NaN or zero outside the released atlas support",
                "voxel_contributor_count": (
                    "varies from 1 to 134; generate the runtime count "
                    "before voxelwise use"
                ),
            },
            limitations=(
                "MRE properties are dynamic 50-Hz quantities, not strain "
                "response fields.",
                "Original studies differ in site, scanner, coil, resolution, "
                "age, and sex composition.",
                "Stiffness source values require raw / 1000 conversion to kPa.",
                "Atlas boundary voxels may have substantially fewer than "
                "134 contributors.",
            ),
        ),
        governance=Governance(
            privacy_classification="deidentified cohort data",
            deidentification="public subject IDs 001-134; age, sex, and study only",
            access_conditions="repository access conditions",
            licence="See source repository and publication",
        ),
        publications=(
            "Hiscox et al. (2020), Standard-Space Atlas of the Viscoelastic "
            "Properties of the Human Brain, Human Brain Mapping "
            "41(18):5282-5300, DOI 10.1002/hbm.25192",
        ),
        citations=(
            "data/external/MRE134-master/Hiscox_et_al_2020_HBM.pdf",
            "data/external/MRE134-master/README.md",
            "data/external/ReCoDE-brain-mesh-creation-main/"
            "src/dependencies/simulation/part_list_full.k",
        ),
        keywords=(
            "brain",
            "magnetic resonance elastography",
            "shear stiffness",
            "damping ratio",
            "MNI152",
            "material properties",
        ),
        related_case_ids=("repo-generalized-maxwell-impact-demo",),
        extensions={
            "cohort": {
                "participants": len(participants),
                "sex_counts": sex_counts,
                "age_years": {
                    "mean": float(np.mean(ages)),
                    "sd": float(np.std(ages, ddof=1)),
                    "range": [float(np.min(ages)), float(np.max(ages))],
                },
                "original_study_counts": study_counts,
            },
            "adapter": {
                "module": "brain_strain.adapters.mre134",
                "primary_field": STIFFNESS_ARRAY,
                "alternate_field": DAMPING_ARRAY,
                "label_mapping": "examples/cases/mre134_recode_label_mapping.json",
            },
        },
    )
    issues = case.validate_references()
    if issues:
        raise MRE134AdapterError(
            "Generated ObservationCase contains unresolved references: "
            + "; ".join(issues)
        )
    return case


def compare_mre134_with_case(
    mre_case: ObservationCase,
    other_case: ObservationCase,
) -> MRE134CaseComparison:
    """Compare case semantics and state what is safe to compare.

    The result intentionally refuses a raw comparison when quantities, units,
    coordinate systems, associations, or temporal semantics differ.
    """
    decision = decide_observation_case_comparability(mre_case, other_case)

    # Preserve the legacy report's deliberately broad quantity labels while
    # taking every compatibility decision from the generic comparison engine.
    legacy_quantities: dict[tuple[str, str], str] = {}
    for observation in other_case.strain_observations:
        quantity = observation.extensions.get("quantity_kind")
        if not quantity:
            quantity = (
                "strain-like"
                if "strain" in observation.name.casefold()
                or observation.measure.value in {"other", "unknown"}
                else f"{observation.measure.value}-strain"
            )
        legacy_quantities[(observation.asset_id, observation.name)] = str(quantity)

    pairs = [
        FieldCompatibility(
            mre_field=pair.current.field_name,
            other_field=pair.simulation.field_name,
            mre_quantity=pair.current.quantity,
            other_quantity=legacy_quantities.get(
                (pair.simulation.asset_id, pair.simulation.field_name),
                pair.simulation.quantity,
            ),
            mre_units=pair.current.units,
            other_units=pair.simulation.units,
            same_quantity=pair.same_quantity,
            same_units=pair.same_units,
            same_coordinate_system=pair.same_coordinate_system,
            same_association=pair.same_association,
            same_temporal_semantics=pair.same_temporal_semantics,
            raw_value_comparison_valid=pair.raw_value_semantically_compatible,
        )
        for pair in decision.field_pairs
    ]

    raw_valid = any(pair.raw_value_comparison_valid for pair in pairs)
    spatial_ready = any(pair.same_coordinate_system for pair in pairs)
    issues: list[str] = []
    if not any(pair.same_quantity for pair in pairs):
        issues.append(
            "MRE fields are material properties; comparison fields are "
            "strain/response values."
        )
    if not spatial_ready:
        issues.append("Cases do not share a declared coordinate system or transform.")
    if mre_case.units.length and not other_case.units.length:
        issues.append(
            "MRE uses millimetres, while the comparison mesh length unit is unstated."
        )
    if not any(pair.same_temporal_semantics for pair in pairs):
        issues.append(
            "MRE properties are static; the comparison field is time varying."
        )
    if not any(pair.same_association for pair in pairs):
        issues.append("MRE values are voxels; the comparison values are mesh cells.")
    if other_case.source is CaseSource.SYNTHETIC:
        issues.append(
            "The comparison case is explicitly synthetic and has no "
            "physical validation."
        )
    elif (
        other_case.source is CaseSource.SIMULATION
        and not other_case.quality.validation.get("physical_validation", False)
    ):
        issues.append("The comparison simulation has no physical validation.")

    return MRE134CaseComparison(
        mre_case_id=mre_case.case_id,
        other_case_id=other_case.case_id,
        metadata_comparison_valid=True,
        normalized_distribution_comparison_valid=True,
        spatial_comparison_ready=spatial_ready,
        raw_value_comparison_valid=raw_valid,
        field_pairs=tuple(pairs),
        issues=tuple(issues),
        allowed_comparisons=(
            "metadata and provenance",
            "sample counts, missingness, and coverage",
            "unit-free normalized distributions",
            "spatial concentration without anatomical correspondence",
        ),
        required_for_physical_comparison=(
            "declare the comparison mesh length unit",
            "provide an MNI-to-mesh transform and resampling policy",
            "replace the reduced-order material-point response with a "
            "calibrated FE result",
            "compare like quantities such as displacement-to-displacement",
            "or declare a constitutive model that uses 50-Hz MRE properties as inputs",
        ),
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MRE134_ROOT)
    parser.add_argument(
        "--case-output",
        type=Path,
        default=EXAMPLE_CASES_ROOT / "mre134_observation_case.json",
    )
    parser.add_argument(
        "--simulation-case",
        type=Path,
        default=DEFAULT_SIMULATION_CASE,
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=EXAMPLE_CASES_ROOT / "mre134_vs_simulation_comparison.json",
    )
    parser.add_argument("--vti-output", type=Path)
    parser.add_argument(
        "--contributors",
        action="store_true",
        help="scan individual maps and include a per-voxel contributor count",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    label_mapping_output = write_mapping()
    loaded = load_mre134(
        args.root,
        include_contributor_count=args.contributors,
    )
    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    loaded.case.write_json(args.case_output, omit_none=True)
    simulation_case = ObservationCase.read_json(args.simulation_case)
    comparison = compare_mre134_with_case(loaded.case, simulation_case)
    comparison.write_json(args.comparison_output)
    if args.vti_output:
        loaded.write_vti(args.vti_output)
    print(f"Loaded MRE134: {loaded.dataset.n_cells:,} voxels")
    print(f"Observation case: {args.case_output.resolve()}")
    print(f"Compatibility report: {args.comparison_output.resolve()}")
    print(f"Label mapping: {label_mapping_output}")
    if args.vti_output:
        print(f"Adapted VTI: {args.vti_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTUATION_FREQUENCY_HZ",
    "CONTRIBUTOR_COUNT_ARRAY",
    "DAMPING_ARRAY",
    "DEFAULT_MRE134_ROOT",
    "FieldCompatibility",
    "LoadedMRE134",
    "MRE134AdapterError",
    "MRE134CaseComparison",
    "MRE134Participant",
    "STIFFNESS_ARRAY",
    "VALID_MASK_ARRAY",
    "build_mre134_observation_case",
    "compare_mre134_with_case",
    "load_mre134",
    "load_mre134_demographics",
]
