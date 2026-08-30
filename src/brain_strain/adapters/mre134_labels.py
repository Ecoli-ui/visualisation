"""Semantic label crosswalk between the MRE134 study and ReCoDE mesh parts.

MRE134 publishes continuous stiffness and damping maps, not an anatomical
label volume.  Its label vocabulary therefore comes from the regions of
interest reported by Hiscox et al. (2020).  ReCoDE labels are LS-DYNA part
names and IDs from ``part_list_full.k``.

The two vocabularies have different granularity.  In particular, MRE134 uses
bilateral subcortical masks, named white-matter tracts, and cortical parcels,
whereas the ReCoDE mesh separates left/right subcortical parts but generally
uses broad cerebral-white-matter and cerebral-cortex parts.  The mapping
records that loss of specificity explicitly through ``match_kind``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..paths import EXAMPLE_CASES_ROOT

DEFAULT_MAPPING_OUTPUT = EXAMPLE_CASES_ROOT / "mre134_recode_label_mapping.json"

MatchKind = Literal[
    "exact-bilateral",
    "aggregate",
    "partitioned",
    "broader-recode",
    "no-direct-match",
]


@dataclass(frozen=True, slots=True)
class ReCoDEPart:
    """One active ReCoDE LS-DYNA part used by the crosswalk."""

    part_id: int
    label: str


@dataclass(frozen=True, slots=True)
class LabelMapping:
    """One MRE134 ROI label and its ReCoDE comparison target."""

    mre134_label: str
    abbreviation: str
    category: str
    unified_label: str
    recode_parts: tuple[ReCoDEPart, ...]
    match_kind: MatchKind
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mre134_label": self.mre134_label,
            "abbreviation": self.abbreviation,
            "category": self.category,
            "unified_label": self.unified_label,
            "recode_parts": [
                {"part_id": part.part_id, "label": part.label}
                for part in self.recode_parts
            ],
            "match_kind": self.match_kind,
            "note": self.note,
        }


def _parts(*values: tuple[int, str]) -> tuple[ReCoDEPart, ...]:
    return tuple(ReCoDEPart(part_id, label) for part_id, label in values)


_CEREBRAL_WHITE_MATTER = _parts(
    (2, "Left-Cerebral-White-Matter"),
    (41, "Right-Cerebral-White-Matter"),
)
_ALL_WHITE_MATTER = _CEREBRAL_WHITE_MATTER + _parts(
    (7, "Left-Cerebellum-White-Matter"),
    (46, "Right-Cerebellum-White-Matter"),
    (77, "WM-hypointensities"),
    (85, "Optic-Chiasm"),
    (251, "CC_Posterior"),
    (252, "CC_Mid_Posterior"),
    (253, "CC_Central"),
    (254, "CC_Mid_Anterior"),
    (255, "CC_Anterior"),
)
_CEREBRAL_CORTEX = _parts(
    (3, "Left-Cerebral-Cortex"),
    (42, "Right-Cerebral-Cortex"),
)
_AMYGDALA = _parts((18, "Left-Amygdala"), (54, "Right-Amygdala"))
_CAUDATE = _parts((11, "Left-Caudate"), (50, "Right-Caudate"))
_HIPPOCAMPUS = _parts((17, "Left-Hippocampus"), (53, "Right-Hippocampus"))
_PALLIDUM = _parts((13, "Left-Pallidum"), (52, "Right-Pallidum"))
_PUTAMEN = _parts((12, "Left-Putamen"), (51, "Right-Putamen"))
_THALAMUS = _parts(
    (10, "Left-Thalamus-Proper*"),
    (49, "Right-Thalamus-Proper*"),
)
_SUBCORTICAL_GREY_MATTER = (
    _AMYGDALA + _CAUDATE + _HIPPOCAMPUS + _PALLIDUM + _PUTAMEN + _THALAMUS
)
_CORPUS_CALLOSUM = _parts(
    (251, "CC_Posterior"),
    (252, "CC_Mid_Posterior"),
    (253, "CC_Central"),
    (254, "CC_Mid_Anterior"),
    (255, "CC_Anterior"),
)


def _mapping(
    mre134_label: str,
    abbreviation: str,
    category: str,
    unified_label: str,
    recode_parts: tuple[ReCoDEPart, ...],
    match_kind: MatchKind,
    note: str,
) -> LabelMapping:
    return LabelMapping(
        mre134_label=mre134_label,
        abbreviation=abbreviation,
        category=category,
        unified_label=unified_label,
        recode_parts=recode_parts,
        match_kind=match_kind,
        note=note,
    )


_BILATERAL_NOTE = (
    "MRE134 reports one bilateral ROI; combine the left and right ReCoDE parts."
)
_TRACT_NOTE = (
    "ReCoDE has no tract-specific part; the listed cerebral-white-matter parts "
    "are a broader fallback and require an atlas mask for spatial comparison."
)
_CORTEX_NOTE = (
    "ReCoDE has no parcel-specific cortical part; the listed cerebral-cortex "
    "parts are a broader fallback and require an atlas mask for spatial comparison."
)


MRE134_RECODE_LABEL_MAPPINGS: tuple[LabelMapping, ...] = (
    # Global masks reported by MRE134.
    _mapping(
        "Whole brain",
        "Global",
        "global",
        "Whole brain (ventricles excluded)",
        (),
        "no-direct-match",
        "MRE134 uses a whole-brain mask excluding ventricles; ReCoDE has no "
        "single equivalent part.",
    ),
    _mapping(
        "White matter",
        "WM",
        "global",
        "White matter",
        _ALL_WHITE_MATTER,
        "aggregate",
        "Combine ReCoDE parts assigned to its broad white-matter tissue group.",
    ),
    _mapping(
        "Subcortical gray matter",
        "SGM",
        "global",
        "Subcortical grey matter",
        _SUBCORTICAL_GREY_MATTER,
        "aggregate",
        "Both sources define this aggregate from the six mapped "
        "subcortical structures.",
    ),
    _mapping(
        "Cortical gray matter",
        "CGM",
        "global",
        "Cerebral cortex",
        _CEREBRAL_CORTEX,
        "broader-recode",
        "MRE134 combines selected cortical ROIs; ReCoDE's bilateral "
        "cerebral-cortex parts are broader.",
    ),
    # Subcortical gray-matter ROIs.
    _mapping(
        "Amygdala",
        "AM",
        "subcortical-gray-matter",
        "Amygdala",
        _AMYGDALA,
        "exact-bilateral",
        _BILATERAL_NOTE,
    ),
    _mapping(
        "Caudate",
        "CA",
        "subcortical-gray-matter",
        "Caudate",
        _CAUDATE,
        "exact-bilateral",
        _BILATERAL_NOTE,
    ),
    _mapping(
        "Hippocampus",
        "HC",
        "subcortical-gray-matter",
        "Hippocampus",
        _HIPPOCAMPUS,
        "exact-bilateral",
        _BILATERAL_NOTE,
    ),
    _mapping(
        "Pallidum",
        "PA",
        "subcortical-gray-matter",
        "Pallidum",
        _PALLIDUM,
        "exact-bilateral",
        _BILATERAL_NOTE,
    ),
    _mapping(
        "Putamen",
        "PU",
        "subcortical-gray-matter",
        "Putamen",
        _PUTAMEN,
        "exact-bilateral",
        _BILATERAL_NOTE,
    ),
    _mapping(
        "Thalamus",
        "TH",
        "subcortical-gray-matter",
        "Thalamus",
        _THALAMUS,
        "exact-bilateral",
        "MRE134 reports one bilateral ROI; combine ReCoDE's left/right "
        "Thalamus-Proper parts.",
    ),
    # White-matter tract ROIs.
    _mapping(
        "Corticospinal tract",
        "CST",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Anterior thalamic radiation",
        "ATR",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Posterior thalamic radiation",
        "PTR",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Corona radiata",
        "CRa",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Corpus callosum",
        "CC",
        "white-matter-tract",
        "Corpus callosum",
        _CORPUS_CALLOSUM,
        "partitioned",
        "Combine the five anterior-to-posterior ReCoDE corpus-callosum parts.",
    ),
    _mapping(
        "Major forceps",
        "FMa",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Minor forceps",
        "FMi",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Fornix",
        "FX",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        "The ReCoDE lookup table names a Fornix label, but the active part "
        "list does not contain it; use a registered tract mask.",
    ),
    _mapping(
        "Uncinate fasciculus",
        "UN",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Inferior frontal-occipital fasciculus",
        "IFOF",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Inferior longitudinal fasciculus",
        "ILF",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    _mapping(
        "Superior longitudinal fasciculus",
        "SLF",
        "white-matter-tract",
        "Cerebral white matter",
        _CEREBRAL_WHITE_MATTER,
        "broader-recode",
        _TRACT_NOTE,
    ),
    # Cortical gray-matter ROIs.
    *(
        _mapping(
            label,
            abbreviation,
            "cortical-gray-matter",
            "Cerebral cortex",
            _CEREBRAL_CORTEX,
            "broader-recode",
            _CORTEX_NOTE,
        )
        for label, abbreviation in (
            ("Superior frontal cortex", "SFC"),
            ("Rostral middle frontal cortex", "RMF"),
            ("Precentral cortex", "PRE"),
            ("Lateral occipital cortex", "LaO"),
            ("Lingual occipital cortex", "LiO"),
            ("Cuneus", "CN"),
            ("Superior parietal cortex", "SPC"),
            ("Postcentral cortex", "POST"),
            ("Precuneus", "PCN"),
            ("Superior temporal cortex", "STC"),
            ("Inferior temporal cortex", "ITC"),
            ("Fusiform gyrus", "FSM"),
        )
    ),
)


def _normalise_label(value: str) -> str:
    """Make label lookup insensitive to case, separators, and punctuation."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _build_mre134_index() -> dict[str, LabelMapping]:
    index: dict[str, LabelMapping] = {}
    for mapping in MRE134_RECODE_LABEL_MAPPINGS:
        aliases = {
            mapping.mre134_label,
            mapping.abbreviation,
            mapping.mre134_label.replace("gray", "grey"),
        }
        for alias in aliases:
            key = _normalise_label(alias)
            previous = index.get(key)
            if previous is not None and previous != mapping:
                raise RuntimeError(f"Ambiguous MRE134 label alias {alias!r}")
            index[key] = mapping
    extra_aliases = {
        "Forceps major": "Major forceps",
        "Forceps minor": "Minor forceps",
        "Inferior fronto-occipital fasciculus": "Inferior frontal-occipital fasciculus",
        "Fusiform": "Fusiform gyrus",
        "FSG": "Fusiform gyrus",
    }
    for alias, label in extra_aliases.items():
        index[_normalise_label(alias)] = index[_normalise_label(label)]
    return index


_MRE134_INDEX = _build_mre134_index()


def _build_recode_index() -> dict[int, tuple[LabelMapping, ...]]:
    index: dict[int, list[LabelMapping]] = {}
    for mapping in MRE134_RECODE_LABEL_MAPPINGS:
        for part in mapping.recode_parts:
            index.setdefault(part.part_id, []).append(mapping)
    return {part_id: tuple(mappings) for part_id, mappings in index.items()}


_RECODE_INDEX = _build_recode_index()

# One comparison label per ReCoDE part for cell-data annotation.  More
# specific shared regions override their containing global tissue group.
_RECODE_UNIFIED_LABELS = {part.part_id: "White matter" for part in _ALL_WHITE_MATTER}
_RECODE_UNIFIED_LABELS.update(
    {part.part_id: "Cerebral white matter" for part in _CEREBRAL_WHITE_MATTER}
)
_RECODE_UNIFIED_LABELS.update(
    {part.part_id: "Cerebral cortex" for part in _CEREBRAL_CORTEX}
)
for _label, _recode_parts in (
    ("Amygdala", _AMYGDALA),
    ("Caudate", _CAUDATE),
    ("Hippocampus", _HIPPOCAMPUS),
    ("Pallidum", _PALLIDUM),
    ("Putamen", _PUTAMEN),
    ("Thalamus", _THALAMUS),
    ("Corpus callosum", _CORPUS_CALLOSUM),
):
    _RECODE_UNIFIED_LABELS.update({part.part_id: _label for part in _recode_parts})


def map_mre134_label(label: str) -> LabelMapping:
    """Resolve an MRE134 full name or abbreviation to its ReCoDE mapping."""
    try:
        return _MRE134_INDEX[_normalise_label(label)]
    except KeyError as exc:
        raise KeyError(f"Unknown MRE134 ROI label {label!r}") from exc


def mappings_for_recode_part(part_id: int) -> tuple[LabelMapping, ...]:
    """Return every MRE134 ROI mapping that references a ReCoDE part ID."""
    return _RECODE_INDEX.get(int(part_id), ())


def unified_label_for_recode_part(part_id: int) -> str | None:
    """Return the most specific common comparison label for a ReCoDE part."""
    return _RECODE_UNIFIED_LABELS.get(int(part_id))


def mapping_document() -> dict[str, object]:
    """Return the complete, JSON-serializable crosswalk and comparison notes."""
    counts: dict[str, int] = {}
    for mapping in MRE134_RECODE_LABEL_MAPPINGS:
        counts[mapping.match_kind] = counts.get(mapping.match_kind, 0) + 1
    return {
        "schema_version": "1.0",
        "source": {
            "name": "MRE134 publication ROI vocabulary",
            "contains_label_volume": False,
            "note": (
                "The released MRE134 NIfTI files are continuous property maps. "
                "These names describe the publication's ROI masks, which are not "
                "included in MRE134-master."
            ),
        },
        "target": {
            "name": "ReCoDE active LS-DYNA part vocabulary",
            "definition": (
                "data/external/ReCoDE-brain-mesh-creation-main/"
                "src/dependencies/simulation/"
                "part_list_full.k"
            ),
        },
        "semantics": {
            "exact-bilateral": "same structure; union the left/right ReCoDE parts",
            "aggregate": (
                "combine several ReCoDE parts into one broad comparison region"
            ),
            "partitioned": "same structure split into several ReCoDE parts",
            "broader-recode": "ReCoDE target is less specific than the MRE134 ROI",
            "no-direct-match": "no defensible part-only mapping",
        },
        "summary": {
            "mre134_labels": len(MRE134_RECODE_LABEL_MAPPINGS),
            "match_kind_counts": counts,
        },
        "mappings": [mapping.to_dict() for mapping in MRE134_RECODE_LABEL_MAPPINGS],
    }


def observation_case_label_metadata() -> dict[str, object]:
    """Return compact crosswalk metadata for ``AnatomyModel``."""
    return {
        "vocabulary": "Hiscox et al. (2020) MRE134 publication ROIs",
        "label_volume_in_release": False,
        "mapping_file": "examples/cases/mre134_recode_label_mapping.json",
        "mapping_target": "ReCoDE active part_list_full.k",
        "labels": {
            mapping.mre134_label: {
                "abbreviation": mapping.abbreviation,
                "category": mapping.category,
                "unified_label": mapping.unified_label,
                "recode_part_ids": [part.part_id for part in mapping.recode_parts],
                "match_kind": mapping.match_kind,
            }
            for mapping in MRE134_RECODE_LABEL_MAPPINGS
        },
    }


def write_mapping(path: str | Path = DEFAULT_MAPPING_OUTPUT) -> Path:
    """Write the crosswalk as stable, human-readable JSON."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(mapping_document(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(write_mapping())
