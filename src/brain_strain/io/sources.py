"""Adapter registry for every mesh and research-data source.

All reads of numerical mesh or image data enter through this module. Dataset
adapters retain their domain-specific validation, while the generic PyVista
adapter is the final fallback for ordinary supported mesh formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pyvista as pv

PathInput = str | PathLike[str]


class SourceAdapterError(ValueError):
    """Raised when a source adapter cannot load or validate its data."""


@dataclass(slots=True)
class AdaptedSource:
    """Normalized output shared by every registered source adapter."""

    source: Path
    adapter_name: str
    mesh: pv.DataObject
    frames: tuple[pv.DataSet, ...]
    time: npt.NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(Protocol):
    """Runtime interface implemented by registered data-source adapters."""

    name: str

    def matches(self, source: Path) -> bool: ...

    def load(self, source: Path) -> AdaptedSource: ...


def _existing_source(path: PathInput) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Mesh source does not exist: {source}")
    return source


def _as_data_set(data: pv.DataObject, source: Path) -> pv.DataSet:
    """Collapse one reader output into the data set consumed by the UI."""
    if isinstance(data, pv.DataSet):
        result = data
    elif isinstance(data, pv.MultiBlock):
        blocks: list[pv.DataSet] = []

        def collect_blocks(collection: pv.MultiBlock) -> None:
            for block in collection:
                if isinstance(block, pv.DataSet):
                    blocks.append(block)
                elif isinstance(block, pv.MultiBlock):
                    collect_blocks(block)

        collect_blocks(data)
        if not blocks:
            raise SourceAdapterError(f"Mesh contains no data sets: {source}")
        result = blocks[0] if len(blocks) == 1 else pv.MultiBlock(blocks).combine()
    else:
        raise SourceAdapterError(
            f"PyVista returned an unsupported object for {source}: "
            f"{type(data).__name__}"
        )

    if result.n_points == 0 or result.n_cells == 0:
        raise SourceAdapterError(f"Mesh is empty: {source}")
    return result


def _validate_time(values: Any, source: Path) -> npt.NDArray[np.float64]:
    try:
        time = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SourceAdapterError(
            f"Time values in {source} must be numeric"
        ) from exc
    if time.ndim == 0:
        time = time.reshape(1)
    if time.ndim != 1 or time.size == 0:
        raise SourceAdapterError(
            f"Time values in {source} must be a non-empty vector"
        )
    if not np.isfinite(time).all():
        raise SourceAdapterError(
            f"Time vector contains NaN or infinity: {source}"
        )
    if np.any(np.diff(time) < 0):
        raise SourceAdapterError(
            f"Time values are not ordered from low to high: {source}"
        )
    return time


class DryadSourceAdapter:
    """Adapter wrapper for complete Dryad HR/NE frame sequences."""

    name = "dryad"

    def matches(self, source: Path) -> bool:
        if not source.is_file():
            return False
        from ..adapters.dryad import is_dryad_frame_path

        return is_dryad_frame_path(source)

    def load(self, source: Path) -> AdaptedSource:
        from ..adapters.dryad import DryadAdapterError, load_dryad_sequence

        try:
            sequence = load_dryad_sequence(source)
        except (DryadAdapterError, OSError, ValueError) as exc:
            raise SourceAdapterError(
                f"Could not load Dryad mesh sequence from: {source}"
            ) from exc
        return AdaptedSource(
            source=source,
            adapter_name=self.name,
            mesh=sequence.mesh,
            frames=sequence.frames,
            time=sequence.times,
            metadata=dict(sequence.metadata),
        )


class MRE134SourceAdapter:
    """Adapter wrapper for an MRE134 release directory or bundle member."""

    name = "mre134"
    _BUNDLE_MARKERS = frozenset(
        {
            "MRE134_Stiffness3D.nii.zip",
            "MRE134_Damping3D.nii.zip",
            "MRE134_Demographics.xlsx",
        }
    )

    @classmethod
    def _root(cls, source: Path) -> Path | None:
        root = source if source.is_dir() else source.parent
        if source.is_file() and source.name not in cls._BUNDLE_MARKERS:
            return None
        if all((root / name).is_file() for name in cls._BUNDLE_MARKERS):
            return root
        return None

    def matches(self, source: Path) -> bool:
        return self._root(source) is not None

    def load(self, source: Path) -> AdaptedSource:
        from ..adapters.mre134 import MRE134AdapterError, load_mre134

        root = self._root(source)
        if root is None:  # pragma: no cover - guarded by registry matching
            raise SourceAdapterError(f"Not an MRE134 bundle source: {source}")
        try:
            loaded = load_mre134(root)
        except (MRE134AdapterError, FileNotFoundError, OSError, ValueError) as exc:
            raise SourceAdapterError(
                f"Could not load MRE134 bundle from: {root}"
            ) from exc
        return AdaptedSource(
            source=source,
            adapter_name=self.name,
            mesh=loaded.dataset,
            frames=(loaded.dataset,),
            time=None,
            metadata=loaded.case.to_dict(omit_none=True),
        )


class LsDynaSourceAdapter:
    """Adapter for LS-DYNA keyword meshes."""

    name = "ls-dyna"

    def matches(self, source: Path) -> bool:
        return source.is_file() and source.suffix.casefold() == ".k"

    def load(self, source: Path) -> AdaptedSource:
        try:
            from ..adapters.lsdyna import convert_k_to_vtu

            with TemporaryDirectory(prefix="brain-mesh-") as directory:
                converted_path = Path(directory) / f"{source.stem}.vtu"
                convert_k_to_vtu(source, converted_path)
                raw = pv.read(converted_path)
                mesh = _as_data_set(raw, source)
        except Exception as exc:
            raise SourceAdapterError(
                f"Could not convert LS-DYNA mesh file: {source}"
            ) from exc
        return AdaptedSource(
            source=source,
            adapter_name=self.name,
            mesh=mesh,
            frames=(mesh,),
        )


class PyVistaSourceAdapter:
    """Fallback adapter for static and temporal PyVista-readable files."""

    name = "pyvista"

    def matches(self, source: Path) -> bool:
        return source.is_file()

    def _read_static(self, source: Path) -> AdaptedSource:
        try:
            raw = pv.read(source)
            mesh = _as_data_set(raw, source)
        except SourceAdapterError:
            raise
        except Exception as exc:
            raise SourceAdapterError(f"Could not read mesh file: {source}") from exc
        return AdaptedSource(
            source=source,
            adapter_name=self.name,
            mesh=raw,
            frames=(mesh,),
        )

    def load(self, source: Path) -> AdaptedSource:
        try:
            reader = pv.get_reader(source)
        except Exception:
            return self._read_static(source)

        if not hasattr(reader, "time_values"):
            return self._read_static(source)
        try:
            raw_times = reader.time_values
        except Exception as exc:
            raise SourceAdapterError(
                f"Could not read time values from mesh file: {source}"
            ) from exc
        if raw_times is None or len(raw_times) == 0:
            return self._read_static(source)

        times = _validate_time(
            raw_times,
            Path(f"<mesh-times:{source.name}>"),
        )
        frames: list[pv.DataSet] = []
        try:
            for index in range(times.size):
                reader.set_active_time_point(index)
                snapshot = _as_data_set(reader.read(), source)
                frames.append(snapshot.copy(deep=True))
        except SourceAdapterError:
            raise
        except Exception as exc:
            raise SourceAdapterError(
                f"Could not read all {times.size} time frames from: {source}"
            ) from exc

        if len({frame.n_cells for frame in frames}) != 1:
            raise SourceAdapterError(
                f"Mesh cell count changes across time frames in: {source}"
            )
        return AdaptedSource(
            source=source,
            adapter_name=self.name,
            mesh=frames[0],
            frames=tuple(frames),
            time=times,
        )


SOURCE_ADAPTERS: tuple[SourceAdapter, ...] = (
    MRE134SourceAdapter(),
    DryadSourceAdapter(),
    LsDynaSourceAdapter(),
    PyVistaSourceAdapter(),
)


def source_adapter_for(path: PathInput) -> tuple[Path, SourceAdapter]:
    """Resolve a source path and select its registered adapter."""
    source = _existing_source(path)
    adapter = next(
        (candidate for candidate in SOURCE_ADAPTERS if candidate.matches(source)),
        None,
    )
    if adapter is None:
        raise SourceAdapterError(f"No data adapter accepts source: {source}")
    return source, adapter


def load_adapted_source(path: PathInput) -> AdaptedSource:
    """Load numerical data exclusively through the selected source adapter."""
    source, adapter = source_adapter_for(path)
    return adapter.load(source)


__all__ = [
    "AdaptedSource",
    "SOURCE_ADAPTERS",
    "SourceAdapter",
    "SourceAdapterError",
    "load_adapted_source",
    "source_adapter_for",
]
