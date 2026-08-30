"""Configurable paths for checkout resources and local research data."""

import os
import sys
from pathlib import Path


def _checkout_root() -> Path:
    """Locate the source checkout without depending on the process cwd."""
    configured = os.environ.get("BRAIN_STRAIN_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    # PyInstaller extracts bundled data below ``sys._MEIPASS``.  Treat that
    # directory as the project root so packaged example metadata resolves in
    # the same way as it does from a checkout.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _checkout_root()
DATA_ROOT = Path(
    os.environ.get("BRAIN_STRAIN_DATA_DIR", PROJECT_ROOT / "data" / "external")
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("BRAIN_STRAIN_OUTPUT_DIR", PROJECT_ROOT / "outputs")
).expanduser().resolve()
EXAMPLE_CASES_ROOT = PROJECT_ROOT / "examples" / "cases"
DEFAULT_MESH = (
    DATA_ROOT
    / "brain-meshing"
    / "coarse_7-23-2020"
    / "coarse_brain_with_regions.vtk"
)
DEFAULT_DRYAD_ROOT = DATA_ROOT / "DRYAD20210528"
DEFAULT_MRE134_ROOT = DATA_ROOT / "MRE134-master"
DEFAULT_RECODE_ROOT = DATA_ROOT / "ReCoDE-brain-mesh-creation-main"
DEFAULT_SIMULATION_CASE = EXAMPLE_CASES_ROOT / "observation_case.json"


__all__ = [
    "DATA_ROOT",
    "DEFAULT_DRYAD_ROOT",
    "DEFAULT_MESH",
    "DEFAULT_MRE134_ROOT",
    "DEFAULT_RECODE_ROOT",
    "DEFAULT_SIMULATION_CASE",
    "EXAMPLE_CASES_ROOT",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
]
