"""Brain-strain visualisation, analysis, and research-data adapters.

The UI objects are loaded lazily so importing :mod:`brain_strain` does not
immediately initialize PyVista and VTK.
"""

from typing import Any

__all__ = [
    "BrainComparisonUI",
    "BrainLauncherUI",
    "BrainUI",
    "create_ui_from_args",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .viewer.app import (
        BrainComparisonUI,
        BrainLauncherUI,
        BrainUI,
        create_ui_from_args,
    )

    return {
        "BrainComparisonUI": BrainComparisonUI,
        "BrainLauncherUI": BrainLauncherUI,
        "BrainUI": BrainUI,
        "create_ui_from_args": create_ui_from_args,
    }[name]
