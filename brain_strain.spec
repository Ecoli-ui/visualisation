# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller definition for the desktop viewer.

PyInstaller must run on the target operating system.  This spec emits a
single ``BrainStrain.exe`` on Windows and a ``BrainStrain.app`` bundle on
macOS.  iOS is not a supported PyInstaller target.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "examples" / "cases"), "examples/cases"),
]
datas += collect_data_files("pyvista")
datas += copy_metadata("brain-strain-visualisation")

# The source adapters are imported only when their matching data format is
# opened.  Listing them explicitly keeps those paths available in the frozen
# application even when PyInstaller cannot infer the lazy imports.
hiddenimports = [
    "brain_strain.adapters.dryad",
    "brain_strain.adapters.lsdyna",
    "brain_strain.adapters.mre134",
    "brain_strain.adapters.mre134_labels",
    "lsdyna_mesh_reader",
]

a = Analysis(
    [str(ROOT / "build_tools" / "brain_strain_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == "win32":
    # A one-file build gives Windows users the requested standalone .exe.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="BrainStrain",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="BrainStrain",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
    )
    collected = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="BrainStrain",
    )

    if sys.platform == "darwin":
        app = BUNDLE(
            collected,
            name="BrainStrain.app",
            icon=None,
            bundle_identifier="org.brainstrain.viewer",
            info_plist={
                "CFBundleDisplayName": "Brain Strain",
                "CFBundleName": "Brain Strain",
                "NSHighResolutionCapable": True,
            },
        )
