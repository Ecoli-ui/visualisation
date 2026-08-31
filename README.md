# Brain Strain Visualisation

This project provides an interactive PyVista viewer for brain meshes and
time-dependent cell data. It can open LS-DYNA `.k` files directly, as well as
VTK and common surface-mesh formats. Temporal PVD, XDMF, EnSight, and Exodus
files are expanded into all of the time frames reported by their readers.

## Install and test

Python 3.11 or newer is required. Install the viewer, its LS-DYNA support, and
the development tools from a source checkout:

```bash
python -m pip install -e ".[lsdyna,dev]"
brain-strain
```

The application can also be started with `python -m brain_strain`. Run the
fast generated-fixture suite and the optional external-data integration suite
with:

```bash
python -m unittest discover -s tests/unit -t . -v
python -m unittest discover -s tests/integration -t . -v
```

## Package the desktop application

PyInstaller builds for the operating system on which it is running; it does
not cross-compile. Install the packaging dependency and LS-DYNA support, then
run the checked-in specification:

```bash
python -m pip install -e ".[lsdyna,package]"
python -m PyInstaller --clean --noconfirm brain_strain.spec
```

On Windows this creates the standalone `dist/BrainStrain.exe`. On macOS it
creates `dist/BrainStrain.app`; the app has to be signed and notarized before
general distribution outside a development machine. A build made on Apple
Silicon targets Apple Silicon, while an Intel build targets Intel unless a
universal Python environment is used.

PyInstaller does not support iOS, and a Windows `.exe` cannot run on iPhone or
iPad. An iOS version would be a separate application/port with an iOS-capable
UI and build toolchain, not another output of this specification.

Large research releases are optional external data dependencies; their
locations, environment-variable overrides, and test behavior are described in
[`data/README.md`](data/README.md). See
[`docs/architecture.md`](docs/architecture.md) for the module layout.

## Research metadata

[`ObservationCase`](src/brain_strain/observation_case.py) defines the research record that
surrounds a strain dataset. It covers case identity and provenance, subject or
specimen details, anatomy and imaging, coordinate/unit conventions, loading
and kinematics, acquisition instruments, finite-element setup and materials,
strain definitions, data assets, processing, derived metrics and outcomes,
quality/validation, and data governance. Large meshes and arrays remain in
their native files and are referenced by asset ID.

All scientific sections are optional so incomplete legacy data can still be
described. Each record also has an `extensions` object for study-specific
information, making the model open-ended rather than claiming that one fixed
schema can anticipate every future strain-research method. See
[`examples/cases/observation_case.json`](examples/cases/observation_case.json) for a case
describing this repository's reduced-order generalized Maxwell simulation.

```python
from brain_strain.observation_case import ObservationCase

case = ObservationCase.read_json("examples/cases/observation_case.json")
print(case.strain_observations[0].measure)
print(case.validate_references())  # empty when every internal ID resolves
```

## MRE134 adapter and case comparison

[`mre134`](src/brain_strain/adapters/mre134.py) reads the zipped MRE134 group
stiffness and damping NIfTI images directly, converts the stored stiffness
scale to kPa, preserves the MNI affine, and represents NIfTI voxels as PyVista
cells. Cell centres therefore coincide with the original voxel centres and the
atlas can use the viewer's existing cell-scalar path.

```python
from brain_strain.adapters.mre134 import (
    CONTRIBUTOR_COUNT_ARRAY,
    STIFFNESS_ARRAY,
    compare_mre134_with_case,
    load_mre134,
)
from brain_strain.io.loader import load_observation_case

mre = load_mre134(
    "data/external/MRE134-master",
    include_contributor_count=True,
)
print(mre.dataset.cell_data[STIFFNESS_ARRAY])
print(mre.dataset.cell_data[CONTRIBUTOR_COUNT_ARRAY])
print(mre.case.validate_references())

simulation = load_observation_case("examples/cases/observation_case.json")
comparison = compare_mre134_with_case(mre.case, simulation)
print(comparison.raw_value_comparison_valid)  # False: unlike quantities
print(comparison.issues)
```

For the viewer-facing decision, use the generic comparison function. It also
accepts a `LoadedData` instance, so its `metadata` can represent the currently
loaded observation case:

```python
from brain_strain.comparison.policy import decide_observation_case_comparability

decision = decide_observation_case_comparability(mre.as_loaded_data(), simulation)
print(decision.is_comparable)                    # True for viewer comparison
print(decision.physical_calculation_applicable)  # Always False in this tool
print(decision.field_pairs[0].limitations)
```

`is_comparable` means that at least one scalar pair can use the viewer's
cell-data rendering path on compatible anatomy. It does not mean that the
fields are scientifically equivalent. The returned report separately retains
quantity, units, native and display associations, coordinate systems, temporal
semantics, label-mapping applicability, allowed operations, and prohibited
interpretations. Missing case metadata produces a structured negative result
rather than an exception.

Regenerate the repository case metadata, compatibility report, and a VTI that
can be opened directly by the viewer:

```bash
mre134-adapter \
  --contributors \
  --vti-output outputs/mre134/adapter/MRE134_atlas.vti

brain-strain \
  outputs/mre134/adapter/MRE134_atlas.vti \
  --field MRE134_Stiffness_kPa
```

The generated records are
[`examples/cases/mre134_observation_case.json`](examples/cases/mre134_observation_case.json)
and
[`examples/cases/mre134_vs_simulation_comparison.json`](examples/cases/mre134_vs_simulation_comparison.json).
The comparison intentionally refuses a physical value comparison with the
generalized Maxwell response: MRE134 contains static 50-Hz material
properties, whereas the simulation contains a time-dependent maximum shear
strain response on an unrelated mesh frame.

The corresponding
[viewer-scoped report](examples/cases/mre134_vs_simulation_visualisation_comparison.json)
permits side-by-side and independently normalized visualisation, while
rejecting raw-value comparison and an unregistered MNI-voxel/ReCoDE-mesh
overlay.

### MRE134/ReCoDE anatomical label mapping

MRE134's released NIfTI files are continuous stiffness and damping maps; they
do not contain an anatomical integer-label volume. The crosswalk therefore
uses the 34 ROI names reported in the MRE134 publication and maps them to the
active ReCoDE part names and IDs in `part_list_full.k`.

The complete machine-readable comparison is
[`examples/cases/mre134_recode_label_mapping.json`](examples/cases/mre134_recode_label_mapping.json).
Its mapping quality is explicit:

- Six bilateral subcortical ROIs map directly to paired left/right ReCoDE
  parts: amygdala, caudate, hippocampus, pallidum, putamen, and thalamus.
- Corpus callosum maps to ReCoDE part IDs 251–255.
- Eleven other named white-matter tracts map only to the broader left/right
  cerebral-white-matter parts.
- Twelve MRE134 cortical parcels map only to the broader left/right
  cerebral-cortex parts.
- The MRE134 whole-brain mask has no single ReCoDE part equivalent.

Look up an MRE134 name or abbreviation without depending on punctuation or
case:

```python
from brain_strain.adapters.mre134_labels import map_mre134_label

mapping = map_mre134_label("TH")
print(mapping.unified_label)  # Thalamus
print([part.part_id for part in mapping.recode_parts])  # [10, 49]
```

When a ReCoDE LS-DYNA mesh is converted with `k-to-vtu`, each cell
also receives `mre134_unified_label`. This is a semantic name crosswalk only;
it does not register MNI voxels to the subject mesh or recreate the MRE134 ROI
masks.

## Run the viewer

Open the empty launcher, then use **Open local mesh** to select a model:

```bash
brain-strain
```

No model or scalar data is loaded before a file is selected.

Open a particular mesh, including an LS-DYNA `.k` file:

```bash
brain-strain path/to/mesh.k
```

### Compare observation and simulation cases

Supplying `--observation-mesh` opens the redesigned comparison UI. The
observation and simulation are shown simultaneously with independent field
names, colour scales, frame numbers, and times:

```bash
brain-strain \
  data/external/brain-meshing/coarse_7-23-2020/coarse_brain_with_regions.vtk \
  --field MPS \
  --simulation-case examples/cases/observation_case.json \
  --observation-mesh outputs/mre134/adapter/MRE134_atlas.vti \
  --observation-field MRE134_Stiffness_kPa \
  --observation-case examples/cases/mre134_observation_case.json
```

The shared **Comparison position** slider advances each case by relative
timeline progress, so a static observation remains on its only frame while a
time-dependent simulation advances normally. The panels retain separate units
and scalar ranges; metadata differences are listed in the simulation panel.

**Show normalized visual difference** (or `D`) opens a third panel only when
the two meshes have identical cell geometry. It displays
`normalized observation - normalized simulation` in the range `[-1, 1]`.
This is a unit-free visual contrast, not a physical error or validation
metric. For unregistered MRE134 voxels and a ReCoDE mesh, the control is
disabled and the UI explains that registration and resampling are required.
Use `--show-difference` to request the panel at startup when it is available.

Use the environment in which the project was installed, such as the local
`.venv`, when invoking the console commands.

### Dryad brain-motion atlas

Files named `HR_MESH_<frame>.vtk` and `NE_MESH_<frame>.vtk` are handled by
[`dryad`](src/brain_strain/adapters/dryad.py). Opening any one frame discovers the
complete head-rotation (`HR`) or neck-extension (`NE`) sequence beside it,
orders the numbered files, validates their tetrahedral topology and expected
arrays, and uses the release's 18 ms frame interval. Each frame retains its
own deformed point coordinates.

`GmaxT2` is selected automatically when the generic default `MPS` is absent:

```bash
brain-strain data/external/DRYAD20210528/HR_MESH_1.vtk
```

Select another scalar cell field explicitly when required:

```bash
brain-strain data/external/DRYAD20210528/NE_MESH_1.vtk --field GmaxT3
brain-strain data/external/DRYAD20210528/HR_MESH_1.vtk --field E1
```

The scalar viewer accepts `GmaxT1`, `GmaxT2`, `GmaxT2_std`, `GmaxT3`,
`GmaxT3_std`, and `E1`. `V1` is preserved as three-component cell data but is
not treated as a scalar time series. Loading a full ten-frame sequence may use
substantial memory because the source files are large ASCII meshes.

The viewer always prefers a real scalar series embedded in the model or
supplied with `--scalar-series`. When real results are present, **Show
simulation results** explicitly switches to the reduced-order generalized
Maxwell response and can be switched off to return to the real data. If no
real scalar series is available, the viewer starts directly with the
simulation. A prominent badge and the status panel always identify the active
source as **REAL RESULT DATA** or **SIMULATED DATA — DEMONSTRATION ONLY**.

The time-frame count comes from the loaded scalar series. If the model has no
scalar series but contains embedded or supplied time values, their count is
used. A static mesh with neither source uses ten frames at 18 ms spacing
(0.0–0.162 s), matching the sampling of the associated in-vivo brain-motion
atlas. `--frames NUMBER` and `--duration SECONDS` override that sampling.

### Generalized Maxwell simulation

[`simulation`](src/brain_strain/simulation.py) implements the relaxation modulus

```text
G(t) = G0 [r_inf + sum_i g_i exp(-t / tau_i)]
r_inf + sum_i g_i = 1
```

and integrates the independent Maxwell-branch stresses under a prescribed
piecewise-linear shear-stress history. Every simulation generates two presets
on the same timeline: Case A uses `rotation_axis = (0, 0, 1)` and Case B uses
`rotation_axis = (1, 0, 0)`. The spatial load grows with distance from the
selected axis. A fixed reference load is calibrated to the reported atlas
maximum-shear-strain reference for neck rotation (0.017) or neck extension
(0.011); adjustable model responses are not rescaled, so material changes
remain visible. The default temporal spacing and strain scales reference
[Gomez et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8220272/).
The paper reports measured kinematics, not Maxwell coefficients: the default
normalized moduli and relaxation times are declared modelling assumptions.

Configure the material branches and impact mode from the command line:

```bash
brain-strain path/to/mesh.vtk \
  --impact-mode neck-extension \
  --branches 4 \
  --modulus-kind G0 \
  --estimated-modulus 2.0 \
  --modulus-scale 1.25 \
  --r-infinity 0.4 \
  --gi 0.25 0.15 0.12 0.08 \
  --log10-tau -3 -2 -1 0
```

The adjustable constraints are:

- \(N=3\ldots6\) Maxwell branches.
- \(E_0\) or \(G_0=0.5\ldots2.0\) times an estimated value. When `E0` is
  selected it is converted to shear modulus using `--poisson-ratio`.
- \(r_\infty=E_\infty/E_0=G_\infty/G_0=0.001\ldots0.999\).
- Positive \(g_i\) satisfying \(r_\infty+\sum_i g_i=1\).
- Positive, strictly increasing τ values. Omit explicit values to generate
  `N` log-spaced times over `--tau-range`, or pass their base-10 logarithms
  through `--log10-tau`.

When simulated results are active, the same quantities appear as sliders in a
separate **Simulation parameters** window after clicking **Open simulation
parameters** in the main results window, leaving the 3D model unobstructed.
Its fonts, slider geometry, and column layout respond to window resizing;
very narrow windows switch to a single-column control list, and fullscreen
mode uses a centered three-column layout for model parameters, material
fractions, and relaxation times. Moving a slider recomputes the response
immediately. Changing
`r_inf` rescales all active `g_i`; changing one `g_i` redistributes the
remaining transient fraction over the other branches so the Prony constraint
continues to hold. Inactive `g_i` and `tau_i` sliders are hidden when `N` is
below six. Relaxation-time sliders display `log10(tau / s)`.

Use the **Case A** and **Case B** selectors to show either preset. Selecting
both closes and disables **Show slices**, then places Case A on the left and
Case B on the right. **Diverging A − B colours** replaces that layout with one
full-width signed-comparison model: red means A is higher, blue means B is
higher, white means the values are similar, and grey means the cell is missing
from either case.

This calculation is a reduced-order constitutive response at independent
material points. It does not solve finite-element momentum balance, contact,
or anatomical boundary conditions and is not physically validated.

### Excel result output

Click **Open result output** (or press `E`) to open a separate, unobstructed
result section. It shows the active source, frame, time, real-data maximum,
and calculated Case A/Case B maximum strains. **Export Excel (.xlsx)** opens
a native save dialog and writes an analysis-ready workbook containing:

- **Results Summary** for the simulation method, active state, calculated
  maximum strain, actual real-data maximum, and their peak locations/times.
- **Parameters** for the selected impact, material, Maxwell-branch, loading,
  threshold, frame, and selection settings.
- **Frame Results** for typed numeric minimum, maximum, mean, standard
  deviation, finite-cell count, and peak cell at every time step.
- **Selected Cell Results** for the real and simulated values and time-history
  statistics of every selected cell.
- **Data Dictionary** for field definitions and interpretation limits.

When real result data are loaded, the workbook retains the real series and
both generalized-Maxwell presets even if only one source is currently
displayed. It keeps them separate and does not assume matching units or
scientific equivalence. The workbook schema is extendable through
`ResultWorkbook.add_sheet()` in [`export`](src/brain_strain/io/export.py).

## On-screen controls

- **Open local mesh** opens a native file-selection window.
- **Open result output** opens the current metrics and Excel export section in
  a separate window.
- **Show slices** toggles orthogonal slices in the 3D model and opens
  sagittal, coronal, and axial 2D colour views on its right.
- **Show parts** colours the mesh by tissue or part ID.
- **Show simulation results** switches from real results to the clearly
  labelled demonstration series. It remains selected when no real results are
  available.
- **Case A** and **Case B** select the fixed Z-axis and X-axis simulations.
  Selecting both shows the models side by side and turns slices off.
- **Diverging A − B colours** compares both cases with red/blue/white values
  and grey missing cells.
- **Maxwell parameter sliders** appear in a separate **Simulation parameters**
  window only after clicking **Open simulation parameters**. They adjust `N`,
  `E0`/`G0`, `r_inf`, every active `g_i`, and every active logarithmic `tau_i`
  without covering the model.
- In comparison mode, **Show normalized visual difference** adds the optional
  third panel when both cases share identical cell geometry.
- **Drag select cells** switches left-dragging between rectangle cell
  selection and camera rotation. Drag selection starts enabled; a single
  click selects the cell directly under the pointer.
- Type a zero-based cell index and press **Enter** to select that scalar-array
  entry. When the source preserves an element ID, the viewer reports that
  element number separately.
- **Hotspots** toggles cells above the selected threshold.
- **Clear selection** removes the current cell selection.
- The top sliders select the time frame and hotspot threshold.

The tissue colours used by **Show parts** are:

- Skin: red
- Skull: light blue
- CSF: green
- Grey matter: yellow
- White matter: brown
- Ventricles: dark blue
- Unclassified parts: light grey

## Mouse and keyboard shortcuts

| Input | Action |
| --- | --- |
| Left drag | Select cells inside the rectangle when **Drag select cells** is enabled |
| Left drag | Rotate the model when **Drag select cells** is disabled |
| Mouse wheel or `+` / `-` | Zoom |
| `R` | Toggle between drag selection and camera rotation |
| Digits | Enter a zero-based cell index |
| `Backspace` | Remove the last cell-index digit |
| `Enter` | Select the entered cell index |
| `Esc` | Clear the cell-index input |
| `C` | Clear selected cells |
| `O` | Open a local mesh |
| `L` | Toggle orthogonal slices |
| `B` | Toggle tissue or part colours |
| `P` | Go to the global peak |
| `E` | Open the result output and Excel export window |
| `S` | Save a screenshot |
| `V` | Reset the camera |
| `D` | Toggle normalized visual difference in comparison mode |
