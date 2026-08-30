# Project structure

The repository uses a conventional `src` layout. Production code lives only
under `src/brain_strain`; the repository root contains project configuration
and documentation rather than importable modules.

```text
src/brain_strain/
├── analysis.py          # rendering-independent scalar analysis
├── simulation.py        # reduced-order generalized-Maxwell model
├── observation_case.py  # typed research metadata
├── paths.py             # checkout, data, example, and output locations
├── io/                  # loaders, source normalization, display data, exports
├── adapters/            # Dryad, MRE134, ReCoDE, and LS-DYNA integration
├── comparison/          # comparison policy and comparison presentation
└── viewer/              # PyVista rendering, dialogs, and application state
```

Dependencies point inward: adapters and I/O normalize external formats;
analysis and simulation operate on normalized data; comparison policy remains
independent of widgets; and `viewer/app.py` coordinates those components. The
package's `__init__.py` lazily exposes the main UI classes so a basic package
import does not initialize VTK.

Console scripts are declared in `pyproject.toml`. The main application can be
started with either `brain-strain` or `python -m brain_strain`; converter and
analysis commands import their package modules directly.

Tests mirror the runtime boundary:

- `tests/unit/` uses generated meshes and small checked-in case documents.
- `tests/integration/` requires optional complete research releases and skips
  when those inputs are unavailable.

Runnable code samples live in `examples/scripts/`, while curated metadata and
comparison documents live in `examples/cases/`.

Large upstream releases are ignored under `data/external/`. Generated images,
meshes, workbooks, and reports belong under the ignored `outputs/` tree. See
`data/README.md` for configuration and expected external-directory names.
