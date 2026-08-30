# External research data

Large research releases are local inputs, not application source or package
content. Place them under the ignored `data/external/` directory:

```text
data/external/
├── brain-meshing/
├── DRYAD20210528/
├── MRE134-master/
└── ReCoDE-brain-mesh-creation-main/
```

These retain their upstream directory names so their papers and release notes
remain easy to identify. The default demonstration mesh, Dryad sequences,
MRE134 atlas, and ReCoDE reference assets are resolved from this directory.

To keep the releases elsewhere, set `BRAIN_STRAIN_DATA_DIR` to the directory
that contains those four folders. `BRAIN_STRAIN_PROJECT_ROOT` can override the
checkout location, and `BRAIN_STRAIN_OUTPUT_DIR` can redirect generated files;
all three values accept absolute or user-relative paths.

Generated viewer and adapter artifacts belong under `outputs/`, not inside an
upstream release. Existing MRE134 VTI and comparison output is organized under
`outputs/mre134/`.

Unit tests use generated or small checked-in fixtures. Integration tests that
need a complete release live in `tests/integration/` and skip when their input
is unavailable.
