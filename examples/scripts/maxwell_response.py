"""Display one generalized-Maxwell maximum-shear-strain response frame."""

import pyvista as pv

from brain_strain.paths import DEFAULT_MESH
from brain_strain.simulation import simulate_generalized_maxwell_strain

MESH_FILE = DEFAULT_MESH


def main() -> None:
    mesh = pv.read(MESH_FILE)
    _, simulated_series = simulate_generalized_maxwell_strain(mesh)
    field_name = "Maximum shear strain"
    # Frame 3 is near the first response peak at the default ten-frame sampling.
    mesh.cell_data[field_name] = simulated_series[2]
    mesh.plot(scalars=field_name, show_edges=False)


if __name__ == "__main__":
    main()
