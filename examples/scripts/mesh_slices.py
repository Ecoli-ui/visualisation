"""Display the brain surface and slices of the volume mesh with PyVista."""

from argparse import ArgumentParser

import pyvista as pv

from brain_strain.paths import DEFAULT_MESH

MESH_FILE = DEFAULT_MESH


def add_region_mesh(plotter: pv.Plotter, dataset, **kwargs) -> None:
    """Add a mesh using the integer RegionID cell labels as categories."""
    plotter.add_mesh(
        dataset,
        scalars="RegionID",
        preference="cell",
        categories=True,
        cmap="tab10",
        clim=(1, 9),
        scalar_bar_args={"title": "Region ID", "n_labels": 9},
        **kwargs,
    )


def show_surface_and_slices(mesh: pv.UnstructuredGrid) -> None:
    """Show the outer surface and three centre slices side by side."""
    surface = mesh.extract_surface(algorithm="dataset_surface")
    slices = mesh.slice_orthogonal()  # defaults to the mesh centre

    plotter = pv.Plotter(shape=(1, 2), window_size=(1500, 750))

    plotter.subplot(0, 0)
    add_region_mesh(plotter, surface, show_edges=False, smooth_shading=True)
    plotter.add_text("Outer surface", font_size=12)
    plotter.show_axes()

    plotter.subplot(0, 1)
    # A faint surface supplies context around the three orthogonal cuts.
    plotter.add_mesh(surface, color="white", opacity=0.12)
    add_region_mesh(plotter, slices, show_edges=True, line_width=1)
    plotter.add_text("Orthogonal centre slices", font_size=12)
    plotter.show_axes()

    plotter.link_views()
    plotter.show()


def show_interactive_slice(mesh: pv.UnstructuredGrid) -> None:
    """Show one slice controlled by a draggable plane widget."""
    surface = mesh.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(window_size=(1000, 800))
    plotter.add_mesh(surface, color="white", opacity=0.15)
    plotter.add_mesh_slice(
        mesh,
        normal="x",
        scalars="RegionID",
        preference="cell",
        categories=True,
        cmap="tab10",
        clim=(1, 9),
        show_edges=True,
        scalar_bar_args={"title": "Region ID", "n_labels": 9},
    )
    plotter.add_text("Drag or rotate the plane to move the slice", font_size=12)
    plotter.show_axes()
    plotter.show()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="use a draggable slicing plane instead of three fixed slices",
    )
    args = parser.parse_args()

    mesh = pv.read(MESH_FILE)
    if args.interactive:
        show_interactive_slice(mesh)
    else:
        show_surface_and_slices(mesh)


if __name__ == "__main__":
    main()
