import numpy as np
import pyvista as pv

from domain import Grid3D
from infrastructure.topography.topographyUtils import un_rotate_points


def _prepare_vtk_data(arr: np.ndarray, n_tuples: int) -> np.ndarray:
    """
    Convert structured (i,j,k) arrays into the flat shape VTK expects.
    Scalars → (n_tuples,), vectors → (n_tuples, n_comp).
    """
    flat = np.asarray(arr)

    if flat.ndim == 1:
        if flat.size != n_tuples:
            raise ValueError(f"Array of length {flat.size} does not match expected {n_tuples}")
        return flat

    # Split component axis (if any) from spatial axes
    spatial_axes = flat.shape[:3]
    n_components = int(np.prod(flat.shape[3:])) if flat.ndim > 3 else 1

    if np.prod(spatial_axes) != n_tuples:
        raise ValueError(
            f"Array with spatial shape {spatial_axes} does not match expected {n_tuples} tuples"
        )

    flat = np.ascontiguousarray(flat).reshape(-1, n_components, order="F")
    return flat[:, 0] if n_components == 1 else flat


def grid3D_to_pyvista(
    grid: Grid3D,
    wind_direction=0,
    center_x=0,
    center_y=0
) -> pv.StructuredGrid:
    mesh = pv.StructuredGrid(grid.X, grid.Y, grid.Z)

    # === APPLY ROTATION TO ALL POINTS ===
    points = mesh.points.copy()
    un_rotate_points(
        points[:, :2],
        angle_deg=wind_direction,
        center_x=center_x,
        center_y=center_y,
        inplace=True
    )
    mesh.points = points

    n_points = mesh.n_points
    n_cells = mesh.n_cells

    # Point data
    for name, arr in grid.point_values.items():
        mesh.point_data[name] = _prepare_vtk_data(arr, n_points)

    # Cell data
    for name, arr in grid.cell_values.items():
        mesh.cell_data[name] = _prepare_vtk_data(arr, n_cells)

    return mesh