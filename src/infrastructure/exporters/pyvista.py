import pyvista as pv

from domain import Grid3D
from infrastructure.topography.topographyUtils import un_rotate_points


def grid3D_to_pyvista(grid:Grid3D, wind_direction=0, center_x=0, center_y=0) -> pv.StructuredGrid:
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

    # Point data
    for name, arr in grid.point_values.items():
        mesh.point_data[name] = arr

    # Cell data
    for name, arr in grid.cell_values.items():
        mesh.cell_data[name] = arr

    return mesh