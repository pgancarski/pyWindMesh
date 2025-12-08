import pyvista as pv
import numpy as np
import cmocean

from domain import Grid3D

from infrastructure.mesh import StructuredHexMesh3d
from infrastructure.topography.topographyUtils import un_rotate_points
from infrastructure.plotting import GroundGridPlot

def to_pyvista(grid:Grid3D, wind_direction=0, center_x=0, center_y=0) -> pv.StructuredGrid:
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


class CrossectionMeshPlot3D:
    def __init__(self, wind_direction=0, center_x=0, center_y=0):
        self.wind_direction = wind_direction
        self.center_x = center_x
        self.center_y = center_y

    def plot(self, mesh: StructuredHexMesh3d):

        grid3d = mesh.mesh

        # === BUILD ROTATED VOLUME ===
        pv_mesh = to_pyvista(
            grid3d,
            wind_direction=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y
        )

        # === COMPUTE MID Y CLIP ===
        slice_width = 2 * mesh.config.ground_mesh.farm_cellsize_x + mesh.config.ground_mesh.farm_cellsize_y
        slice_y = self.get_rotated_slice(pv_mesh, width=slice_width)

        # === BUILD GROUND GRID ===
        ground_plotter = GroundGridPlot(
            wind_direction=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y
        )

        ground_surface = ground_plotter.build_pyvista_grid(
            mesh.ground_mesh.to_ground_grid(),
            field_name="Z"
        )

        # === PLOT EVERYTHING TOGETHER ===
        p = pv.Plotter()

        # Ground surface
        p.add_mesh(
            ground_surface,
            cmap=cmocean.tools.crop(cmocean.cm.topo, 0, 1, pivot=0),
            show_edges=True,
            name="Ground"
        )

        # Middle Y slice
        p.add_mesh(
            slice_y,
            cmap="viridis",
            show_edges=True,
            name="CrossSection"
        )

        # Optional: Add transparent outer volume for context
        p.add_mesh(
            pv_mesh.outline(),
            color="k"
        )

        p.show_axes()
        p.show()

    def get_rotated_slice(self, pv_mesh, width=10):
        # === COMPUTE CENTER FROM MESH ===
        xmin, xmax, ymin, ymax, _, _ = pv_mesh.bounds
        center = np.array([
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0
        ])

        # === ROTATE SLICE NORMAL ===
        theta = np.deg2rad(self.wind_direction)
        c, s = np.cos(theta), np.sin(theta)
        normal_xy = np.array([[c, -s], [s, c]]) @ np.array([1.0, 0.0])
        slice_normal = np.array([normal_xy[0], normal_xy[1], 0.0])
        slice_normal = slice_normal / np.linalg.norm(slice_normal)

        # === PICK CELLS WHOSE CENTERS ARE CLOSE TO THE PLANE ===
        cell_centers = pv_mesh.cell_centers().points
        rel = cell_centers - center
        distances = np.dot(rel, slice_normal)

        mask = np.abs(distances) <= width
        cell_ids = np.nonzero(mask)[0]

        if cell_ids.size == 0:
            return pv.PolyData()

        return pv_mesh.extract_cells(cell_ids)


