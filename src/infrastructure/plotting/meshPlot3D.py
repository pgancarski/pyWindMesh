import pyvista as pv
import numpy as np
import cmocean

from infrastructure.mesh import StructuredHexMesh3d
from config import GroundMeshConfig
from infrastructure.exporters import grid3D_to_pyvista
from infrastructure.plotting import GroundGridPlot


class CrossectionMeshPlot3D:
    def __init__(self, wind_direction=None, center_x=None, center_y=None):
        self.wind_direction = wind_direction
        self.center_x = center_x
        self.center_y = center_y

    def _set_configs(self, config: GroundMeshConfig):
        if self.wind_direction is None:
            self.wind_direction = config.wind_direction

        if self.center_x is None:
            self.center_x = config.center_x

        if self.center_y is None:
            self.center_y = config.center_y

    def plot(self, mesh: StructuredHexMesh3d):

        grid3d = mesh.mesh

        # define wd and center if undefined
        self._set_configs(mesh.config.ground_mesh)

        # === BUILD ROTATED VOLUME ===
        pv_mesh = grid3D_to_pyvista(
            grid3d,
            wind_direction=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y
        )

        # === COMPUTE MID Y CLIP ===
        slice_width = 10 * mesh.config.ground_mesh.farm_cellsize_x + mesh.config.ground_mesh.farm_cellsize_y
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


class NonOrthogonalityHotspotsPlot3D(CrossectionMeshPlot3D):
    """
    Plots the ground surface plus only those volumetric cells whose
    non_orthogonality_deg field exceeds a specified threshold.
    """
    def __init__(self, threshold_deg: float = 60.0,
                 wind_direction=None, center_x=None, center_y=None):
        super().__init__(wind_direction=wind_direction,
                         center_x=center_x,
                         center_y=center_y)
        self.threshold_deg = float(threshold_deg)

    def plot(self, mesh: StructuredHexMesh3d):
        grid3d = mesh.mesh

        self._set_configs(mesh.config.ground_mesh)

        if "non_orthogonality_deg" not in grid3d.cell_values:
            raise KeyError(
                "Grid3D.cell_values does not contain 'non_orthogonality_deg'. "
                "Ensure the mesh quality metric is computed before plotting."
            )

        pv_mesh = grid3D_to_pyvista(
            grid3d,
            wind_direction=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y
        )

        non_orth = pv_mesh.cell_data["non_orthogonality_deg"]
        mask = non_orth > self.threshold_deg
        hotspot_ids = np.nonzero(mask)[0]

        hotspots = (
            pv_mesh.extract_cells(hotspot_ids)
            if hotspot_ids.size > 0 else None
        )

        ground_plotter = GroundGridPlot(
            wind_direction=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y
        )

        ground_surface = ground_plotter.build_pyvista_grid(
            mesh.ground_mesh.to_ground_grid(),
            field_name="Z"
        )

        p = pv.Plotter()

        p.add_mesh(
            ground_surface,
            cmap=cmocean.tools.crop(cmocean.cm.topo, 0, 1, pivot=0),
            show_edges=True,
            name="Ground"
        )

        if hotspots is not None and hotspots.n_cells > 0:
            vmax = float(non_orth[mask].max())
            if np.isclose(vmax, self.threshold_deg):
                vmax += 1e-6  # avoid zero range
            p.add_mesh(
                hotspots,
                scalars="non_orthogonality_deg",
                cmap="plasma",
                clim=[self.threshold_deg, vmax],
                show_edges=True,
                opacity=0.85,
                name="NonOrthHotspots"
            )
        else:
            p.add_text(
                f"No cells exceed {self.threshold_deg:.1f}°",
                position="upper_left",
                font_size=10,
                name="NoHotspotsLabel"
            )

        p.add_mesh(
            pv_mesh.outline(),
            color="k",
            name="DomainOutline"
        )

        p.show_axes()
        p.show()