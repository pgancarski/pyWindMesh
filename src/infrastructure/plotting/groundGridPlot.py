import numpy as np
import pyvista as pv
import cmocean

from src.application.interfaces import MeshPlotter
from src.infrastructure.topography.topographyUtils import un_rotate_points

from src.domain import Mesh2D
from src.domain import Grid2D


class GroundGridPlot(MeshPlotter):
    def __init__(self, wind_direction:float = 0, center_x:float = 0, center_y:float = 0):
        super().__init__()
        self.wind_direction=wind_direction
        self.center_x=center_x
        self.center_y=center_y

    def build_pyvista_grid(self, grid: Grid2D, field_name: str = "Z") -> pv.StructuredGrid:
        X = grid.X
        Y = grid.Y

        try:
            Z = grid.point_values["Z"]
        except KeyError as exc:
            raise KeyError('Grid must contain a "Z" field in point_values for elevation.') from exc

        C = grid.point_values[field_name]

        if X.shape != Y.shape or Z.shape != X.shape or C.shape != X.shape:
            raise ValueError("Shape mismatch in ground grid")

        nx, ny = X.shape

        sg = pv.StructuredGrid()

        points = np.column_stack(
            [X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
        )

        # ✅ SAME rotation logic reused
        un_rotate_points(
            points[:, :2],
            angle_deg=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y,
            inplace=True
        )

        sg.points = points
        sg.dimensions = (nx, ny, 1)

        sg[field_name] = C.ravel(order="F")

        # color faces not vertices
        return sg.point_data_to_cell_data()


    def plot(self, mesh: Mesh2D, field_name: str = "Z") -> None:
        grid = mesh.to_ground_grid()
  
        sg_cell = self.build_pyvista_grid(grid, field_name)

        p = pv.Plotter()
        p.add_mesh(
            sg_cell,
            scalars=field_name,
            show_edges=True,
            cmap=cmocean.tools.crop(cmocean.cm.topo, 0, 1, pivot=0),
        )
        p.show_axes()
        p.show()
