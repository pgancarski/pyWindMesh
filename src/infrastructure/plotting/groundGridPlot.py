import numpy as np
import pyvista as pv
import cmocean

from src.application.interfaces import MeshPlotter
from src.domain import Mesh2D
from src.domain import Grid2D


class GroundGridPlot(MeshPlotter):
    def __init__(self):
        super().__init__()

        self.X = None
        self.Y = None
        self.Z = None

    def plot(self, mesh: Mesh2D) -> None:
        grid = mesh.to_ground_grid()
        self.plot_pyvista(grid)

        self.X = grid.X
        self.Y = grid.Y
        self.Z = grid.point_values["Z"]

        #self.plot_plotly()

    def plot_pyvista(self, grid: Grid2D, field_name: str = "Z") -> None:
        """
        Plot a 2D structured grid using pyvista, with faces colored by Z values
        and visible edges.

        Parameters
        ----------
        grid : Grid2D
            Instance of Grid2D containing X, Y, and point_values[field_name].
        field_name : str, optional
            Name of the point field to use as Z, by default "Z".
        """
        X = grid.X
        Y = grid.Y
        Z = grid.point_values[field_name]

        if Z.shape != X.shape or Z.shape != Y.shape:
            raise ValueError(
                f"Shape mismatch: X{X.shape}, Y{Y.shape}, {field_name}{Z.shape}"
            )

        nx, ny = X.shape

        # Build a StructuredGrid: points are (X, Y, Z)
        sg = pv.StructuredGrid()
        # Use Fortran order because meshgrid with indexing='ij' is column-major in sense of pyvista
        sg.points = np.column_stack(
            [X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
        )
        # Dimensions are number of points in each direction
        sg.dimensions = (nx, ny, 1)

        # Put Z in point data first
        sg[field_name] = Z.ravel(order="F")

        # Convert to cell data so faces are colored by Z
        sg_cell = sg.point_data_to_cell_data()
        # Optionally rename to a clean cell-data name
        if field_name not in sg_cell.cell_data:
            # After point_data_to_cell_data, the key will typically be the same as field_name
            # but in case pyvista added a suffix, take the only one
            k = list(sg_cell.cell_data.keys())[0]
            sg_cell.cell_data[field_name] = sg_cell.cell_data.pop(k)

        # Plot with visible edges
        p = pv.Plotter()
        p.add_mesh(
            sg_cell,
            scalars=field_name,
            show_edges=True,
            cmap=cmocean.tools.crop(cmocean.cm.topo, 0, 1, pivot=0),
        )
        p.show()