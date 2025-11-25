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

    def plot(self, mesh: Mesh2D, field_name: str = "Z") -> None:
        grid = mesh.to_ground_grid()
        self.plot_pyvista(grid, field_name)

    def plot_pyvista(self, grid: Grid2D, field_name: str = "Z") -> None:
        """
        Plot a 2D structured grid using pyvista, with faces colored by the
        given point field and geometry defined by point_values["Z"].

        Parameters
        ----------
        grid : Grid2D
            Instance of Grid2D containing X, Y, and point_values["Z"] for the
            vertical coordinate, plus point_values[field_name] for colouring.
        field_name : str, optional
            Name of the point field to use for colouring, by default "Z".
        """
        X = grid.X
        Y = grid.Y

        # Z is always taken from the "Z" field for geometry
        try:
            Z = grid.point_values["Z"]
        except KeyError as exc:
            raise KeyError('Grid must contain a "Z" field in point_values for elevation.') from exc

        # Scalars for colouring come from field_name (which may or may not be "Z")
        C = grid.point_values[field_name]

        if X.shape != Y.shape or Z.shape != X.shape or C.shape != X.shape:
            raise ValueError(
                f"Shape mismatch: X{X.shape}, Y{Y.shape}, Z{Z.shape}, {field_name}{C.shape}"
            )

        nx, ny = X.shape

        # Build a StructuredGrid: points are (X, Y, Z)
        sg = pv.StructuredGrid()

        # Flatten coordinates
        # Use Fortran order because meshgrid with indexing='ij' is column-major in sense of pyvista
        points = np.column_stack(
            [X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]
        )

        # === APPLY ROTATION TO XY ===
        un_rotate_points(
            points[:, :2],                     # XY slice
            angle_deg=self.wind_direction,
            center_x=self.center_x,
            center_y=self.center_y,
            inplace=True
        )

        # Assign points to grid
        sg.points = points

        # Dimensions are number of points in each direction
        sg.dimensions = (nx, ny, 1)

        # Put colouring field in point data
        sg[field_name] = C.ravel(order="F")

        # Convert to cell data so faces are colored by the chosen field
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
        p.show_axes()
        p.show()
