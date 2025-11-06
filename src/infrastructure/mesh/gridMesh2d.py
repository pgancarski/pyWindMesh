
import numpy as np
from typing import List, Optional

from src.domain import Mesh2D
from src.domain import Grid2D
from src.domain import SpatialValueProvider
from src.domain import GridSmoother2D

from config import Config



# helper to compute angle between a,b: arccos((a·b)/(|a||b|))
def angle(a, b):
    dot = np.sum(a*b, axis=-1)
    na  = np.linalg.norm(a, axis=-1)
    nb  = np.linalg.norm(b, axis=-1)
    # clamp to [-1,1] to avoid NaNs
    cosang = np.clip(dot/(na*nb), -1.0, 1.0)
    return np.arccos(cosang)

class GridMesh2D(Mesh2D):
    """
    Constructs a regular 2D mesh based on a configuration dictionary.

    Attributes:
        X (np.ndarray): 2D array of X-coordinates.
        Y (np.ndarray): 2D array of Y-coordinates.
        Z (np.ndarray): 2D array of Z-coordinates (initialized to zeros).
    """
    def __init__(self, config: Config):
        # Unpack configuration
        xt = config.farm_xt
        xf = config.farm_xf
        yt = config.farm_yt
        yf = config.farm_yf
        dx = config.farm_cellsize_x
        dy = config.farm_cellsize_y

        # Determine number of points along each axis
        nx = int(round((xf - xt) / dx)) + 1
        ny = int(round((yf - yt) / dy)) + 1

        # Generate linspace for each axis
        x_vals = np.linspace(xt, xf, nx)
        y_vals = np.linspace(yt, yf, ny)

        # Create coordinate grids
        self.grid = Grid2D(x_vals, y_vals)

        # Initialize Z to zero
        self.grid.create_point_values("Z", np.zeros_like(self.grid.X))

    def to_ground_grid(self) -> Grid2D:
        return self.grid
    
    def to_ground_points(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self.grid.to_points_vector()

    def get_point(self, ix: int, iy: int) -> tuple[float, float, float]:
        """
        Retrieve the point at mesh indices (ix, iy).

        Args:
            ix (int): Index along the X-direction (0 <= ix < nx).
            iy (int): Index along the Y-direction (0 <= iy < ny).

        Returns:
            Point: The Point object at the specified indices.
        """
        x = float(self.grid.X[ix, iy])
        y = float(self.grid.Y[ix, iy])
        z = float(self.grid.point_values["Z"][ix, iy])
        return x, y, z
    
    def set_point_values(self, name: str, values_provider:SpatialValueProvider):
        values = values_provider.values_at_points(self.grid.X, self.grid.Y)
        self.grid.set_point_values(name, values, create=True)

    def set_face_values(self, name: str, values_provider:SpatialValueProvider):
        values = values_provider.values_at_points(self.grid.X, self.grid.Y)
        self.grid.set_face_values(name, values, create=True)

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the mesh as (nx, ny).
        """
        return self.grid.X.shape
    
    def check_mesh_quality(self):
        max_skewness, mean_skewness = self.mesh_skewness_stats()

        print("The maximum skewness angle [rad]: ",max_skewness)
        print("The mean skewness angle [rad]: ",mean_skewness)
    
    def mesh_skewness_stats(self):
        """
        Given X, Y, Z of shape (n, m), interpreted as a regular quad‐mesh,
        compute for each quad (cell) its skewness = max_i(|angle_i - 90°|)/90°,
        where angle_i are the four interior angles of the quad.
        Returns (max_skewness, avg_skewness), both in [0, 1].
        """
        # pack into points array of shape (n, m, 3)
        P = np.stack((self.grid.X, self.grid.Y, self.grid.point_values["Z"]), axis=-1)

        # grab the four corners of each quad:
        P00 = P[:-1, :-1]   # lower‐left
        P10 = P[1:,  :-1]   # upper‐left
        P11 = P[1:,  1:]    # upper‐right
        P01 = P[:-1, 1:]    # lower‐right

        # for each corner, form the two edges meeting there:
        # at P00: edges to P10 and to P01
        e00a = P10 - P00
        e00b = P01 - P00

        # at P10: edges to P11 and to P00
        e10a = P11 - P10
        e10b = P00 - P10

        # at P11: edges to P01 and to P10
        e11a = P01 - P11
        e11b = P10 - P11

        # at P01: edges to P00 and to P11
        e01a = P00 - P01
        e01b = P11 - P01

        # compute the four angle arrays (shape (n-1, m-1))
        θ00 = angle(e00a, e00b)
        θ10 = angle(e10a, e10b)
        θ11 = angle(e11a, e11b)
        θ01 = angle(e01a, e01b)

        # deviation from 90° = |θ - π/2|
        dev00 = np.abs(θ00 - np.pi/2)
        dev10 = np.abs(θ10 - np.pi/2)
        dev11 = np.abs(θ11 - np.pi/2)
        dev01 = np.abs(θ01 - np.pi/2)

        # per‐face skewness = (max deviation)/(π/2)
        max_dev = np.maximum.reduce([dev00, dev10, dev11, dev01])
        skewness = max_dev / (np.pi/2)

        return skewness.max(), skewness.mean()
        
    def apply_grid_smoother(
            self,
            grid_smoother: GridSmoother2D,
            tol: Optional[float] = None,
            max_steps: Optional[int] = None,
            zone: Optional[List[str]] = None
        ):
        """
        Apply grid smoothing
        """

        # deal with None's
        kwargs = {}
        if tol is not None:
            kwargs["tol"] = tol
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if zone is not None:
            kwargs["zone"] = zone


        new_grid, error = grid_smoother.smooth(self.grid, **kwargs)
        self.grid = new_grid

        return error