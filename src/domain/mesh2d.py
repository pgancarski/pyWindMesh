from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

from config import Config

from .grid2d import Grid2D
from .gridSmoother2D import GridSmoother2D
from .spatialValueProvider import SpatialValueProvider

class Mesh2D(ABC):
    def __init__(self, config: Config) -> None:
        """
        Base constructor for 2D mesh classes.

        Args:
            config (Config): Configuration object containing mesh or simulation parameters.
        """
        self.config = config

    @abstractmethod
    def mesh_skewness_stats(self) -> tuple[float, float]:
        """
        Abstract method returning max and avg skewness.
        """
        ...

    @abstractmethod
    def to_ground_grid(self) -> Grid2D:
        """
        Abstract method returning the mesh projected onto a ground grid.
        """
        ...

    @abstractmethod
    def to_ground_points(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Abstract method returning vectors of X and Y, and a dict of value vectors.
        """
        ...

    @abstractmethod
    def set_point_values(self, name: str, values_provider:SpatialValueProvider):
        """
        Abstract method for setting values for points.
        """
        ...

    @abstractmethod
    def set_face_values(self, name: str, values_provider:SpatialValueProvider):
        """
        Abstract method for setting values for faces.
        """
        ...

    def set_Z(self, topography_provider:SpatialValueProvider):
        """
        Shorthand for setting up Z
        """
        self.set_point_values("Z", topography_provider)

    @abstractmethod
    def apply_grid_smoother(
            self,
            grid_smoother: GridSmoother2D,
            tol: Optional[float] = None,
            max_steps: Optional[int] = None,
            zone: Optional[List[str]] = None
        ):
        """
        Applies GridSmoother2D filter where possible.

        tol - override for convergence tolerance
        max_steps - override for maximum number of steps
        zone - Optional list of mesh zones where the filter will be applied
        """
        ...