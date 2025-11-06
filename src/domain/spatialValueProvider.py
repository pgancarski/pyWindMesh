from abc import ABC, abstractmethod
import numpy as np

from config import Config

class SpatialValueProvider(ABC):
    """Abstract interface for computing spatially varying values (e.g. elevation, roughness)."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def value_at_point(self, x: float, y: float, z: float = 0.0) -> float:
        """Compute the value at a single spatial point."""
        ...

    @abstractmethod
    def values_at_points(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray|None = None) -> np.ndarray:
        """Vectorized version for multiple points."""
        ...
