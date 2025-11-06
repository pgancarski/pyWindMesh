import numpy as np

from domain import SpatialValueProvider

class GaussianHill(SpatialValueProvider):
    def __init__(self, config):
        super().__init__(config)
        
        # Precompute useful parameters
        farm = config
        self.hill_height = 100.0
        self.x0 = 0.5 * (farm.farm_xt + farm.farm_xf)
        self.y0 = 0.5 * (farm.farm_yt + farm.farm_yf)
        
        Lx = farm.farm_xf - farm.farm_xt
        Ly = farm.farm_yf - farm.farm_yt
        self.sigma = min(Lx, Ly) / 12.0
    
    def value_at_point(self, x: float, y: float, _:float) -> float:
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2
        return self.hill_height * np.exp(-r2 / (2 * self.sigma**2))

    def values_at_points(self, X: np.ndarray, Y: np.ndarray, _: np.ndarray|None = None) -> np.ndarray:
        """
        Vectorized hill: accepts X, Y arrays of any shape and returns Z array.
        """
        # squared distance from hill center
        r2 = (X - self.x0)**2 + (Y - self.y0)**2
        # Gaussian hill
        return self.hill_height * np.exp(-r2 / (2 * self.sigma**2))