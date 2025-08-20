from dataclasses import dataclass
import numpy as np

from topography import Topography


@dataclass(kw_only=True, slots=True)          # keep slots if you like
class GaussianHill(Topography):
    hill_height: float = 100.0

    @property
    def x0(self) -> float:        
        return 0.5 * (self.xmin + self.xmax)
    @property
    def y0(self) -> float:        
        return 0.5 * (self.ymin + self.ymax)      
    @property
    def sigma(self) -> float:   
        span = min(self.Lx, self.Ly)
        return span / 12
    
    def get_domain_range(self) -> tuple[float, float, float, float]:
        return 0,0,0,0
    
    def sample_z(self, x: float, y: float) -> float:
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2
        return self.hill_height * np.exp(-r2 / (2 * self.sigma**2))

    def array_sample_Z(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized hill: accepts X, Y arrays of any shape and returns Z array.
        """
        # squared distance from hill center
        r2 = (X - self.x0)**2 + (Y - self.y0)**2
        # Gaussian hill
        return self.hill_height * np.exp(-r2 / (2 * self.sigma**2))