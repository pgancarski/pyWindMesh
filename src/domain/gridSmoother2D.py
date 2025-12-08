from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np

from .grid2d import Grid2D

class GridSmoother2D(ABC):

    @abstractmethod
    def smooth_step(self, grid:Grid2D, relaxation_map: np.ndarray) -> Tuple[Grid2D, float]:
        """
        Perform one smoothing pass, returning corrected grid and the residual error..
        Must be implemented by subclasses.
        """
        ...
    
    def smooth(
        self,
        grid:Grid2D,
        relaxation_map: np.ndarray,
        tol: float = 1e-6,
        max_steps: int = 10
    ) -> Tuple[Grid2D, float]:
        """
        Iteratively apply `smooth_step` until convergence or max_steps reached.

        Args:
            grid: The Grid2D instance to smooth in-place.
            relaxation_factor: 0-no updates 1-full update (no relaxation)
            tol: Convergence tolerance on the smoothing residual epsilon.
            max_steps: Maximum number of smoothing iterations.

        Returns:
            A corrected grid
            and the final residual error.
        """
        epsilon = np.inf
        for step in range(1, max_steps + 1):
            # perform one smoothing iteration
            grid, epsilon = self.smooth_step(grid, relaxation_map)

            # check for convergence
            # print("GridSmoother2D, convergence: "+str(epsilon))
            if epsilon <= tol:
                print(f"Converged in {step} steps (epsilon={epsilon:.2e}).")
                break
        else:
            print(f"Reached max_steps={max_steps} with epsilon={epsilon:.2e}.")

        return grid, epsilon