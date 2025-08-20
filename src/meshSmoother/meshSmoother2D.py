from mesh import Mesh2D
from typing import Tuple
from pydantic import BaseModel
import numpy as np

class MeshSmoother2D(BaseModel):


    def smooth_step(self, mesh:Mesh2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Perform one smoothing pass, returning new (X, Y, epsilon).
        Must be implemented by subclasses.
        """
        
        raise NotImplementedError("Subclasses must implement this")
    
    def smooth(
        self,
        mesh: Mesh2D,
        tol: float = 1e-6,
        max_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Iteratively apply `smooth_step` until convergence or max_steps reached.

        Args:
            mesh: The Mesh2D instance to smooth in-place.
            tol: Convergence tolerance on the smoothing residual epsilon.
            max_steps: Maximum number of smoothing iterations.

        Returns:
            A tuple (X, Y, epsilon) where X and Y are the smoothed coordinates
            and epsilon is the final residual error.
        """
        epsilon = np.inf
        for step in range(1, max_steps + 1):
            # perform one smoothing iteration
            new_X, new_Y, _, epsilon = self.smooth_step(mesh)
            # update the mesh in-place
            mesh.X = new_X
            mesh.Y = new_Y
            # check for convergence
            if epsilon <= tol:
                print(f"Converged in {step} steps (epsilon={epsilon:.2e}).")
                break
        else:
            print(f"Reached max_steps={max_steps} with epsilon={epsilon:.2e}.")

        return mesh.X, mesh.Y, epsilon