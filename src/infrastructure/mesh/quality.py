import numpy as np
from typing import Literal

from domain import Grid3D  

def compute_non_orthogonality(
    grid: Grid3D,
    units: Literal["deg", "rad"] = "deg",
    eps: float = 1e-12
) -> np.ndarray:
    """
    Returns an array of shape (nx-1, ny-1, nz-1) with the maximum deviation
    from 90° between the three principal edge vectors of each cell.
    """
    X, Y, Z = grid.X, grid.Y, grid.Z

    ex = np.stack([
        X[1:, :-1, :-1] - X[:-1, :-1, :-1],
        Y[1:, :-1, :-1] - Y[:-1, :-1, :-1],
        Z[1:, :-1, :-1] - Z[:-1, :-1, :-1],
    ], axis=-1)

    ey = np.stack([
        X[:-1, 1:, :-1] - X[:-1, :-1, :-1],
        Y[:-1, 1:, :-1] - Y[:-1, :-1, :-1],
        Z[:-1, 1:, :-1] - Z[:-1, :-1, :-1],
    ], axis=-1)

    ez = np.stack([
        X[:-1, :-1, 1:] - X[:-1, :-1, :-1],
        Y[:-1, :-1, 1:] - Y[:-1, :-1, :-1],
        Z[:-1, :-1, 1:] - Z[:-1, :-1, :-1],
    ], axis=-1)

    def angle(a, b):
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos_theta = np.clip(dot / (np.maximum(na * nb, eps)), -1.0, 1.0)
        return np.arccos(cos_theta)

    angles = np.stack([
        angle(ex, ey),
        angle(ey, ez),
        angle(ex, ez),
    ], axis=-1)

    ortho = np.abs(np.pi / 2 - angles)  # radians

    max_dev = np.max(ortho, axis=-1)

    if units == "deg":
        max_dev = np.degrees(max_dev)

    return max_dev