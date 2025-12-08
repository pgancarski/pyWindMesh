import numpy as np
from typing import Tuple

from src.domain import Grid2D
from src.domain import GridSmoother2D

class SurfaceSquerifyFaces(GridSmoother2D):
    #max_error: float = 1e-8
    first_cell_size: float=0

    def __init__(self):
        super().__init__()

        print("WARNING: SurfaceSquerifyFaces filter is unstable and not recomended to use, work in progress")

    def smooth_step(self, grid:Grid2D, relaxation_map: np.ndarray) -> Tuple[Grid2D, float]:
        """
        Improves surface mesh quality without flattening the topography.

        The idea is to relax the angles of the faces by making each point centered in reference to its neighbours.
        The twist is that the new point should maintain the curvature of the terrain.

        To do that we build a plane based on the neighbours, we find out how high above the plane is the old point,
        and then we move it to the center while maintaining the height above the plane.
        """

        # grab current coords
        X = grid.X.copy()
        Y = grid.Y.copy()
        Z = grid.point_values["Z"].copy()

        # if unset, setup the reference cell size
        if self.first_cell_size == 0:
            dx = X[1, 0] - X[0, 0]
            dy = Y[0, 1] - Y[0, 0]
            self.first_cell_size = max(dx, dy)

        # interior slices
        Xc = X[1:-1, 1:-1]
        Yc = Y[1:-1, 1:-1]
        Zc = Z[1:-1, 1:-1]

        # neighbor slices
        Xw, Xe = X[0:-2, 1:-1], X[2:, 1:-1]
        Xs, Xn = X[1:-1, 0:-2], X[1:-1, 2:]
        Yw, Ye = Y[0:-2, 1:-1], Y[2:, 1:-1]
        Ys, Yn = Y[1:-1, 0:-2], Y[1:-1, 2:]
        Zw, Ze = Z[0:-2, 1:-1], Z[2:, 1:-1]
        Zs, Zn = Z[1:-1, 0:-2], Z[1:-1, 2:]

        # build plane normal from two vectors in neighbor patch
        v1x, v1y, v1z = Xs - Xe, Ys - Ye, Zs - Ze
        v2x = (Xn + Xw) * 0.5 - Xs
        v2y = (Yn + Yw) * 0.5 - Ys
        v2z = (Zn + Zw) * 0.5 - Zs

        # plane coefficients a, b, c
        a = v1y * v2z - v2y * v1z
        b = v1z * v2x - v2z * v1x
        c = v1x * v2y - v2x * v1y
        # plane constant d passing through old center
        d = a * Xc + b * Yc + c * Zc

        # centroid of neighbors
        px = (Xs + Xn + Xw + Xe) * 0.25
        py = (Ys + Yn + Yw + Ye) * 0.25
        pz = (Zs + Zn + Zw + Ze) * 0.25

        # project centroid onto plane preserving height above it
        denom = a * a + b * b + c * c
        alpha = np.zeros_like(px)
        valid = denom > 1e-12
        alpha[valid] = (a[valid] * px[valid] + b[valid] * py[valid] + c[valid] * pz[valid] - d[valid]) / denom[valid]

        new_Xc = px - a * alpha
        new_Yc = py - b * alpha
        new_Zc = pz - c * alpha

        # clamp X between east/west neighbors to avoid folding
        xmin = np.minimum(Xw, Xe)
        xmax = np.maximum(Xw, Xe)
        new_Xc = np.clip(new_Xc, xmin + (1e-8), xmax-(1e-8))
        # clamp Y between south/north neighbors to avoid folding
        ymin = np.minimum(Ys, Yn)
        ymax = np.maximum(Ys, Yn)
        new_Yc = np.clip(new_Yc, ymin+(1e-8), ymax-(1e-8))

        # assemble new arrays
        new_X = X.copy()
        new_Y = Y.copy()
        new_Z = Z.copy()
        new_X[1:-1, 1:-1] = new_Xc
        new_Y[1:-1, 1:-1] = new_Yc
        new_Z[1:-1, 1:-1] = new_Zc

        # apply relaxation factor

        new_X = (1.0 - relaxation_map) * X + relaxation_map * new_X
        new_Y = (1.0 - relaxation_map) * Y + relaxation_map * new_Y
        new_Z = (1.0 - relaxation_map) * Z + relaxation_map * new_Z

        # compute maximum displacement error
        error = max(
            np.abs(new_X - X).max(),
            np.abs(new_Y - Y).max(),
            np.abs(new_Z - Z).max()
        )

        grid.set_new_XYZ(new_X, new_Y, new_Z)

        return grid, error
