import numpy as np
from typing import Tuple
from math import acos, radians

from src.domain import Grid2D
from src.domain import GridSmoother2D


def compute_angle(p1, pc, p2):
    v1 = p1 - pc
    v2 = p2 - pc
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    # guard against roundoff
    cosv = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return acos(cosv)

def add_padding(X,Y,Z):
    """
    Extend structured grid (X,Y) outward by one cell width in all directions.
    Z is padded by copying edge values.
    """
    # --- pad in Y-direction (columns) ---
    dXdy_start = X[:, 1] - X[:, 0]
    dXdy_end   = X[:, -1] - X[:, -2]
    dYdy_start = Y[:, 1] - Y[:, 0]
    dYdy_end   = Y[:, -1] - Y[:, -2]

    X = np.hstack([
        (X[:, [0]] - dXdy_start[:, None]),   # extrapolate one col to the left
        X,
        (X[:, [-1]] + dXdy_end[:, None])     # extrapolate one col to the right
    ])
    Y = np.hstack([
        (Y[:, [0]] - dYdy_start[:, None]),
        Y,
        (Y[:, [-1]] + dYdy_end[:, None])
    ])

    # --- pad in X-direction (rows) ---
    dXdx_start = X[1, :] - X[0, :]
    dXdx_end   = X[-1, :] - X[-2, :]
    dYdx_start = Y[1, :] - Y[0, :]
    dYdx_end   = Y[-1, :] - Y[-2, :]

    X = np.vstack([
        X[[0], :] - dXdx_start,
        X,
        X[[-1], :] + dXdx_end
    ])
    Y = np.vstack([
        Y[[0], :] - dYdx_start,
        Y,
        Y[[-1], :] + dYdx_end
    ])

    # --- Z: copy edges only ---
    Z = np.pad(Z, 1, mode='edge')

    return X, Y, Z

class SurfaceMaxAngleSmoothing(GridSmoother2D):
    max_angle: float=30 #in deegres
    first_cell_size: float=0.0
    smooth_edges: bool=True

    def smooth_step(self, grid:Grid2D, relaxation_factor:float) -> Tuple[Grid2D, float]:
        X = grid.X.copy()
        Y = grid.Y.copy()
        Z = grid.point_values["Z"].copy()

        if self.smooth_edges:
            X,Y,Z = add_padding(X,Y,Z)

        # set up reference cell size on first call
        if self.first_cell_size == 0.0:
            # assume uniform grid spacing
            dx = np.hypot(X[1,0] - X[0,0], Y[1,0] - Y[0,0])
            dy = np.hypot(X[0,1] - X[0,0], Y[0,1] - Y[0,0])
            self.first_cell_size = max(dx, dy)

        nrows, ncols = Z.shape
        new_Z = Z.copy()

        # target angle at the center (in radians)
        theta_target = radians(180.0 - self.max_angle)

        # iterate over interior nodes
        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                pc_xy = np.array([X[i,j], Y[i,j]])

                # collect neighbor positions
                pN = np.array([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                pS = np.array([X[i, j-1], Y[i, j-1], Z[i, j-1]])
                pE = np.array([X[i+1, j], Y[i+1, j], Z[i+1, j]])
                pW = np.array([X[i-1, j], Y[i-1, j], Z[i-1, j]])

                zc = Z[i,j]

                # check N–C–S
                zc = Z[i,j]
                pc = np.array([*pc_xy, zc])
                θ_ns = compute_angle(pN, pc, pS)
                supp_ns = np.degrees(np.pi - θ_ns)
                if supp_ns > self.max_angle:
                    # bisection to solve angle(pN, [x,y,zc], pS) == θ_target
                    def f(zc_trial):
                        ctr = np.array([*pc_xy, zc_trial])
                        return compute_angle(pN, ctr, pS) - theta_target
                    # bracket
                    z_lo = min(Z[i,j+1], Z[i,j-1]) - self.first_cell_size
                    z_hi = max(Z[i,j+1], Z[i,j-1]) + self.first_cell_size
                    # if bracket doesn't work, fallback to midpoint
                    if f(z_lo) * f(z_hi) > 0:
                        zc = 0.5 * (Z[i,j+1] + Z[i,j-1])
                    else:
                        for _ in range(20):
                            zm = 0.5 * (z_lo + z_hi)
                            if f(z_lo) * f(zm) <= 0:
                                z_hi = zm
                            else:
                                z_lo = zm
                        zc = 0.5 * (z_lo + z_hi)

                # save the result
                zNS=zc

                # check E–C–W 
                zc = Z[i,j]
                pc = np.array([*pc_xy, zc])
                θ_ew = compute_angle(pE, pc, pW)
                supp_ew = np.degrees(np.pi - θ_ew)
                if supp_ew > self.max_angle:
                    # bisection to solve angle(pE, [x,y,zc], pW) == θ_target
                    def g(zc_trial):
                        ctr = np.array([*pc_xy, zc_trial])
                        return compute_angle(pE, ctr, pW) - theta_target
                    z_lo = min(Z[i+1,j], Z[i-1,j]) - self.first_cell_size
                    z_hi = max(Z[i+1,j], Z[i-1,j]) + self.first_cell_size
                    if g(z_lo) * g(z_hi) > 0:
                        zc = 0.5 * (Z[i+1,j] + Z[i-1,j])
                    else:
                        for _ in range(20):
                            zm = 0.5 * (z_lo + z_hi)
                            if g(z_lo) * g(zm) <= 0:
                                z_hi = zm
                            else:
                                z_lo = zm
                        zc = 0.5 * (z_lo + z_hi)
                zEW=zc

                # take the average from the two new proposed zc
                # this is to prevent instable, oscilating behaviour from the two proposals when facing sadles/folds in the terrain
                new_Z[i,j] = 0.5 * (zNS + zEW)

        if self.smooth_edges:
            # --- Remove the padded border before relaxation ---
            new_Z = new_Z[1:-1, 1:-1]
            Z = Z[1:-1, 1:-1]
            X = X[1:-1, 1:-1]
            Y = Y[1:-1, 1:-1]

        # apply relaxation factor
        new_Z = (1-relaxation_factor) * Z + relaxation_factor * new_Z

        # compute max displacement
        error = abs(new_Z - Z).max()

        grid.set_new_XYZ(X, Y, new_Z)

        return grid, error