import numpy as np
from typing import Tuple
from math import acos, radians
from numba import njit, prange

from src.domain import Grid2D
from src.domain import GridSmoother2D

@njit(inline='always')
def compute_angle(p1, pc, p2):
    v1x = p1[0] - pc[0]
    v1y = p1[1] - pc[1]
    v1z = p1[2] - pc[2]
    v2x = p2[0] - pc[0]
    v2y = p2[1] - pc[1]
    v2z = p2[2] - pc[2]

    dot = v1x*v2x + v1y*v2y + v1z*v2z
    n1 = (v1x*v1x + v1y*v1y + v1z*v1z) ** 0.5
    n2 = (v2x*v2x + v2y*v2y + v2z*v2z) ** 0.5

    # manual clamp (Numba-safe)
    val = dot / (n1 * n2)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    return acos(val)

@njit
def pad_edge_2d(Z):
    nrows, ncols = Z.shape
    Zp = np.empty((nrows + 2, ncols + 2), dtype=Z.dtype)

    # center
    Zp[1:-1, 1:-1] = Z

    # top/bottom edges
    Zp[0, 1:-1] = Z[0, :]
    Zp[-1, 1:-1] = Z[-1, :]

    # left/right edges
    Zp[1:-1, 0] = Z[:, 0]
    Zp[1:-1, -1] = Z[:, -1]

    # corners
    Zp[0, 0]     = Z[0, 0]
    Zp[0, -1]    = Z[0, -1]
    Zp[-1, 0]    = Z[-1, 0]
    Zp[-1, -1]   = Z[-1, -1]

    return Zp

@njit
def add_padding(X, Y, Z):
    """
    Extend structured grid (X,Y) outward by one cell width in all directions.
    Z is padded by copying edge values.
    """
    nrows, ncols = X.shape

    # --- pad in Y-direction (columns) ---
    dXdy_start = X[:, 1] - X[:, 0]
    dXdy_end   = X[:, ncols-1] - X[:, ncols-2]
    dYdy_start = Y[:, 1] - Y[:, 0]
    dYdy_end   = Y[:, ncols-1] - Y[:, ncols-2]

    X_left  = (X[:, 0:1] - dXdy_start[:, None])   # slice instead of list
    X_right = (X[:, -1:] + dXdy_end[:, None])
    Y_left  = (Y[:, 0:1] - dYdy_start[:, None])
    Y_right = (Y[:, -1:] + dYdy_end[:, None])

    X = np.hstack((X_left, X, X_right))
    Y = np.hstack((Y_left, Y, Y_right))

    # --- pad in X-direction (rows) ---
    dXdx_start = X[1, :] - X[0, :]
    dXdx_end   = X[-1, :] - X[-2, :]
    dYdx_start = Y[1, :] - Y[0, :]
    dYdx_end   = Y[-1, :] - Y[-2, :]

    X_top = X[0:1, :] - dXdx_start[None, :]
    X_bottom = X[-1:, :] + dXdx_end[None, :]
    Y_top = Y[0:1, :] - dYdx_start[None, :]
    Y_bottom = Y[-1:, :] + dYdx_end[None, :]

    X = np.vstack((X_top, X, X_bottom))
    Y = np.vstack((Y_top, Y, Y_bottom))

    # --- Z: copy edges only ---
    Z = pad_edge_2d(Z)

    return X, Y, Z

@njit(parallel=False)
def smooth_step_inner(X, Y, Z, first_cell_size, max_angle):
    nrows, ncols = Z.shape
    new_Z = Z.copy()

    # target angle at the center (in radians)
    theta_target = radians(180.0 - max_angle)

    # iterate over interior nodes
    for i in prange(1, nrows - 1):
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
            if supp_ns > max_angle:
                # bracket
                z_lo = min(Z[i,j+1], Z[i,j-1]) - first_cell_size
                z_hi = max(Z[i,j+1], Z[i,j-1]) + first_cell_size

                # bisection to solve angle(pN, [x,y,zc], pS) == θ_target
                # def f(zc_trial):
                #    ctr = np.array([*pc_xy, zc_trial])
                #    return compute_angle(pN, ctr, pS) - theta_target
                ctr = np.array([*pc_xy, z_lo])
                f_z_lo = compute_angle(pN, ctr, pS) - theta_target

                ctr = np.array([*pc_xy, z_hi])
                f_z_hi = compute_angle(pN, ctr, pS) - theta_target

                # if bracket doesn't work, fallback to midpoint
                if f_z_lo * f_z_hi > 0:
                    zc = 0.5 * (Z[i,j+1] + Z[i,j-1])
                else:
                    for _ in range(20):
                        zm = 0.5 * (z_lo + z_hi)
                        ctr = np.array([*pc_xy, zm])
                        f_zm = compute_angle(pN, ctr, pS) - theta_target
                        if f_z_lo * f_zm <= 0:
                            z_hi, f_z_hi = zm, f_zm
                        else:
                            z_lo, f_z_lo = zm, f_zm
                    zc = 0.5 * (z_lo + z_hi)

            # save the result
            zNS=zc

            # check E–C–W 
            zc = Z[i,j]
            pc = np.array([*pc_xy, zc])
            θ_ew = compute_angle(pE, pc, pW)
            supp_ew = np.degrees(np.pi - θ_ew)
            if supp_ew > max_angle:
                z_lo = min(Z[i+1,j], Z[i-1,j]) - first_cell_size
                z_hi = max(Z[i+1,j], Z[i-1,j]) + first_cell_size

                # bisection to solve angle(pE, [x,y,zc], pW) == θ_target
                # def g(zc_trial):
                #    ctr = np.array([*pc_xy, zc_trial])
                #    return compute_angle(pE, ctr, pW) - theta_target
                ctr_lo = np.array([*pc_xy, z_lo])
                g_z_lo = compute_angle(pE, ctr_lo, pW) - theta_target

                ctr_hi = np.array([*pc_xy, z_hi])
                g_z_hi = compute_angle(pE, ctr_hi, pW) - theta_target

                if g_z_lo * g_z_hi > 0:
                    zc = 0.5 * (Z[i+1,j] + Z[i-1,j])
                else:
                    for _ in range(20):
                        zm = 0.5 * (z_lo + z_hi)
                        ctr_zm = np.array([*pc_xy, zm])
                        g_zm = compute_angle(pE, ctr_zm, pW) - theta_target
                        if g_z_lo * g_zm <= 0:
                            z_hi, g_z_hi = zm, g_zm
                        else:
                            z_lo, g_z_lo = zm, g_zm

                    zc = 0.5 * (z_lo + z_hi)
            zEW=zc

            # take the average from the two new proposed zc
            # this is to prevent instable, oscilating behaviour from the two proposals when facing sadles/folds in the terrain
            new_Z[i,j] = 0.5 * (zNS + zEW)
    return new_Z

class SurfaceMaxAngleSmoothing(GridSmoother2D):
    max_angle: float=30 #in deegres
    first_cell_size: float=0.0
    smooth_edges: bool=True

    def smooth_step(self, grid:Grid2D,  relaxation_map: np.ndarray) -> Tuple[Grid2D, float]:
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

        new_Z = smooth_step_inner(X,Y,Z,self.first_cell_size,self.max_angle)

        if self.smooth_edges:
            # --- Remove the padded border before relaxation ---
            new_Z = new_Z[1:-1, 1:-1]
            Z = Z[1:-1, 1:-1]
            X = X[1:-1, 1:-1]
            Y = Y[1:-1, 1:-1]

        # apply relaxation factor
        new_Z = (1.0 - relaxation_map) * Z + relaxation_map * new_Z

        # compute max displacement
        error = abs(new_Z - Z).max()

        grid.set_new_XYZ(X, Y, new_Z)

        return grid, error