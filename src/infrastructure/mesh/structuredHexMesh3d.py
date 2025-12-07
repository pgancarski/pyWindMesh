import numpy as np
from numba import njit

from src.config import Config

from src.domain import Mesh2D
from src.domain import Grid3D

from .gridMesh2d import GridMesh2D
from .verticalLayersDistributor import VerticalLayersDistributor

class StructuredHexMesh3d():
    def __init__(self, config: Config, ground_mesh: Mesh2D):
        # test if groundMesh is of type GridMesh2D(Mesh2D)
        if not isinstance(ground_mesh, GridMesh2D):
            raise TypeError(
                f"ground_mesh must be of type GridMesh2D, got {type(ground_mesh)}"
            )

        self.config = config
        inlet_Z = ground_mesh.get_point(0,0)[2]
        self.vertical_distributor = VerticalLayersDistributor(config.vertical_distribution, inlet_Z)
        self.ground_mesh = ground_mesh
        self.ground_grid = ground_mesh.to_ground_grid()

        nx, ny = self.ground_grid.X.shape
        nz = self.vertical_distributor.mesh_size_z

        # Create Grid3D with sizes only, coordinates are zeros initially
        self.mesh = Grid3D(nx, ny, nz)

        # Figure out the shape and size of the mesh
        self.mesh_shape = (nx, ny, nz)
        self.mesh_size = nx * ny * nz

    def build_3d_mesh(self):
        X2D = self.ground_grid.X  # shape (nx, ny)
        Y2D = self.ground_grid.Y
        Zground = self.ground_grid.get_point_values("Z")  # base elevation (DEM or topography)

        nx, ny, nz = self.mesh_shape

        # Expand X and Y into 3D
        self.mesh.X = np.repeat(X2D[:, :, np.newaxis], nz, axis=2)
        self.mesh.Y = np.repeat(Y2D[:, :, np.newaxis], nz, axis=2)

        # Fill Z column-by-column (ground varying)
        for i in range(nx):
            for j in range(ny):
                ground_z = Zground[i, j]
                column_Z = self.vertical_distributor.column_distribution(ground_z)
                self.mesh.Z[i, j, :] = column_Z

        # Curve the first layers (limit so we never exceed nz-1)
        n_curved_out_layers = int(nz/2)
        self.make_curved_out_layers(n_curved_out_layers)

        n_curved_in_layers = nz-self.vertical_distributor.n_const_top
        self.make_curved_in_layers(n_curved_out_layers,n_curved_in_layers)

        self.elliptic_smoothing()


    def make_curved_out_layers(self, iz_n):
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        _, _, nz = Z.shape

        n_layers = min(iz_n, nz - 1)
        if n_layers <= 0:
            return

        # --- 1. Base-layer normals (True surface normals) ---
        dx_di, dx_dj = np.gradient(X[:, :, 0])
        dy_di, dy_dj = np.gradient(Y[:, :, 0])
        dz_di, dz_dj = np.gradient(Z[:, :, 0])

        Nx = dy_di * dz_dj - dz_di * dy_dj
        Ny = dz_di * dx_dj - dx_di * dz_dj
        Nz = dx_di * dy_dj - dy_di * dx_dj

        mag = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        mag[mag == 0.0] = 1.0
        Nx /= mag
        Ny /= mag
        Nz /= mag

        mask = Nz < 0.0
        Nx[mask] *= -1.0
        Ny[mask] *= -1.0
        Nz[mask] *= -1.0

        # --- 2. Reference column data ---
        Zref = Z[0, 0, :]
        domain_top = Zref[-1]
        eps = 1e-12

        for iz in range(n_layers):
            ref_z = Zref[iz]
            ref_next_z = Zref[iz + 1]
            ref_dz = ref_next_z - ref_z
            denom = domain_top - ref_z
            if abs(denom) < eps:
                denom = eps

            # --- 3. Direction vector for this layer ---
            if iz == 0:
                Vx, Vy, Vz = Nx.copy(), Ny.copy(), Nz.copy()
            else:
                dx = X[:, :, iz] - X[:, :, iz - 1]
                dy = Y[:, :, iz] - Y[:, :, iz - 1]
                dz = Z[:, :, iz] - Z[:, :, iz - 1]

                vec_mag = np.sqrt(dx**2 + dy**2 + dz**2)
                vec_mag[vec_mag == 0.0] = 1.0

                Vx = dx / vec_mag
                Vy = dy / vec_mag
                Vz = dz / vec_mag

            mask = Vz < 0.0
            Vx[mask] *= -1.0
            Vy[mask] *= -1.0
            Vz[mask] *= -1.0

            # --- 4. Blend direction toward vertical (matches C++ logic) ---
            alpha = (n_layers - iz) / float(n_layers)
            Vx = alpha * Vx
            Vy = alpha * Vy
            Vz = alpha * (Vz - 1.0) + 1.0

            # --- 5. Scale the vector using the reference spacing rule ---
            remaining_height = np.clip(domain_top - Z[:, :, iz], 0.0, None)
            length = ref_dz * (remaining_height / denom)

            disp_x = Vx * length
            disp_y = Vy * length
            disp_z = Vz * length

            X[:, :, iz + 1] = X[:, :, iz] + disp_x
            Y[:, :, iz + 1] = Y[:, :, iz] + disp_y
            Z[:, :, iz + 1] = Z[:, :, iz] + disp_z

    def make_curved_in_layers(self, iz0, izn):
        """
        Gradually recover (flatten) layers between iz0 and izn, pulling them back
        toward the reference column Z and the original top-plane X/Y.
        """
        nx, ny, nz = self.mesh.X.shape

        if not (0 < iz0 < izn < nz):
            raise ValueError("Require 0 < iz0 < izn < nz.")

        for ix in range(1, nx - 1):
            for iy in range(1, ny - 1):
                self._make_curved_in_column(ix, iy, iz0, izn)


    def _make_curved_in_column(self, ix, iy, iz0, izn):
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        refZ = Z[0, 0, :]
        refX = X[ix, iy, -1]
        refY = Y[ix, iy, -1]

        target_top = refZ[izn]
        prev_ref_z = refZ[iz0 - 1]
        denom_top = target_top - prev_ref_z
        eps = 1e-12
        if abs(denom_top) < eps:
            denom_top = eps

        bottom_layer = iz0 - 1
        bottomZ = Z[ix, iy, bottom_layer]
        bottomX = X[ix, iy, bottom_layer]
        bottomY = Y[ix, iy, bottom_layer]
        denom_alpha = target_top - bottomZ
        if abs(denom_alpha) < eps:
            denom_alpha = eps

        for iz in range(iz0, izn):
            prevZ = Z[ix, iy, iz - 1]
            prevRefZ = refZ[iz - 1]
            currRefZ = refZ[iz]
            refStep = currRefZ - prevRefZ

            denom = target_top - prevRefZ
            if abs(denom) < eps:
                denom = eps

            length = refStep * ((target_top - prevZ) / denom)
            z = prevZ + length
            Z[ix, iy, iz] = z

            alpha = (z - bottomZ) / denom_alpha
            alpha = float(np.clip(alpha, 0.0, 1.0))

            X[ix, iy, iz] = alpha * refX + (1.0 - alpha) * bottomX
            Y[ix, iy, iz] = alpha * refY + (1.0 - alpha) * bottomY

    def elliptic_smoothing(self,
                           start_layer: int = 3,
                           blend_layers: int = 5,
                           max_iters: int = 10,
                           tol: float = 1e-3):
        """
        Smooths interior layers (>= start_layer) by solving the homogeneous
        Laplace equation with Gauss–Seidel relaxation, then blends the result
        back toward the original mesh near the bottom to preserve boundaries.

        Parameters
        ----------
        start_layer : int
            First layer allowed to move (>= 1 keeps the ground plane fixed).
        blend_layers : int
            Number of layers above `start_layer` that get blended with the
            original mesh to avoid sharp transitions.
        max_iters : int
            Maximum Gauss–Seidel sweeps.
        tol : float
            Stop once max displacement per sweep falls below this threshold.
        """
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        nx, ny, nz = X.shape

        start_layer = int(np.clip(start_layer, 1, nz - 2))
        if start_layer >= nz - 1:
            return  # nothing to smooth

        blend_layers = max(0, blend_layers)
        max_blend_layer = min(nz - 1, start_layer + blend_layers + 1)

        original_X = X.copy()
        original_Y = Y.copy()
        original_Z = Z.copy()

        for _ in range(max_iters):
            sweep_err = self._gauss_seidel_sweep(start_layer)
            if sweep_err < tol:
                break

        self._blend_layers(original_X, original_Y, original_Z,
                           start_layer, max_blend_layer)

    @staticmethod
    @njit(cache=True)
    def _gauss_seidel_core(X, Y, Z, valid_cols, start_layer):
        _, _, nz = X.shape
        max_disp = 0.0

        for k in range(start_layer, nz - 1):  # keep the top plane fixed
            km = k - 1
            kp = k + 1

            for idx in range(valid_cols.shape[0]):
                i = valid_cols[idx, 0]
                j = valid_cols[idx, 1]

                im = i - 1
                ip = i + 1
                jm = j - 1
                jp = j + 1

                new_x = (X[ip, j, k] + X[im, j, k] +
                         X[i, jp, k] + X[i, jm, k] +
                         X[i, j, kp] + X[i, j, km]) / 6.0
                new_y = (Y[ip, j, k] + Y[im, j, k] +
                         Y[i, jp, k] + Y[i, jm, k] +
                         Y[i, j, kp] + Y[i, j, km]) / 6.0
                new_z = (Z[ip, j, k] + Z[im, j, k] +
                         Z[i, jp, k] + Z[i, jm, k] +
                         Z[i, j, kp] + Z[i, j, km]) / 6.0

                dx = new_x - X[i, j, k]
                dy = new_y - Y[i, j, k]
                dz = new_z - Z[i, j, k]
                disp = (dx*dx + dy*dy + dz*dz) ** 0.5
                if disp > max_disp:
                    max_disp = disp

                X[i, j, k] = new_x
                Y[i, j, k] = new_y
                Z[i, j, k] = new_z

        return max_disp

    def _gauss_seidel_sweep(self, start_layer: int) -> float:
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        zone_mask = self.ground_mesh.get_point_values("zone_id")

        good_zones = {
            self.ground_mesh.get_zone_code("FARM"),
            self.ground_mesh.get_zone_code("TRANSITION"),
        }
        good_zone_mask = np.isin(zone_mask, list(good_zones))

        # only interior columns can be relaxed
        interior_mask = np.zeros_like(good_zone_mask, dtype=bool)
        interior_mask[1:-1, 1:-1] = True
        valid_cols = np.argwhere(good_zone_mask & interior_mask)

        if valid_cols.size == 0:
            return 0.0

        # ensure int32 for numba friendliness
        valid_cols = valid_cols.astype(np.int32)

        return self._gauss_seidel_core(X, Y, Z, valid_cols, start_layer)
    
    
    def _blend_layers(self,
                      original_X,
                      original_Y,
                      original_Z,
                      start_layer: int,
                      max_blend_layer: int):
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        total_layers = X.shape[2]

        buffer_blending = self.ground_mesh.get_point_values("buffer_blending")
        if buffer_blending is None:
            buffer_blending = np.zeros_like(X[..., 0], dtype=X.dtype)

        # Interior (matching the existing 1:-1 trimming)
        interior_slice = np.s_[1:-1, 1:-1, :]
        buffer_core = np.clip(buffer_blending[1:-1, 1:-1], 0.0, 1.0)
        buffer_full = buffer_core[..., None]  # broadcast across layers

        # Build the per-layer alpha profile (default = 1 ⇒ “keep new”)
        alpha_profile = np.ones(total_layers, dtype=X.dtype)

        # Below the start layer we want original values (alpha -> 0)
        lower_bound = max(0, min(start_layer, total_layers))
        alpha_profile[:lower_bound] = 0.0

        # Blend window [start_layer, max_blend_layer)
        capped_max = max(lower_bound, min(max_blend_layer, total_layers))
        blend_span = max(1, max_blend_layer - start_layer)
        blend_count = max(0, capped_max - lower_bound)
        if blend_count > 0:
            offsets = np.arange(1, blend_count + 1, dtype=X.dtype)
            alpha_profile[lower_bound:capped_max] = np.sqrt(offsets / blend_span)

        alpha_3d = alpha_profile[np.newaxis, np.newaxis, :]                 # shape (1,1,Nz)
        orig_weight = buffer_full + (1.0 - buffer_full) * (1.0 - alpha_3d)  # prefer originals
        new_weight = 1.0 - orig_weight

        # Snapshot current (new) mesh values so we don’t mix partially blended data
        new_X = X[interior_slice].copy()
        new_Y = Y[interior_slice].copy()
        new_Z = Z[interior_slice].copy()

        orig_X = original_X[interior_slice]
        orig_Y = original_Y[interior_slice]
        orig_Z = original_Z[interior_slice]

        X[interior_slice] = orig_weight * orig_X + new_weight * new_X
        Y[interior_slice] = orig_weight * orig_Y + new_weight * new_Y
        Z[interior_slice] = orig_weight * orig_Z + new_weight * new_Z