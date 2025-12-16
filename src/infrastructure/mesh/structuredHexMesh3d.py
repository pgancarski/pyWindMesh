import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter

from src.config import Config

from src.domain import Mesh2D
from src.domain import Grid3D

from .gridMesh2d import GridMesh2D
from .verticalLayersDistributor import VerticalLayersDistributor
from .quality import compute_non_orthogonality


def _normalize(vec, eps=1e-12):
    mag = np.linalg.norm(vec, axis=-1, keepdims=True)
    mag = np.maximum(mag, eps)
    return vec / mag


def _smooth_normals(nx, ny, nz, passes=2):
    for _ in range(passes):
        nx = uniform_filter(nx, size=3, mode="nearest")
        ny = uniform_filter(ny, size=3, mode="nearest")
        nz = uniform_filter(nz, size=3, mode="nearest")
    return nx, ny, nz


class StructuredHexMesh3d():
    def __init__(self, config: Config, ground_mesh: Mesh2D):
        # test if groundMesh is of type GridMesh2D(Mesh2D)
        if not isinstance(ground_mesh, GridMesh2D):
            raise TypeError(
                f"ground_mesh must be of type GridMesh2D, got {type(ground_mesh)}"
            )

        self.config = config
        inlet_Z = ground_mesh.get_point(0, 0)[2]
        self.vertical_distributor = VerticalLayersDistributor(config.vertical_distribution, inlet_Z)
        self.ground_mesh = ground_mesh
        self.ground_grid = ground_mesh.to_ground_grid()
        self.top_grid = ground_mesh.get_reference_grid()

        nx, ny = self.ground_grid.X.shape
        nz = self.vertical_distributor.mesh_size_z

        # Create Grid3D with sizes only, coordinates are zeros initially
        self.mesh = Grid3D(nx, ny, nz)

        # Figure out the shape and size of the mesh
        self.mesh_shape = (nx, ny, nz)
        self.mesh_size = nx * ny * nz

    def build_3d_mesh(self):

        Zground = self.ground_grid.get_point_values("Z")  # base elevation (DEM or topography)

        nx, ny, nz = self.mesh_shape

        # Expand X and Y into 3D
        # First layers from the ground adapted grid
        X = self.ground_grid.X
        Y = self.ground_grid.Y
        n = nz - self.vertical_distributor.n_const_top
        primary_X = np.repeat(X[:, :, np.newaxis], n, axis=2)
        primary_Y = np.repeat(Y[:, :, np.newaxis], n, axis=2)

        # the rest of layers from the reference grid - distribution not affected by the terrain
        X = self.top_grid.X
        Y = self.top_grid.Y
        n = self.vertical_distributor.n_const_top
        ref_X = np.repeat(X[:, :, np.newaxis], n, axis=2)
        ref_Y = np.repeat(Y[:, :, np.newaxis], n, axis=2)

        # Stack them along the third axis
        self.mesh.X = np.concatenate((primary_X, ref_X), axis=2)
        self.mesh.Y = np.concatenate((primary_Y, ref_Y), axis=2)

        # Fill Z column-by-column (ground varying)
        for i in range(nx):
            for j in range(ny):
                ground_z = Zground[i, j]
                column_Z = self.vertical_distributor.column_distribution(ground_z)
                self.mesh.Z[i, j, :] = column_Z

        # Curve the first layers (limit so we never exceed nz-1)
        n_curved_out_layers = int(nz * 2/3)
        self.make_curved_out_layers(n_curved_out_layers)

        last_curved_in_layer = nz - self.vertical_distributor.n_const_top
        self.make_curved_in_layers(n_curved_out_layers, last_curved_in_layer)

        self.elliptic_smoothing(min_protected_height= 1.1 * self.config.vertical_distribution.first_cell_size)

        non_orth = compute_non_orthogonality(self.mesh, units="deg")
        self.mesh.set_cell_values("non_orthogonality_deg", non_orth, create=True)

    def make_curved_out_layers(self,
                               iz_n,
                               min_vertical_component=0.15,
                               normal_smooth_passes=2,
                               lateral_limit=0.45,
                               eps=1e-9,
                               mid_relax=0.6,
                               curve_profile_power=2.0,
                               tangential_soft_cap=0.6):
        # layers speedup, makes the curveout more aggressive earlier,
        # lowers the quality at the ground level but makes the mesh easier later on
        # mid_relax make an aditional "bump" later on, both variables work togather and should be changed togather
        n_skipped_layers = 2

        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        _, _, nz = Z.shape

        n_layers = min(iz_n, nz - 1)
        if n_layers <= 0:
            return

        # --- 1. Base-layer normals (smoothed & upward) ---
        dx_di, dx_dj = np.gradient(X[:, :, 0])
        dy_di, dy_dj = np.gradient(Y[:, :, 0])
        dz_di, dz_dj = np.gradient(Z[:, :, 0])

        Nx = dy_di * dz_dj - dz_di * dy_dj
        Ny = dz_di * dx_dj - dx_di * dz_dj
        Nz = dx_di * dy_dj - dy_di * dx_dj
        Nx, Ny, Nz = _smooth_normals(Nx, Ny, Nz, passes=normal_smooth_passes)

        base_normals = np.stack([Nx, Ny, Nz], axis=-1)
        base_normals = _normalize(base_normals)
        downward = base_normals[..., 2] < 0.0
        base_normals[downward] *= -1.0
        base_normals[..., 2] = np.maximum(base_normals[..., 2], min_vertical_component)
        base_normals = _normalize(base_normals)

        # --- 2. Reference column geometry (single target) ---
        Zref = Z[0, 0, :]
        domain_top = Zref[-1]
        eps_denom = 1e-12

        profile = np.linspace(0.0, 1.0, n_layers, endpoint=False)
        curve_profile_power = max(curve_profile_power, 1.0)
        tangential_soft_cap = np.clip(tangential_soft_cap, 0.0, 1.0)
        mid_relax = np.clip(mid_relax, 0.0, 1.0)

        for iz, t in enumerate(profile):
            ref_z = Zref[iz]
            ref_next_z = Zref[iz + 1]
            ref_dz = ref_next_z - ref_z
            denom = domain_top - ref_z
            if abs(denom) < eps_denom:
                denom = np.sign(denom) * eps_denom or eps_denom

            # --- 3. Direction field ---
            if iz == 0:
                direction = base_normals.copy()
            else:
                dx = X[:, :, iz] - X[:, :, iz - 1]
                dy = Y[:, :, iz] - Y[:, :, iz - 1]
                dz = Z[:, :, iz] - Z[:, :, iz - 1]
                direction = np.stack([dx, dy, dz], axis=-1)
                direction = _normalize(direction)
                needs_reset = direction[..., 2] < min_vertical_component
                direction[needs_reset] = base_normals[needs_reset]

            alpha = (n_layers - iz - n_skipped_layers) / float(n_layers + n_skipped_layers)
            alpha = np.clip(alpha, 0.0, 1.0)
            blended = np.empty_like(direction)
            blended[..., 0] = alpha * direction[..., 0]
            blended[..., 1] = alpha * direction[..., 1]
            blended[..., 2] = alpha * direction[..., 2] + (1.0 - alpha)

            smooth = t**curve_profile_power
            smooth /= (smooth + (1.0 - t)**curve_profile_power + eps)
            mid_weight = 4.0 * smooth * (1.0 - smooth)  # bell curve

            vertical_bias = 1.0 - mid_relax * mid_weight
            vertical_bias = np.clip(vertical_bias, 0.2, 1.0)

            blended[..., 2] *= vertical_bias
            blended[..., 2] = np.maximum(blended[..., 2], min_vertical_component)
            direction = _normalize(blended)

            # --- 4. Single-reference scaling ---
            remaining_height = domain_top - Z[:, :, iz]
            remaining_height = np.clip(remaining_height, eps, None)
            length = ref_dz * (remaining_height / denom)

            disp = direction * length[..., None]

            # Tangential soft cap: reduce lateral movement when columns stay vertical
            tangential_scale = tangential_soft_cap + (1.0 - tangential_soft_cap) * direction[..., 2]
            tangential_scale = np.clip(tangential_scale, tangential_soft_cap, 1.0)
            disp[..., 0] *= tangential_scale
            disp[..., 1] *= tangential_scale

            # --- 5. Lateral drift limiter ---
            lateral_mag = np.linalg.norm(disp[..., :2], axis=-1)
            grad_x = np.maximum(np.abs(np.gradient(X[:, :, iz])[0]), eps)
            grad_y = np.maximum(np.abs(np.gradient(Y[:, :, iz])[1]), eps)
            local_width = np.maximum(grad_x, grad_y)
            max_lateral = lateral_limit * local_width
            scale = np.minimum(1.0, max_lateral / np.maximum(lateral_mag, eps))
            disp[..., 0] *= scale
            disp[..., 1] *= scale

            # --- 6. Update layers ---
            X[:, :, iz + 1] = X[:, :, iz] + disp[..., 0]
            Y[:, :, iz + 1] = Y[:, :, iz] + disp[..., 1]
            Z[:, :, iz + 1] = Z[:, :, iz] + disp[..., 2]

            # --- 7. Monotonic guard ---
            Z[:, :, :iz + 2] = np.maximum.accumulate(Z[:, :, :iz + 2], axis=2)

    def make_curved_in_layers(self, iz0, izn):
        """
        Gradually recover (flatten) layers between iz0 and izn, pulling them back
        toward the reference column Z and the original top-plane X/Y.
        """
        nx, ny, nz = self.mesh.X.shape

        if not (0 < iz0 < izn < nz):
            raise ValueError("Require 0 < iz0 < izn < nz.")

        for ix in range(0, nx):
            for iy in range(0, ny):
                self._make_curved_in_column(ix, iy, iz0, izn)

    def _make_curved_in_column(self, ix, iy, iz0, izn):
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        refZ = Z[0, 0, :]
        refX = self.top_grid.X[ix, iy]
        refY = self.top_grid.Y[ix, iy]

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
                           start_layer: int = 2,
                           blend_layers: int = 10,
                           max_iters: int = 5,
                           tol: float = 1e-3,
                           protected_factor: float = 1.25,
                           min_protected_height: float = 0.5,
                           near_ground_relax: float = 0.65,
                           omega: float = 0.85,
                           redistribute_after: bool = True,
                           redistribute_keep_layers: int = 2):
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
        protected_factor : float
            Multiplier applied to the first cell height to define a protected
            depth where smoothing is disabled.
        min_protected_height : float
            Minimum absolute protected height in meters.
        near_ground_relax : float
            Under-relaxation factor applied at the first movable layer.
        omega : float
            Target relaxation factor away from the ground.
        redistribute_after : bool
            If True, re-impose a monotone vertical distribution after smoothing.
        redistribute_keep_layers : int
            Number of bottom layers to preserve when redistributing.
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

        first_spacing = np.maximum(original_Z[:, :, 1] - original_Z[:, :, 0], 1e-6)
        protected_depth = np.maximum(first_spacing * protected_factor, min_protected_height)
        protect_stop = self._compute_protect_stop(original_Z, protected_depth)

        layer_relax = np.ones(nz, dtype=X.dtype) * omega
        layer_relax[:start_layer] = 0.0

        if blend_layers > 0:
            for offset in range(blend_layers):
                k = start_layer + offset
                if k >= nz - 1:
                    break
                frac = (offset + 1) / max(1, blend_layers)
                layer_relax[k] = near_ground_relax + (omega - near_ground_relax) * frac

        layer_relax[start_layer + blend_layers:nz - 1] = omega
        layer_relax = np.clip(layer_relax, 0.0, 1.0)
        protect_stop = np.ascontiguousarray(protect_stop.astype(np.int32))
        layer_relax = np.ascontiguousarray(layer_relax)

        for _ in range(max_iters):
            sweep_err = self._gauss_seidel_sweep(start_layer, layer_relax, protect_stop)
            if sweep_err < tol:
                break

        self._blend_layers(original_X, original_Y, original_Z,
                           start_layer, max_blend_layer)

        if redistribute_after:
            self.redistribute_columns(original_Z,
                                      keep_layers=redistribute_keep_layers,
                                      respect_first_layers=True)

    @staticmethod
    @njit
    def _gauss_seidel_core(X, Y, Z, valid_cols, start_layer,
                           layer_relax, protect_stop):
        _, _, nz = X.shape
        max_disp = 0.0

        for k in range(start_layer, nz - 1):  # keep the top plane fixed
            relax = layer_relax[k]
            if relax <= 0.0:
                continue
            km = k - 1
            kp = k + 1

            for idx in range(valid_cols.shape[0]):
                i = valid_cols[idx, 0]
                j = valid_cols[idx, 1]

                if k <= protect_stop[i, j]:
                    continue

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
                disp = (dx * dx + dy * dy + dz * dz) ** 0.5
                if disp > max_disp:
                    max_disp = disp

                X[i, j, k] = X[i, j, k] + relax * dx
                Y[i, j, k] = Y[i, j, k] + relax * dy
                Z[i, j, k] = Z[i, j, k] + relax * dz

        return max_disp

    def _gauss_seidel_sweep(self,
                            start_layer: int,
                            layer_relax: np.ndarray,
                            protect_stop: np.ndarray) -> float:
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

        return self._gauss_seidel_core(X, Y, Z,
                                       valid_cols,
                                       start_layer,
                                       layer_relax,
                                       protect_stop)

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

    def _compute_protect_stop(self, original_Z: np.ndarray, protected_depth: np.ndarray) -> np.ndarray:
        """
        Returns, for each column, the highest layer index that must remain
        untouched by the elliptic smoother (inclusive).
        """
        nx, ny, nz = original_Z.shape
        protect_stop = np.zeros((nx, ny), dtype=np.int32)
        base = original_Z[:, :, 0]
        for k in range(1, nz):
            depth = original_Z[:, :, k] - base
            mask = (depth < protected_depth) & (protect_stop < k)
            protect_stop[mask] = k
        protect_stop = np.clip(protect_stop, 0, nz - 2)
        return protect_stop

    def redistribute_columns(self,
                             reference_Z: np.ndarray,
                             respect_first_layers: bool = True,
                             keep_layers: int = 2):
        """
        Re-impose a monotone vertical distribution per column using the
        original column as a template, optionally preserving the first
        `keep_layers` levels exactly.
        """
        X, Y, Z = self.mesh.X, self.mesh.Y, self.mesh.Z
        nx, ny, nz = Z.shape
        eps = 1e-9
        keep_layers = max(0, min(keep_layers, nz))

        for i in range(nx):
            for j in range(ny):
                bottom = Z[i, j, 0]
                top = Z[i, j, -1]
                ref_col = reference_Z[i, j, :]
                denom = max(ref_col[-1] - ref_col[0], eps)
                weights = (ref_col - ref_col[0]) / denom
                column = bottom + weights * (top - bottom)
                if respect_first_layers and keep_layers > 0:
                    column[:keep_layers] = Z[i, j, :keep_layers]
                Z[i, j, :] = column