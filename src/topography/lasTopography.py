from topography import Topography
from dataclasses import dataclass, field
import numpy as np
import laspy
from scipy.spatial import KDTree

@dataclass
class LAS_Topography(Topography):
    laz_path: str = field(repr=False)
    k: int = 8                  # number of neighbors for interpolation
    power: float = 1.0          # inverse-distance power
    _points: np.ndarray = field(init=False, repr=False)
    _z: np.ndarray = field(init=False, repr=False)
    _kdtree: KDTree = field(init=False, repr=False)

    def __post_init__(self):
        # Load LAZ and build KD-tree for neighbor queries
        las = laspy.read(self.laz_path)
        self._points = las.xyz[:, :2]
        self._z = las.xyz[:, 2]

        self._kdtree = KDTree(self._points)

    def get_domain_range(self) -> tuple[float, float, float, float]:
        x_coords = self._points[:, 0]
        y_coords = self._points[:, 1]
        return x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()

    def sample_z(self, x: float, y: float) -> float:
        """Shepard (IDW) interpolation of z at (x, y)."""
        # query k neighbors
        dists, idxs = self._kdtree.query([x, y], k=min(self.k, len(self._z)))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        # if any neighbor is exactly at the query point, return its z
        zero_mask = (dists == 0.0)
        if np.any(zero_mask):
            return float(self._z[idxs[zero_mask][0]])

        # compute weights and weighted average
        weights = 1.0 / (dists ** self.power)
        z_vals = self._z[idxs]
        return float(np.sum(weights * z_vals) / np.sum(weights))

    def array_sample_Z(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Apply Shepard interpolation over a grid.
        X, Y must be same-shaped arrays of query coords.
        """
        flat_xy = np.column_stack((X.ravel(), Y.ravel()))
        # query k neighbors for all points
        k_eff = min(self.k, len(self._z))
        dists, idxs = self._kdtree.query(flat_xy, k=k_eff)
        # ensure 2D arrays of shape (n_points, k_eff)
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        Z_flat = np.empty(dists.shape[0], dtype=float)
        for i in range(dists.shape[0]):
            di = dists[i]
            ii = idxs[i]
            # exact match?
            zero = np.where(di == 0.0)[0]
            if zero.size:
                Z_flat[i] = self._z[ii[zero[0]]]
            else:
                w = 1.0 / (di ** self.power)
                Z_flat[i] = np.sum(w * self._z[ii]) / np.sum(w)

        return Z_flat.reshape(X.shape)
