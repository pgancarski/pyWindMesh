import numpy as np
from numba import njit

from src.config import VerticalDistributionConfig

FLAT_FRACTION = 0.25
TOLLERANCE = 1e-9
MAX_ITERATIONS = 60

@njit(cache=True)
def _compute_column(col, ground_z, refZ, n_const_top):
    """
    Numba-accelerated core: fill `col` with vertical coordinates for one column.
    """
    nz = len(refZ)
    n_strechable_layers = nz - n_const_top
    prevZ = ground_z
    stretch_top = refZ[n_strechable_layers]

    # --- stretched region ---
    for iz in range(1, n_strechable_layers):
        prevRefZ = refZ[iz - 1]
        refStep = refZ[iz] - prevRefZ
        scale = (stretch_top - prevZ) / (stretch_top - prevRefZ) # how far am I? how much space to recover?
        nextStep = refStep * scale # stretch/squeze the step to recover before stretch_top
        prevZ += nextStep
        col[iz] = prevZ

    # --- flat-top region: copy reference ---
    for iz in range(n_strechable_layers, nz):
        col[iz] = refZ[iz]


class VerticalLayersDistributor:
    """
    Builds a vertical layer distribution with:
      - constant bottom layers
      - geometric growth middle section
      - constant top layers
    Automatically stops near target height.

    config:
      first_cell_size   : base cell size
      growth_rate       : geometric growth rate for middle section
      n_const_bottom    : number of constant-thickness layers at bottom
      n_const_top       : number of constant-thickness layers at top
      target_height     : target vertical extent
      tolerance         : acceptable overshoot (default 0.01)
    """

    def __init__(self, config:VerticalDistributionConfig, refZ0:float):
        self.first_cell_size = config.first_cell_size
        self.growth_rate = config.growth_rate
        self.n_const_bottom = config.n_flat_layers
        self.target_height = config.minztop
        self.refZ0 = refZ0 #hight of the ground for the reference column (buffer inlet)

        self.h_bottom = self.n_const_bottom+self.first_cell_size # height of the bottom constant layers

        self.flat_fraction   = FLAT_FRACTION
        self.tol             = TOLLERANCE
        self.max_iterations  = MAX_ITERATIONS

        n_total, last_cell, height = self._solve_N()

        self.n_const_top = int(n_total * self.flat_fraction)
        self.last_cell_size = last_cell
        self.total_height = height

        # --- Build reference distribution automatically ---
        self.refZ = self._build_reference_layers(
            self.first_cell_size,
            self.last_cell_size,
            self.growth_rate,
            self.refZ0,
            self.n_const_bottom,
            self.n_const_top
        )

        self.mesh_size_z = self.refZ.size #for the final number of layers use the refZ column to deal with rounding errors
        self.ref_top = self.refZ[-1]



    def _height_for_N(self, N):
        """Return total column height and final top layer thickness."""
        N_top = int(N * self.flat_fraction)
        N_geo = N - N_top - self.n_const_bottom 
        if N_geo < 1:
            return float("inf"), None

        # correct bottom height: n_const_bottom layers of size first_cell_size
        h_bottom = self.n_const_bottom * self.first_cell_size

        # geometric region
        r = self.growth_rate
        a = self.first_cell_size

        h_last = a * r**(N_geo - 1)
        h_geo  = a * (r**N_geo - 1) / (r - 1)

        # flat-top region
        h_top = N_top * h_last

        return h_bottom + h_geo + h_top, h_last

    def _solve_N(self):
        # --- BRACKETING --------------------------
        N_min = self.n_const_bottom*(1+self.flat_fraction)+1
        # make sure you get reasonable first prediction
        height, _ = self._height_for_N(N_min)
        if height > self.target_height:
            raise RuntimeError("Unable to find vertical layers distribution. Review the config.")


        N_max = int(self.target_height / self.first_cell_size) * 2  # give some slack


        # --- BINARY SEARCH -------------------------------
        best_N = N_max
        best_height, best_last = self._height_for_N(best_N)

        for _ in range(self.max_iterations):
            N_mid = (N_min + N_max) // 2
            height, last_cell = self._height_for_N(N_mid)

            # track best (closest) solution
            if abs(height - self.target_height) < abs(best_height - self.target_height):
                best_N, best_height, best_last = N_mid, height, last_cell

            if height > self.target_height:
                N_max = N_mid - 1
            else:
                N_min = N_mid + 1

            # use a *relative* tolerance, not 1e-9 absolute
            if abs(height - self.target_height) <= max(self.tol, 1e-6 * self.target_height):
                break

        return best_N, best_last, best_height

    @staticmethod
    def _build_reference_layers(first_cell_size, last_cell_size,
                                growth_rate, refZ0,
                                n_const_bottom, n_const_top
                                ):
        """
        Automatically build the reference vertical coordinates.
        """
        refZ = [refZ0]

        # --- constant bottom region ---
        dz = first_cell_size
        
        for _ in range(n_const_bottom):
            refZ.append(refZ[-1] + dz)

        # --- geometric growth region ---
        while True:
            next_z = refZ[-1] + dz
            refZ.append(next_z)
            dz *= growth_rate
            if dz >= last_cell_size:
                break

        # --- constant top region ---
        dz = last_cell_size  # keep last step constant
        for _ in range(n_const_top):
            refZ.append(refZ[-1] + dz)

        return np.array(refZ, dtype=np.float64)

    def column_distribution(self, ground_z):
        """
        Compute the vertical coordinates for a single column at (x, y),
        given its ground elevation (bottom value).
        """
        nz = self.refZ.size
        col = np.empty(nz, dtype=np.float64)
        col[0] = ground_z

        _compute_column(
            col,
            float(ground_z),
            self.refZ,
            self.n_const_top,
        )

        return col
