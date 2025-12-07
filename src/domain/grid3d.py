import numpy as np

def estimate_grid3d_memory(nx, ny, nz, dtype=np.float64):
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = 3 * nx * ny * nz * bytes_per_element
    return total_bytes

def bytes_to_gb(b):
    return b / (1024**3)

class Grid3D:
    """
    Represents a 3D structured grid defined by X, Y, Z coordinate arrays.
    
    This class supports storage and manipulation of fields defined at:
      - grid points (nodes)
      - grid cells (cell-centered values)

    Attributes
    ----------
    X, Y, Z : np.ndarray
        3D arrays of coordinates created via meshgrid with 'ij' indexing.
    point_values : dict[str, np.ndarray]
        Dictionary of fields defined at grid nodes, shape (Nx, Ny, Nz).
    cell_values : dict[str, np.ndarray]
        Dictionary of fields defined at cell centers, shape (Nx-1, Ny-1, Nz-1).
    """

    def __init__(self, nx: int, ny: int, nz: int):
        """
        Initialize a 3D grid with given dimensions.

        Parameters
        ----------
        nx, ny, nz : int
            Number of points in x, y, z directions.
        """
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, nz must all be positive integers")

        # Coordinate arrays, initialized as zeros (to be filled / skewed later)
        try:
            self.X = np.zeros((nx, ny, nz), dtype=float)
            self.Y = np.zeros((nx, ny, nz), dtype=float)
            self.Z = np.zeros((nx, ny, nz), dtype=float)

        except MemoryError:
            needed_momory_in_GB = bytes_to_gb(estimate_grid3d_memory(nx,ny,nz)) 
            raise MemoryError(
                f"Unable to allocate Grid3D of size ({nx}, {ny}, {nz}). "
                f"Estimated memory usage of {needed_momory_in_GB} is too large."
            )


        # Values at points and at cell centers
        self.point_values: dict[str, np.ndarray] = {}
        self.cell_values: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------
    # Point values
    # ------------------------------------------------------------

    def create_point_values(self, name: str, values: np.ndarray):
        """Create and store a new field defined at grid points."""
        if values.shape != self.X.shape:
            raise ValueError(
                f"Point values shape {values.shape} does not match grid shape {self.X.shape}"
            )
        self.point_values[name] = np.array(values)

    def set_point_values(self, name: str, values: np.ndarray, create: bool = False):
        """Update or create a field defined at grid points."""
        if values.shape != self.X.shape:
            raise ValueError(
                f"Point values shape {values.shape} does not match grid shape {self.X.shape}"
            )

        if self.in_point_values(name):
            self.point_values[name] = np.array(values)
        elif create:
            self.create_point_values(name, values)
        else:
            raise KeyError(f"Point field '{name}' does not exist and create=False")

    def in_point_values(self, name: str) -> bool:
        return name in self.point_values

    def get_point_values(self, name: str) -> np.ndarray:
        """Retrieve point-based field values."""
        if name not in self.point_values:
            raise KeyError(f"Point field '{name}' does not exist")
        return self.point_values[name]

    # ------------------------------------------------------------
    # Cell-centered values
    # ------------------------------------------------------------

    def create_cell_values(self, name: str, values: np.ndarray):
        """
        Create and store a new field defined at grid cell centers.
        Expected shape: (Nx-1, Ny-1, Nz-1)
        """
        expected_shape = (
            self.X.shape[0] - 1,
            self.X.shape[1] - 1,
            self.X.shape[2] - 1,
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"Cell values shape {values.shape} does not match expected shape {expected_shape}"
            )
        self.cell_values[name] = np.array(values)

    def set_cell_values(self, name: str, values: np.ndarray, create: bool = False):
        """
        Update or create a field defined at cell centers.
        """
        expected_shape = (
            self.X.shape[0] - 1,
            self.X.shape[1] - 1,
            self.X.shape[2] - 1,
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Cell values shape {values.shape} does not match expected shape {expected_shape}"
            )

        if self.in_cell_values(name):
            self.cell_values[name] = np.array(values)
        elif create:
            self.create_cell_values(name, values)
        else:
            raise KeyError(f"Cell field '{name}' does not exist and create=False")

    def in_cell_values(self, name: str) -> bool:
        return name in self.cell_values

    def get_cell_values(self, name: str) -> np.ndarray:
        """Retrieve cell-centered field values."""
        if name not in self.cell_values:
            raise KeyError(f"Cell field '{name}' does not exist")
        return self.cell_values[name]

    # ------------------------------------------------------------
    # Flattening utilities
    # ------------------------------------------------------------

    def to_points_vector(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Flatten X, Y, Z and point_values into 1D arrays.
        """
        X_vec = self.X.ravel()
        Y_vec = self.Y.ravel()
        Z_vec = self.Z.ravel()
        vals_vec = {key: arr.ravel() for key, arr in self.point_values.items()}
        return X_vec, Y_vec, Z_vec, vals_vec

