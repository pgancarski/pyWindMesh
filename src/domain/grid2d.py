import numpy as np

class Grid2D:
    """
    Represents a 2D structured grid defined by X and Y coordinate arrays.
    
    This class allows storage and manipulation of scalar or vector fields
    defined at grid **points** (nodes) or **faces** (cells). Point-based values
    are stored at each grid coordinate, while face-based values are defined
    between adjacent grid points (one fewer in each direction).

    Attributes
    ----------
    X : np.ndarray
        2D array of x-coordinates (created using meshgrid with 'ij' indexing).
    Y : np.ndarray
        2D array of y-coordinates (created using meshgrid with 'ij' indexing).
    point_values : dict[str, np.ndarray]
        Dictionary mapping field names to arrays of values defined at grid points.
    face_values : dict[str, np.ndarray]
        Dictionary mapping field names to arrays of values defined on grid faces.
    """

    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """
        Initialize the 2D grid using 1D arrays of x and y coordinates.

        Parameters
        ----------
        x_vals : np.ndarray
            1D array of x-coordinate values.
        y_vals : np.ndarray
            1D array of y-coordinate values.

        Raises
        ------
        ValueError
            If `x_vals` or `y_vals` are not 1D arrays.
        """
        if x_vals.ndim != 1 or y_vals.ndim != 1:
            raise ValueError("x_vals and y_vals must be 1D arrays")

        self.X, self.Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        self.point_values = {}
        self.face_values = {}

    def create_point_values(self, name: str, values: np.ndarray):
        """
        Create and store a new field defined at grid points.

        Parameters
        ----------
        name : str
            Name of the field to create.
        values : np.ndarray
            2D array of values matching the grid shape (same shape as `X` and `Y`).

        Raises
        ------
        ValueError
            If the shape of `values` does not match the grid shape.
        """
        if values.shape != self.X.shape:
            raise ValueError(
                f"Point values shape {values.shape} does not match grid shape {self.X.shape}"
            )
        self.point_values[name] = np.array(values)

    def create_face_values(self, name: str, values: np.ndarray):
        """
        Create and store a new field defined at grid faces (cells).

        Parameters
        ----------
        name : str
            Name of the face field to create.
        values : np.ndarray
            2D array of values defined at grid faces. Expected shape is one less
            in both dimensions than the grid (X.shape[0]-1, X.shape[1]-1).

        Raises
        ------
        ValueError
            If the shape of `values` does not match the expected face shape.
        """
        expected_shape = (self.X.shape[0] - 1, self.X.shape[1] - 1)
        if values.shape != expected_shape:
            raise ValueError(
                f"Face values shape {values.shape} does not match expected shape {expected_shape}"
            )
        self.face_values[name] = np.array(values)

    def set_point_values(self, name: str, values: np.ndarray, create: bool = False):
        """
        Update or create a field defined at grid points.

        Parameters
        ----------
        name : str
            Name of the point field to set.
        values : np.ndarray
            2D array of values matching the grid shape.
        create : bool, optional
            If True, creates a new field if it doesn't exist. Defaults to False.

        Raises
        ------
        ValueError
            If the shape of `values` does not match the grid shape.
        KeyError
            If the field does not exist and `create` is False.
        """
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

    def in_point_values(self, name: str)->bool:
        return name in self.point_values
    
    def in_face_values(self, name: str)->bool:
        return name in self.face_values

    def set_face_values(self, name: str, values: np.ndarray, create: bool = False):
        """
        Update or create a field defined at grid faces (cells).

        Parameters
        ----------
        name : str
            Name of the face field to set.
        values : np.ndarray
            2D array of values matching the expected face shape.
        create : bool, optional
            If True, creates a new field if it doesn't exist. Defaults to False.

        Raises
        ------
        ValueError
            If the shape of `values` does not match the expected face shape.
        KeyError
            If the field does not exist and `create` is False.
        """
        expected_shape = (self.X.shape[0] - 1, self.X.shape[1] - 1)
        if values.shape != expected_shape:
            raise ValueError(
                f"Face values shape {values.shape} does not match expected shape {expected_shape}"
            )

        if self.in_face_values(name):
            self.face_values[name] = np.array(values)
        elif create:
            self.create_face_values(name, values)
        else:
            raise KeyError(f"Face field '{name}' does not exist and create=False")
    
    def get_point_values(self, name: str) -> np.ndarray:
        """
        Retrieve a field defined at grid points.

        Parameters
        ----------
        name : str
            Name of the point field to retrieve.

        Returns
        -------
        np.ndarray
            2D array of values defined at grid points.

        Raises
        ------
        KeyError
            If the specified point field does not exist.
        """
        if name not in self.point_values:
            raise KeyError(f"Point field '{name}' does not exist")
        return self.point_values[name]

    def get_face_values(self, name: str) -> np.ndarray:
        """
        Retrieve a field defined at grid faces (cells).

        Parameters
        ----------
        name : str
            Name of the face field to retrieve.

        Returns
        -------
        np.ndarray
            2D array of values defined at grid faces.

        Raises
        ------
        KeyError
            If the specified face field does not exist.
        """
        if name not in self.face_values:
            raise KeyError(f"Face field '{name}' does not exist")
        return self.face_values[name]


    def to_points_vector(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Flatten X, Y, and all point_values arrays into 1D vectors.

        Returns:
            X_vec: np.ndarray
            Y_vec: np.ndarray
            vals_vec: dict[str, np.ndarray] with flattened arrays
        """
        X_vec = self.X.ravel()
        Y_vec = self.Y.ravel()
        vals_vec = {key: arr.ravel() for key, arr in self.point_values.items()}
        return X_vec, Y_vec, vals_vec
    
    def set_new_XYZ(self, X:np.ndarray, Y:np.ndarray, Z:np.ndarray):
        self.X = X
        self.Y = Y
        self.set_point_values("Z", Z)

    def set_new_XY(self, X:np.ndarray, Y:np.ndarray):
        self.X = X
        self.Y = Y
        

