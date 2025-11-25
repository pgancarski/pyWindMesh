import numpy as np
import numba as nb

Array2D = np.ndarray  # alias for readability


@nb.njit(fastmath=True)
def _rotate_kernel(points: Array2D,
                   out: Array2D,
                   angle_deg: float,
                   cx: float,
                   cy: float) -> None:
    """
    Internal Numba kernel: writes results into `out`.
    """
    if angle_deg == 0:
        out = points
        return

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for i in range(points.shape[0]):
        x = points[i, 0] - cx
        y = points[i, 1] - cy

        out[i, 0] = cos_t * x - sin_t * y + cx
        out[i, 1] = sin_t * x + cos_t * y + cy


def rotate_points(points: Array2D,
                  angle_deg: float,
                  center_x: float = 0.0,
                  center_y: float = 0.0,
                  inplace: bool = False) -> Array2D:
    """
    Rotate an (N,2) array of points around (center_x, center_y).

    Parameters
    ----------
    points : ndarray (N,2)
    angle_deg : float
    center_x, center_y : float
    inplace : bool
        - True: modify points array in place
        - False: return new rotated array

    Returns
    -------
    points_rotated : ndarray (if inplace=False)
    """
    if inplace:
        _rotate_kernel(points, points, angle_deg, center_x, center_y)
        return points
    else:
        out: Array2D = np.empty_like(points)
        _rotate_kernel(points, out, angle_deg, center_x, center_y)
        return out


def un_rotate_points(points: Array2D,
                     angle_deg: float,
                     center_x: float = 0.0,
                     center_y: float = 0.0,
                     inplace: bool = False) -> Array2D:
    """
    Apply the inverse rotation.
    """
    return rotate_points(points, -1*angle_deg, center_x, center_y, inplace)
