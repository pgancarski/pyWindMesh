import numpy as np
from typing import List
import warnings

from .zoneConfig import ZoneConfig

GROWTH_RATE_WARNING = (0.8 , 1.2)
GROWTH_RATE_ERROR =   (0.7 , 1.3)

# helper to compute angle between a,b: arccos((a·b)/(|a||b|))
def angle(a, b):
    dot = np.sum(a*b, axis=-1)
    na  = np.linalg.norm(a, axis=-1)
    nb  = np.linalg.norm(b, axis=-1)
    # clamp to [-1,1] to avoid NaNs
    cosang = np.clip(dot/(na*nb), -1.0, 1.0)
    return np.arccos(cosang)

    # ---- helper to fill bounds for one axis

def _fill_axis_bounds(
    zones: List[ZoneConfig],
    n_attr: str,
    dx_attr: str,
    ix_inner_attr: str,
    ix_outer_attr: str,
    x_inner_attr: str,
    x_outer_attr: str,
    sign: int,
    acc:int = 0,
) -> None:
    """
    Compute inner/outer indices and coordinates for a list of zones along one axis.

    sign = -1 for left/down, +1 for right/up
    """

    for z in zones:
        n = getattr(z, n_attr)
        dx = getattr(z, dx_attr)

        setattr(z, ix_inner_attr, acc)
        setattr(z, ix_outer_attr, acc + n)

        # coordinates: center at 0, going outwards with given sign
        setattr(z, x_inner_attr, sign * acc * dx)
        setattr(z, x_outer_attr, sign * (acc + n) * dx)

        acc += n
        
    return acc

def _distance_to_zone_border(xf, xt, yf, yt, x, y):
    # xt, yt = min; xf, yf = max
    in_zone = (x < xf and x > xt and y < yf and y > yt)

    if in_zone:
        # Point strictly inside: distance to nearest side (orthogonal)
        distance = min(
            x - xt,   # to left side
            xf - x,   # to right side
            y - yt,   # to bottom side
            yf - y    # to top side
        )
    else:
        # Point on or outside: distance to rectangle (which equals distance to border)
        # dx, dy are 0 if the point is horizontally/vertically aligned with the box
        dx = max(xt - x, 0, x - xf)
        dy = max(yt - y, 0, y - yf)
        distance = (dx*dx + dy*dy) ** 0.5

    return distance, in_zone

# Helper function: generate coordinates expanding from farm zone center
def generate_axis(
    zones: List[ZoneConfig],
    horizontal_axis: bool,
    center: float = 0.0,
):
    """
    Build a 1D coordinate axis expanding OUTWARD from the center.

    Rules:
      - Iterate zones from inner to outer: Farm -> Transition -> Buffer (reverse input [BZ, TZ, FZ]).
      - Positive side (right/up): use q as given (FZ->BZ).
      - Negative side (left/down): marching FZ->BZ but q is defined BZ->FZ, so use 1/q.
      - First step of each segment uses the farm spacing at the center edge.
    """
    eps = 1e-12
    zones_io = zones[::-1]  # expected input is [BZ, TZ, FZ]

    # ---- NEGATIVE SIDE (left/down) ----
    x_neg_vals = [center]
    cur = center
    for z in zones_io:
        if horizontal_axis:
            n = z.n_left
            q_raw = z.q_left
            step0 = z.dx_right  # farm spacing at the center edge along +X
        else:
            n = z.n_down
            q_raw = z.q_down
            step0 = z.dy_up     # farm spacing at the center edge along +Y

        if n <= 0:
            continue

        r = q_raw
        d = step0 * (r ** np.arange(n))
        cur -= np.cumsum(d)
        x_neg_vals.extend(cur.tolist())
        cur = x_neg_vals[-1]

    x_neg = np.array(x_neg_vals)

    # ---- POSITIVE SIDE (right/up) ----
    x_pos_vals = [center]
    cur = center
    for z in zones_io:
        if horizontal_axis:
            n = z.n_right
            q_raw = z.q_right
            step0 = z.dx_right  # farm spacing at the center edge along +X
        else:
            n = z.n_up
            q_raw = z.q_up
            step0 = z.dy_up     # farm spacing at the center edge along +Y

        if n <= 0:
            continue

        r = q_raw
        d = step0 * (r ** np.arange(n))
        cur += np.cumsum(d)
        x_pos_vals.extend(cur.tolist())
        cur = x_pos_vals[-1]

    x_pos = np.array(x_pos_vals)

    # Combine: negative (reversed) + center + positive (skip duplicate center)
    return np.concatenate([x_neg[::-1][:-1], x_pos])

def compute_growth_rate(farm_cellsize, buffer_cellsize, transition_zone_size):
    a = float(farm_cellsize)
    b = float(buffer_cellsize)
    L_target = float(transition_zone_size)

    if a <= 0 or b <= 0 or L_target <= 0:
        raise ValueError("Cell sizes and transition_zone_size must be positive.")

    # initial guess for n based on average spacing
    avg = 0.5 * (a + b)
    n0 = max(2, int(round(L_target / avg)))

    best_n = None
    best_q = None
    best_err = float("inf")

    # search a reasonable window around n0
    for n in range(2, max(3, 4 * n0 + 1)):
        # enforce end-size constraint: a * q^(n-1) = b
        q = (b / a) ** (1.0 / (n - 1))

        # compute resulting length
        if abs(q - 1.0) < 1e-10:
            L = a * n
        else:
            L = a * (q**n - 1.0) / (q - 1.0)

        err = abs(L - L_target)
        if err < best_err:
            best_err = err
            best_n = n
            best_q = q

    if best_n is None or best_q is None:
        raise RuntimeError("Could not determine growth rate.")

    q = best_q
    n_cells = best_n

    # --- sanity checks on q (your thresholds, but with correct logic) ---
    if not (GROWTH_RATE_ERROR[0] <= q <= GROWTH_RATE_ERROR[1]):
        raise ValueError(
            f"Transition mesh growth rate q={q:.4f} is outside safe range {GROWTH_RATE_ERROR}"
        )
    if not (GROWTH_RATE_WARNING[0] <= q <= GROWTH_RATE_WARNING[1]):
        warnings.warn(
            f"Transition mesh growth rate q={q:.4f} is high. "
            "Consider increasing transition zone size."
        )

    return q, n_cells
