
import numpy as np
from typing import List, Optional



from src.domain import Mesh2D
from src.domain import Grid2D
from src.domain import SpatialValueProvider
from src.domain import GridSmoother2D

from .zoneConfig import ZoneConfig
from .meshUtils import angle, compute_growth_rate, generate_axis, _distance_to_zone_border, _fill_axis_bounds

from config import GroundMeshConfig




"""
Conventions:

X axis in 2D is horizontal, Y axis is vertical
X grows to the right, Y grows up

xf,0 is a vector right, without rotation it points to E
xt,0 is a vector left, without rotation it points to W
0,yf is a vector up, without rotation it points to N
0,yt is a vector down, without rotation it points to S

mesh oeprates in a space that was rotated and is relative to the center point

"""

class GridMesh2D(Mesh2D):
    """
    Constructs a regular 2D mesh based on a configuration dictionary.
    """
    def __init__(self, config: GroundMeshConfig):
        # Unpack configuration
        self.mesh_zones_measurments = self._create_zone_measurments(config)

        # Generate coordinates for each axis
        x_vals = generate_axis(zones=self.mesh_zones_measurments, horizontal_axis=True, center=config.center_x)
        y_vals = generate_axis(zones=self.mesh_zones_measurments, horizontal_axis=False, center=config.center_y)

        # Create coordinate grids
        self.grid = Grid2D(x_vals, y_vals)

        # Initialize Z to zero
        self.grid.create_point_values("Z", np.zeros_like(self.grid.X))

        # Create zones
        self.grid.create_point_values("zone_id", self._compute_zone_id())
        self.grid.create_point_values("buffer_blending", self._compute_buffer_blending())

    def _create_zone_measurments(self, config: "GroundMeshConfig") -> List[ZoneConfig]:
        # We operate in a space relative to the center point, that has already been rotated

        # -------------------------------------------------------------------------
        # Farm Zone
        # -------------------------------------------------------------------------
        _FZ = ZoneConfig()
        _FZ.dx_right = config.farm_cellsize_x
        _FZ.dx_left  = config.farm_cellsize_x
        _FZ.dy_up    = config.farm_cellsize_y
        _FZ.dy_down  = config.farm_cellsize_y

        _FZ.n_left  = int(round(config.farm_size_left  / _FZ.dx_left))  + 1
        _FZ.n_right = int(round(config.farm_size_right / _FZ.dx_right)) + 1
        _FZ.n_up    = int(round(config.farm_size_up    / _FZ.dy_up))    + 1
        _FZ.n_down  = int(round(config.farm_size_down  / _FZ.dy_down))  + 1

        # -------------------------------------------------------------------------
        # Buffer Zone
        # -------------------------------------------------------------------------
        _BZ = ZoneConfig()
        _BZ.dx_right = config.buffer_cellsize_x
        _BZ.dx_left  = config.buffer_cellsize_x
        _BZ.dy_up    = config.buffer_cellsize_y
        _BZ.dy_down  = config.buffer_cellsize_y

        _BZ.n_left  = int(round(config.buffer_size_left  / _BZ.dx_left))  + 1
        _BZ.n_right = int(round(config.buffer_size_right / _BZ.dx_right)) + 1
        _BZ.n_up    = int(round(config.buffer_size_up    / _BZ.dy_up))    + 1
        _BZ.n_down  = int(round(config.buffer_size_down  / _BZ.dy_down))  + 1

        # -------------------------------------------------------------------------
        # Transition Zone
        # -------------------------------------------------------------------------
        _TZ = ZoneConfig()

        # start with farm size and finish with buffer
        _TZ.dx_right = config.farm_cellsize_x
        _TZ.dx_left  = config.buffer_cellsize_x
        _TZ.dy_up    = config.farm_cellsize_y
        _TZ.dy_down  = config.buffer_cellsize_y


        _TZ.q_left,  _TZ.n_left  = compute_growth_rate(_TZ.dx_right, _TZ.dx_left,  config.transition_size_left)
        _TZ.q_right, _TZ.n_right = compute_growth_rate(_TZ.dx_right, _TZ.dx_left,  config.transition_size_right)
        _TZ.q_down,  _TZ.n_down  = compute_growth_rate(_TZ.dy_up,   _TZ.dy_down,   config.transition_size_down)
        _TZ.q_up,    _TZ.n_up    = compute_growth_rate(_TZ.dy_up,   _TZ.dy_down,   config.transition_size_up)

        # -------------------------------------------------------------------------
        # Compute index and coordinate bounds for all zones
        # Order is relevant: go left/down -> center
        # Zones are in order [Buffer, Transition, Farm]
        # -------------------------------------------------------------------------
        zones = [_BZ, _TZ, _FZ]

        # LEFT  (negative X)
        acc_left = _fill_axis_bounds(
            zones,
            n_attr="n_left",
            dx_attr="dx_left",
            ix_inner_attr="ix_left_iner",
            ix_outer_attr="ix_left_outer",
            x_inner_attr="x_left_iner",
            x_outer_attr="x_left_outer",
            sign=-1,
        )

        # RIGHT (positive X)
        _fill_axis_bounds(
            zones,
            n_attr="n_right",
            dx_attr="dx_right",
            ix_inner_attr="ix_right_iner",
            ix_outer_attr="ix_right_outer",
            x_inner_attr="x_right_iner",
            x_outer_attr="x_right_outer",
            sign=+1,
            acc=acc_left,
        )

        # DOWN (negative Y)
        acc_down = _fill_axis_bounds(
            zones,
            n_attr="n_down",
            dx_attr="dy_down",
            ix_inner_attr="iy_down_iner",
            ix_outer_attr="iy_down_outer",
            x_inner_attr="y_down_iner",
            x_outer_attr="y_down_outer",
            sign=-1,
        )

        # UP (positive Y)
        _fill_axis_bounds(
            zones,
            n_attr="n_up",
            dx_attr="dy_up",
            ix_inner_attr="iy_up_iner",
            ix_outer_attr="iy_up_outer",
            x_inner_attr="y_up_iner",
            x_outer_attr="y_up_outer",
            sign=+1,
            acc=acc_down,
        )

        # the order of zones is relevant!!!
        # go left/down -> center
        # see generate_axis()
        return zones

    def _compute_zone_id(self) -> np.ndarray:
        """
        Creates an integer zone map:
            0 = Farm
            1 = Transition
            2 = Buffer
        """

        Z_FARM = 0
        Z_TRANS = 1
        Z_BUFF = 2

        BZ, TZ, FZ = self.mesh_zones_measurments

        ny, nx = self.grid.X.shape
        zone_id = np.full((ny, nx), Z_BUFF, dtype=int)   # start everything as buffer

        # TRANSITION 
        xt = BZ.n_left                                                # B
        xf = xt + TZ.n_left + FZ.n_left + FZ.n_right + TZ.n_right - 1 # B+T+F+F+T

        yt = BZ.n_down                                          # B
        yf = yt + TZ.n_down + FZ.n_down + FZ.n_up + TZ.n_up - 1 # B+T+F+F+T

        zone_id[xt:xf, yt:yf+1] = Z_TRANS

        # FARM 
        xt = BZ.n_left + TZ.n_left            # B+T
        xf = xt + FZ.n_left + FZ.n_right - 1  # B+T+F+F

        yt = BZ.n_down + TZ.n_down            # B+T
        yf = yt + FZ.n_down + FZ.n_up - 1     # B+T+F+F

        zone_id[xt:xf, yt:yf+1] = Z_FARM

        return zone_id
    
    def _compute_buffer_blending(self) -> np.ndarray:
        """
        0 = Farm, use terrain values
        1 = Buffer, use buffer value
        sin/cos-type smooth transition in between.

        Shape convention: (nx, ny)  ->  [x, y]
        """
        BZ, TZ, FZ = self.mesh_zones_measurments

        # X first, Y second
        nx, ny = self.grid.X.shape

        # blending mask: start with full buffer
        # blend[ix, iy] with shape (nx, ny)
        blend = np.ones((nx, ny), dtype=float)

        # ------------------------------------------------------------------
        # Layout along X:
        #   [ BZ.left | TZ.left |  FZ  | TZ.right | BZ.right ]
        #
        # Same concept along Y.
        # ------------------------------------------------------------------

        # (TZ + FZ + TZ) region bounds – inclusive
        xt = BZ.n_left
        xf = xt + TZ.n_left + FZ.n_left + FZ.n_right + TZ.n_right - 1

        yt = BZ.n_down
        yf = yt + TZ.n_down + FZ.n_down + FZ.n_up + TZ.n_up - 1

        # FARM-only rectangle (no transition inside)
        x_farm_min = xt + TZ.n_left
        x_farm_max = xf - TZ.n_right
        y_farm_min = yt + TZ.n_down
        y_farm_max = yf - TZ.n_up

        # Effective thickness of the transition zone in index units
        max_transition = max(TZ.n_left, TZ.n_right, TZ.n_up, TZ.n_down)

        # Degenerate case: no transition → hard switch farm/buffer
        if max_transition <= 0:
            for ix in range(nx):
                for iy in range(ny):
                    _, in_farm = _distance_to_zone_border(
                        x_farm_max, x_farm_min,
                        y_farm_max, y_farm_min,
                        ix, iy
                    )
                    if in_farm:
                        blend[ix, iy] = 0.0
            return blend

        # ------------------------------------------------------------------
        # Main loop: compute blending based on distance to FARM rectangle
        # ------------------------------------------------------------------
        for ix in range(nx):
            for iy in range(ny):
                d_to_farm, in_farm = _distance_to_zone_border(
                    x_farm_max, x_farm_min,
                    y_farm_max, y_farm_min,
                    ix, iy
                )

                if in_farm:
                    # Inside farm: pure farm value
                    blend[ix, iy] = 0.0
                else:
                    # Normalized distance 0..1 across transition thickness
                    t = min(1.0, d_to_farm / float(max_transition))

                    # Smooth "sin-type" ramp:
                    # 0 -> 0, 1 -> 1, C^1 continuous (equivalent to sin^2(pi * t / 2))
                    blend[ix, iy] = 0.5 * (1.0 - np.cos(np.pi * t))

        return blend


    def _compute_buffer_level(self, down_buffer_level: float, up_buffer_level: float) -> np.ndarray:
        """
        Returns a 2D array of expected buffer levels.

        - Constant along X (columns)
        - Varies along Y (rows)
        - = down_buffer_level in the bottom buffer region
        - = up_buffer_level in the top buffer region
        - Cos-shaped smooth transition across (TZ + FZ + TZ) in Y
        """

        BZ, TZ, FZ = self.mesh_zones_measurments

        # Correct dimension order
        nx, ny = self.grid.X.shape   # X = columns, Y = rows

        # Vertical layout in Y:
        #   [ BZ.down | TZ.down | FZ | TZ.up | BZ.up ]
        yt = BZ.n_down
        yf = yt + TZ.n_down + FZ.n_down + FZ.n_up + TZ.n_up - 1

        # 1D profile along Y (vertical)
        levels_y = np.empty(ny, dtype=float)

        # Degenerate case: no middle region
        if yf <= yt:
            for iy in range(ny):
                t = iy / float(ny - 1) if ny > 1 else 0.5
                w = 0.5 * (1.0 - np.cos(np.pi * t))
                levels_y[iy] = (1.0 - w) * down_buffer_level + w * up_buffer_level

        else:
            for iy in range(ny):
                if iy < yt:
                    levels_y[iy] = down_buffer_level
                elif iy > yf:
                    levels_y[iy] = up_buffer_level
                else:
                    # Inside the transition band
                    t = (iy - yt) / float(yf - yt) if yf > yt else 0.5
                    w = 0.5 * (1.0 - np.cos(np.pi * t))
                    levels_y[iy] = (1.0 - w) * down_buffer_level + w * up_buffer_level

        # Broadcast vertically-varying profile across X (constant in X)
        buffer_map = np.tile(levels_y[None, :], (nx, 1))

        return buffer_map


    def to_ground_grid(self) -> Grid2D:
        return self.grid
    
    def to_ground_points(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self.grid.to_points_vector()

    def get_point(self, ix: int, iy: int) -> tuple[float, float, float]:
        """
        Retrieve the point at mesh indices (ix, iy).

        Args:
            ix (int): Index along the X-direction (0 <= ix < nx).
            iy (int): Index along the Y-direction (0 <= iy < ny).

        Returns:
            Point: The Point object at the specified indices.
        """
        x = float(self.grid.X[ix, iy])
        y = float(self.grid.Y[ix, iy])
        z = float(self.grid.point_values["Z"][ix, iy])
        return x, y, z
    
    def set_point_values(self, name: str, values_provider:SpatialValueProvider):
        values = values_provider.values_at_points(self.grid.X, self.grid.Y)

        # if topography, apply blending for buffer and transition
        if name=="Z": 
            #if undefined figure out the buffer
            if not self.grid.in_point_values("buffer_level"):
                _,TZ,_ = self.mesh_zones_measurments
                down_buffer_level  = np.mean(values[TZ.ix_left_outer:TZ.ix_right_outer , TZ.iy_down_outer])
                up_buffer_level    = np.mean(values[TZ.ix_left_outer:TZ.ix_right_outer , TZ.iy_up_outer])
                self.grid.create_point_values("buffer_level", self._compute_buffer_level(down_buffer_level,up_buffer_level))

            buffer_level = self.grid.get_point_values("buffer_level")
            mask = self.grid.get_point_values("buffer_blending")
            values = (1.0 - mask) * values + mask * buffer_level

        self.grid.set_point_values(name, values, create=True)

    def set_face_values(self, name: str, values_provider:SpatialValueProvider):
        values = values_provider.values_at_points(self.grid.X, self.grid.Y)
        self.grid.set_face_values(name, values, create=True)

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the mesh as (nx, ny).
        """
        return self.grid.X.shape
    
    def check_mesh_quality(self):
        max_skewness, mean_skewness = self.mesh_skewness_stats()

        print("The maximum skewness angle [rad]: ",max_skewness)
        print("The mean skewness angle [rad]: ",mean_skewness)
    
    def mesh_skewness_stats(self):
        """
        Given X, Y, Z of shape (n, m), interpreted as a regular quad‐mesh,
        compute for each quad (cell) its skewness = max_i(|angle_i - 90°|)/90°,
        where angle_i are the four interior angles of the quad.
        Returns (max_skewness, avg_skewness), both in [0, 1].
        """
        # pack into points array of shape (n, m, 3)
        P = np.stack((self.grid.X, self.grid.Y, self.grid.point_values["Z"]), axis=-1)

        # grab the four corners of each quad:
        P00 = P[:-1, :-1]   # lower‐left
        P10 = P[1:,  :-1]   # upper‐left
        P11 = P[1:,  1:]    # upper‐right
        P01 = P[:-1, 1:]    # lower‐right

        # for each corner, form the two edges meeting there:
        # at P00: edges to P10 and to P01
        e00a = P10 - P00
        e00b = P01 - P00

        # at P10: edges to P11 and to P00
        e10a = P11 - P10
        e10b = P00 - P10

        # at P11: edges to P01 and to P10
        e11a = P01 - P11
        e11b = P10 - P11

        # at P01: edges to P00 and to P11
        e01a = P00 - P01
        e01b = P11 - P01

        # compute the four angle arrays (shape (n-1, m-1))
        θ00 = angle(e00a, e00b)
        θ10 = angle(e10a, e10b)
        θ11 = angle(e11a, e11b)
        θ01 = angle(e01a, e01b)

        # deviation from 90° = |θ - π/2|
        dev00 = np.abs(θ00 - np.pi/2)
        dev10 = np.abs(θ10 - np.pi/2)
        dev11 = np.abs(θ11 - np.pi/2)
        dev01 = np.abs(θ01 - np.pi/2)

        # per‐face skewness = (max deviation)/(π/2)
        max_dev = np.maximum.reduce([dev00, dev10, dev11, dev01])
        skewness = max_dev / (np.pi/2)

        return skewness.max(), skewness.mean()
        
    def apply_grid_smoother(
            self,
            grid_smoother: GridSmoother2D,
            relaxation_factor: float = 0.5,
            tol: Optional[float] = None,
            max_steps: Optional[int] = None,
            zone: Optional[List[str]] = None
        ):
        """
        Apply grid smoothing
        """

        # deal with None's
        kwargs = {}
        if tol is not None:
            kwargs["tol"] = tol
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if zone is not None:
            kwargs["zone"] = zone


        new_grid, error = grid_smoother.smooth(self.grid,relaxation_factor, **kwargs)
        self.grid = new_grid

        return error