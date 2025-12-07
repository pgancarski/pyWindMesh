from dataclasses import dataclass



@dataclass
class ZoneConfig: 
    zone_code: int = 0 # the zone code

    dx_right: float = 0 # cell size, 
    dx_left: float = 0
    dy_up: float = 0
    dy_down: float = 0

    q_left: float = 1 # growth rate
    q_right: int = 1
    q_up: int = 1
    q_down: int = 1

    n_left: int = 0 # n cells in each direction
    n_right: int = 0
    n_up: int = 0
    n_down: int = 0

    ix_left_iner: int = 0 # indexes of iner binding box
    ix_right_iner: int = 0
    iy_up_iner: int = 0
    iy_down_iner: int = 0

    ix_left_outer: int = 0 # indexes of outer binding box
    ix_right_outer: int = 0
    iy_up_outer: int = 0
    iy_down_outer: int = 0

    x_left_iner: int = 0 # coordinates of iner binding box
    x_right_iner: int = 0
    y_up_iner: int = 0
    y_down_iner: int = 0

    x_left_outer: int = 0 # coordinates of outer binding box
    x_right_outer: int = 0
    y_up_outer: int = 0
    y_down_outer: int = 0 
