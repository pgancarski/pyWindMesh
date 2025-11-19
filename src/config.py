from pathlib import Path
from typing import Union, Optional
import yaml
import math
from pydantic import BaseModel, model_validator

"""
Conventions:

X axis in 2D is horizontal, Y axis is vertical
X grows to the right, Y grows up

xf,0 is a vector right, without rotation it points to E
xt,0 is a vector left, without rotation it points to W
0,yf is a vector up, without rotation it points to N
0,yt is a vector down, without rotation it points to S

"""
 

class GroundMeshConfig(BaseModel):
    # Core parameters
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    wind_direction: float # Wind comming from south is 0, comming from west is 90

    #Farm
    farm_cellsize_x: float
    farm_cellsize_y: float

    ## Farm extents relative
    farm_size_up: Optional[float] = None
    farm_size_down: Optional[float] = None
    farm_size_left: Optional[float] = None
    farm_size_right: Optional[float] = None

    ## Farm extents absolute
    farm_xt: Optional[float] = None
    farm_xf: Optional[float] = None
    farm_yt: Optional[float] = None
    farm_yf: Optional[float] = None

    # Transition
    transition_size_up: float
    transition_size_down: float
    transition_size_left: float
    transition_size_right: float

    # Buffer grid parameters
    buffer_size_up: float
    buffer_size_down: float
    buffer_size_left: float
    buffer_size_right: float
    buffer_cellsize_x: float
    buffer_cellsize_y: float

    # >>> Declare the fields you set later
    expected_topography_xf: Optional[float] = None
    expected_topography_xt: Optional[float] = None
    expected_topography_yf: Optional[float] = None
    expected_topography_yt: Optional[float] = None

    @model_validator(mode='after')
    def set_ranges(self):
        well_defined_inputs = False

        # Are we using relative or absolute farm definition
        if (
                self.center_x is not None and
                self.center_y is not None and
                self.farm_size_up is not None and self.farm_size_up > 0 and
                self.farm_size_down is not None and self.farm_size_down > 0 and
                self.farm_size_left is not None and self.farm_size_left > 0 and
                self.farm_size_right is not None and self.farm_size_right > 0
            ):
            well_defined_inputs = True
        
        if (
                self.farm_xt is not None and
                self.farm_xf is not None and
                self.farm_yt is not None and
                self.farm_yf is not None and
                self.farm_xf > self.farm_xt and
                self.farm_yf > self.farm_yt
            ):
            # Default center to midpoint of farm extents if not provided
            if self.center_x is None:
                self.center_x = (self.farm_xt + self.farm_xf) / 2
            if self.center_y is None:
                self.center_y = (self.farm_yt + self.farm_yf) / 2

            self.set_farm_size_variables()
            well_defined_inputs = True

        
        # abort if the inputs are not ok
        if not well_defined_inputs:
            raise Exception("ERROR: Insuficient or incorrect config provided for farm zone definition.") 


        self.set_topography_range()

        return self

    def set_farm_size_variables(self):
        # we work directly in relative space so center is 0,0
        corner_points = [
            (0, 0) 
        ]

        # vectors from center to corners
        extension_vectors = [
            (self.farm_xt - self.center_x , self.farm_yf - self.center_y ), #left  top
            (self.farm_xf - self.center_x , self.farm_yf - self.center_y ), #right top
            (self.farm_xt - self.center_x , self.farm_yt - self.center_y ), #left  bottom
            (self.farm_xf - self.center_x , self.farm_yt - self.center_y ), #right bottom
        ]
    
        xf, xt, yf, yt = self.compute_bounding_box(corner_points, extension_vectors, self.wind_direction)

        # the minimum bounds are negative because the center is (0,0) so change the signs
        self.farm_size_right = xf
        self.farm_size_left = -1 * xt 
        self.farm_size_up = yf
        self.farm_size_down = -1 * yt

    def set_topography_range(self):
        # we only have the center point as reference
        corner_points = [
            (self.center_x, self.center_y) 
        ]

        # from the center point we extend to 4 corners that include farm+transition zones
        extension_vectors = [
            (self.farm_size_left  + self.transition_size_left  , self.farm_size_up    + self.transition_size_up   ), #left  top
            (self.farm_size_right + self.transition_size_right , self.farm_size_up    + self.transition_size_up   ), #right top
            (self.farm_size_left  + self.transition_size_left  , self.farm_size_down + self.transition_size_down), #left  bottom
            (self.farm_size_right + self.transition_size_right , self.farm_size_down + self.transition_size_down), #right bottom
        ]

        # set topography range
        xf, xt, yf, yt = self.compute_bounding_box(corner_points, extension_vectors, self.wind_direction)

        self.expected_topography_xf = xf
        self.expected_topography_xt = xt
        self.expected_topography_yf = yf
        self.expected_topography_yt = yt

    def compute_bounding_box(            
            self,
            corner_points: list[tuple[float, float]],
            extension_vectors: list[tuple[float, float]],
            rotation_deg: float
        ):
            # Prepare rotation
            theta = math.radians(rotation_deg or 0.0)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            # 2D rotation: [cos -sin; sin cos] * (vx, vy)
            rotated_vectors = [(cos_t * vx - sin_t * vy, sin_t * vx + cos_t * vy) for vx, vy in extension_vectors]

            # Collect candidate points: the corners themselves and corner+rotated_vector
            xs = []
            ys = []
            for x, y in corner_points:
                # Original corner
                xs.append(float(x))
                ys.append(float(y))
                # Extended points for each rotated vector
                for rvx, rvy in rotated_vectors:
                    xs.append(float(x) + rvx)
                    ys.append(float(y) + rvy)

            return max(xs), min(xs), max(ys), min(ys)




#class GeneralConfig(BaseModel):
    

class TopographyConfig(BaseModel):
    file_path: str

class Config(BaseModel):
    #general:GeneralConfig
    ground_mesh:GroundMeshConfig
    topography:TopographyConfig

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to a YAML file containing the configuration.

        Returns:
            An instance of Config populated from the file.
        """
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        try:
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {p}: {e}") from e

        return cls.model_validate(data)