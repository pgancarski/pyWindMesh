from pathlib import Path
from typing import Union, Optional
import yaml
from pydantic import BaseModel, model_validator


class Config(BaseModel):
    # Core parameters
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    rotation_deg: float

    # Farm extents
    farm_xt: float
    farm_xf: float
    farm_yt: float
    farm_yf: float
    farm_cellsize_x: float
    farm_cellsize_y: float

    # Buffer grid parameters
    buffer_size_x: float
    buffer_size_y: float
    buffer_cellsize_x: float
    buffer_cellsize_y: float

    @model_validator(mode='after')
    def set_center_defaults(self):
        # Default center to midpoint of farm extents if not provided
        if self.center_x is None:
            self.center_x = (self.farm_xt + self.farm_xf) / 2
        if self.center_y is None:
            self.center_y = (self.farm_yt + self.farm_yf) / 2
        return self

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to a YAML file containing the configuration.

        Returns:
            An instance of Config populated from the file.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        return cls(**data)


# Example usage:
# 1. Direct initialization; center_x/center_y default to farm midpoint if omitted:
# data = {
#     "rotation_deg": 45.0,
#     "farm_xt": 0.0,
#     "farm_xf": 100.0,
#     "farm_yt": -50.0,
#     "farm_yf": 50.0,
#     "farm_cellsize_x": 1.0,
#     "farm_cellsize_y": 1.0,
#     "buffer_size_x": 10.0,
#     "buffer_size_y": 5.0,
#     "buffer_cellsize_x": 0.5,
#     "buffer_cellsize_y": 0.5
# }
# config = Config(**data)
# print(config.center_x, config.center_y)  # 50.0, 0.0

# 2. Loading from YAML file:
# config = Config.from_file("config.yaml")