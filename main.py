# %%

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from config import GroundMeshConfig
from config import TopographyConfig

from infrastructure.topography import LAS_Topography
from infrastructure.topography import GaussianHill

from infrastructure.mesh import GridMesh2D
from infrastructure.meshSmoother import SurfaceMaxAngleSmoothing
from infrastructure.meshSmoother import SurfaceSquerifyFaces

from infrastructure import Timer
from infrastructure.plotting import GroundGridPlot


terrain_mesh_setup = {
     "wind_direction": 45.0,
     #FARM
     "farm_size_up": 300,
     "farm_size_down": 400,
     "farm_size_left": 500,
     "farm_size_right": 400,
     "farm_cellsize_x": 9.0,
     "farm_cellsize_y": 10.0,
     # Transition
     "transition_size_up": 900,
     "transition_size_down": 1000,
     "transition_size_left": 1100,
     "transition_size_right": 1000,

     # Buffer grid parameters
     "buffer_size_up": 400,
     "buffer_size_down": 500,
     "buffer_size_left": 600,
     "buffer_size_right": 500,
     "buffer_cellsize_x": 40,
     "buffer_cellsize_y": 50,
}
topography_config = TopographyConfig(
     file_path="./data/CoromandelForestPark_OpenTopography.laz"
)

#topo = GaussianHill(config)
topo = LAS_Topography(topography_config=topography_config)

# Autoset the center point to the center of topography
farm_xt , farm_xf, farm_yt, farm_yf = topo.get_domain_range()

terrain_mesh_setup["center_x"] = (farm_xf+farm_xt)/2
terrain_mesh_setup["center_y"] = (farm_yf+farm_yt)/2

config = Config(
     ground_mesh=GroundMeshConfig(**terrain_mesh_setup),
     topography=topography_config
)


mesh2d = GridMesh2D(config.ground_mesh)
max_angle_smoother = SurfaceMaxAngleSmoothing()
#squerify_smoother = SurfaceSquerifyFaces()
timer = Timer()



relaxation_factor = 0.5

max_angle_smoother.max_angle=10

timer.start()
mesh2d.set_Z(topo)

print("Mesh quality before smoothing")
mesh2d.check_mesh_quality()
epsilon = mesh2d.apply_grid_smoother(max_angle_smoother, max_steps=50,relaxation_factor=0.8) #n=20
#epsilon = mesh2d.apply_grid_smoother(squerify_smoother, max_steps=80, relaxation_factor=0.99) #n=5
#epsilon = mesh2d.apply_grid_smoother(max_angle_smoother, max_steps=20) #n=20
print("Mesh quality after smoothing")
mesh2d.check_mesh_quality()
timer.stop()


plotter = GroundGridPlot()

plotter.plot(mesh2d)

#plotter.plot(mesh2d,"buffer_blending")
#plotter.plot(mesh2d,"zone_id")