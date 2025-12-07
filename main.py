# %%

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from config import GroundMeshConfig
from config import TopographyConfig
from config import VerticalDistributionConfig

from infrastructure.topography import LAS_Topography
from infrastructure.topography import GaussianHill

from infrastructure.mesh import GridMesh2D
from infrastructure.meshSmoother import SurfaceMaxAngleSmoothing
from infrastructure.meshSmoother import SurfaceSquerifyFaces

from infrastructure.mesh import StructuredHexMesh3d

from infrastructure import Timer
from infrastructure.plotting import GroundGridPlot
from infrastructure.plotting import CrossectionMeshPlot3D


terrain_mesh_setup = {
     "wind_direction": 45.0,
     #FARM
     "farm_size_up": 400,
     "farm_size_down": 400,
     "farm_size_left": 400,
     "farm_size_right": 400,
     "farm_cellsize_x": 5.0,
     "farm_cellsize_y": 5.0,
     # Transition
     "transition_size_up": 700,
     "transition_size_down": 700,
     "transition_size_left": 700,
     "transition_size_right": 700,

     # Buffer grid parameters
     "buffer_size_up": 500,
     "buffer_size_down": 500,
     "buffer_size_left": 500,
     "buffer_size_right": 500,
     "buffer_cellsize_x": 50,
     "buffer_cellsize_y": 50,
}
topography_config = TopographyConfig(
     file_path="./data/CoromandelForestPark_OpenTopography.laz"
)

vertical_config = VerticalDistributionConfig(
     minztop=2000,
     height_multiplier=5,
     growth_rate=1.15,
     n_flat_layers=5,
     first_cell_size=3,
)

# Autoset the center point to the center of topography
temp_topo = LAS_Topography(topography_config=topography_config)
farm_xt , farm_xf, farm_yt, farm_yf = temp_topo.get_domain_range()
terrain_mesh_setup["center_x"] = (farm_xf+farm_xt)/2
terrain_mesh_setup["center_y"] = (farm_yf+farm_yt)/2


# Load the config
config = Config(
     ground_mesh=GroundMeshConfig(**terrain_mesh_setup),
     topography=topography_config,
     vertical_distribution=vertical_config,
)

# properly load the topography
topo = LAS_Topography(topography_config=topography_config,mesh_config=config.ground_mesh)


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

if False:
     plotter = GroundGridPlot(
     config.ground_mesh.wind_direction,
     config.ground_mesh.center_x,
     config.ground_mesh.center_y,
     )

     plotter.plot(mesh2d)
     #plotter.plot(mesh2d,"buffer_blending")
     #plotter.plot(mesh2d,"zone_id")
timer.start()
mesh3d = StructuredHexMesh3d(config,mesh2d)
mesh3d.build_3d_mesh()

plotter_3D = CrossectionMeshPlot3D()

print("Main, mesh shape:",mesh3d.mesh_shape)
size = 1
for s in mesh3d.mesh_shape:
     size = s*size

size=int(size/100000)/10.0
print("Main, mesh size M: ",size)
timer.stop()
plotter_3D.plot(mesh3d)