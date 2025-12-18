# %%

import os
import sys
from pathlib import Path

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
from infrastructure.plotting import NonOrthogonalityHotspotsPlot3D

from infrastructure.exporters import grid3D_to_gmsh
from infrastructure.exporters.openFoam import OpenFoamStructuredMeshExporter
from infrastructure.exporters.openFoam import CellFieldExportSpec


terrain_mesh_setup = {
     "wind_direction": 0.0,
     #FARM
     "farm_size_up": 400,
     "farm_size_down": 400,
     "farm_size_left": 400,
     "farm_size_right": 400,
     "farm_cellsize_x": 20.0,
     "farm_cellsize_y": 20.0,
     # Transition
     "transition_size_up": 1000,
     "transition_size_down": 1000,
     "transition_size_left": 1000,
     "transition_size_right": 1000,

     # Buffer grid parameters
     "buffer_size_up": 500,
     "buffer_size_down": 500,
     "buffer_size_left": 500,
     "buffer_size_right": 500,
     "buffer_cellsize_x": 100,
     "buffer_cellsize_y": 100,
}
topography_config = TopographyConfig(
     file_path="./data/CoromandelForestPark_OpenTopography.laz"
)

vertical_config = VerticalDistributionConfig(
     minztop=3000,
     height_multiplier=5,
     growth_rate=1.13,
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
squerify_smoother = SurfaceSquerifyFaces()
timer = Timer()

relaxation_factor = 0.9
max_angle_smoother.max_angle=10

timer.start()
mesh2d.set_Z(topo)

print("Mesh quality before smoothing")
mesh2d.check_mesh_quality()
epsilon = mesh2d.apply_grid_smoother(max_angle_smoother, max_steps=20,relaxation_factor=relaxation_factor)
epsilon = mesh2d.apply_grid_smoother(squerify_smoother, max_steps=10, relaxation_factor=relaxation_factor, zones=["TRANSITION", "FARM"]) 
# reinterpolate the topography not to get too far from oryginal, and apply the filters again
mesh2d.set_Z(topo) 
epsilon = mesh2d.apply_grid_smoother(max_angle_smoother, max_steps=20,relaxation_factor=relaxation_factor) 
# get more agressive in farm zone, preventing the filter from modifying too much growth rates in transition
epsilon = mesh2d.apply_grid_smoother(squerify_smoother, max_steps=40, relaxation_factor=1, zones=["FARM"])  
epsilon = mesh2d.apply_grid_smoother(max_angle_smoother, max_steps=10, relaxation_factor=relaxation_factor) 


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

print("Main, mesh shape:",mesh3d.mesh_shape)
size = 1
for s in mesh3d.mesh_shape:
     size = s*size
size=int(size/100000)/10.0
print("Main, mesh size M: ",size)
timer.stop()

plotter_3D = CrossectionMeshPlot3D()
plotter_3D.plot(mesh3d)
plotter_3D = NonOrthogonalityHotspotsPlot3D()
plotter_3D.plot(mesh3d)

if False:
     grid3D_to_gmsh(
          grid = mesh3d.mesh, 
          wind_direction=config.ground_mesh.wind_direction,
          center_x=config.ground_mesh.center_x,
          center_y=config.ground_mesh.center_y,
          file_path="./data/of/pyWindMesh.msh"
     )

if True:

     exporter = OpenFoamStructuredMeshExporter(
          patch_map={
               "zmax": {"name": "atmosphere", "type": "patch"},
               "zmin": {"name": "terrain", "type": "wall"},
          }
     )

     exporter.export(
          mesh3d,
          case_dir=Path("data/of"),
          apply_rotation=True,  # keep consistent with your plotting workflow
          cell_field_specs=[
               CellFieldExportSpec(
                    name="non_orthogonality_deg",
                    dimensions="[0 0 0 0 0 0 0]",
                    default_boundary="zeroGradient",
                    time_folder="0"
               )
          ],
     )