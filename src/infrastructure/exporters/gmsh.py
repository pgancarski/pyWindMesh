from __future__ import annotations

import numpy as np
import gmsh
import statistics as stats
from typing import Iterable, Sequence, Dict, Any

from domain import Grid3D
from infrastructure.topography.topographyUtils import un_rotate_points

import gmsh
import statistics as stats
import numpy as np
from typing import Dict, Any


from typing import Any, Dict
import statistics as stats
import numpy as np
import gmsh

def grid3D_to_gmsh(
    grid: Grid3D,
    wind_direction: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    model_name: str = "structured_cube",
    file_path: str | None = "structured_cube.msh",
    volume_name: str = "DOMAIN",
    auto_finalize: bool = True,
) -> gmsh.model:
    """
    Convert a regular Grid3D object into a Gmsh discrete hexahedral mesh and
    register six boundary physical groups:

        ix=0        -> LEFT_WALL
        ix=-1       -> RIGHT_WALL
        iy=0        -> INLET
        iy=-1       -> OUTLET
        iz=0        -> GROUND
        iz=-1       -> TOP

    Parameters
    ----------
    grid : Grid3D
        Structured grid containing 3D coordinate arrays `X`, `Y`, `Z` plus
        optional dictionaries `point_values` and `cell_values`.
    wind_direction, center_x, center_y : float
        Same rotation arguments used in the PyVista version.
    model_name : str
        Name assigned to the Gmsh model.
    file_path : str or None
        If not None, the mesh is written to this path (e.g. "mesh.msh").
    volume_name : str
        Physical group name for the 3D volume.
    auto_finalize : bool
        When True, finalize Gmsh if this function performed the initialize() call.

    Returns
    -------
    gmsh.model
        The populated Gmsh model (useful if you need additional post-processing).
    """
    def _ensure_numpy(arr):
        return np.asarray(arr, order="F")

    own_session = not gmsh.isInitialized()
    if own_session:
        gmsh.initialize()
    gmsh.model.add(model_name)

    try:
        X = _ensure_numpy(grid.X)
        Y = _ensure_numpy(grid.Y)
        Z = _ensure_numpy(grid.Z)
        nx, ny, nz = X.shape

        total_nodes = nx * ny * nz
        node_tags = np.arange(1, total_nodes + 1, dtype=np.int64)

        points = np.column_stack((X.ravel(order="F"),
                                  Y.ravel(order="F"),
                                  Z.ravel(order="F")))

        # === Apply the same XY rotation used for the PyVista mesh ===
        xy = points[:, :2].copy()
        un_rotate_points(
            xy,
            angle_deg=wind_direction,
            center_x=center_x,
            center_y=center_y,
            inplace=True
        )
        points[:, :2] = xy

        model = gmsh.model
        volume_tag = model.addDiscreteEntity(3, 1)
        coords = points.astype(np.float64, copy=False).ravel()
        model.mesh.addNodes(3, volume_tag, node_tags, coords)

        def nid(i, j, k):
            return 1 + i + nx * (j + ny * k)

        hex_type = model.mesh.getElementType("Hexahedron", 1)
        volume_elements = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    volume_elements.append([
                        nid(i,     j,     k),
                        nid(i + 1, j,     k),
                        nid(i + 1, j + 1, k),
                        nid(i,     j + 1, k),
                        nid(i,     j,     k + 1),
                        nid(i + 1, j,     k + 1),
                        nid(i + 1, j + 1, k + 1),
                        nid(i,     j + 1, k + 1),
                    ])

        vol_elem_tags = np.arange(1, len(volume_elements) + 1, dtype=np.int64)
        model.mesh.addElements(
            3,
            volume_tag,
            [hex_type],
            [vol_elem_tags],
            [np.array(volume_elements, dtype=np.int64).ravel()]
        )

        boundary_specs = {
            "LEFT_WALL":  {"axis": "i", "index": 0,      "normal": np.array([-1, 0, 0]), "entity": 101},
            "RIGHT_WALL": {"axis": "i", "index": nx - 1, "normal": np.array([ 1, 0, 0]), "entity": 102},
            "INLET":      {"axis": "j", "index": 0,      "normal": np.array([ 0,-1, 0]), "entity": 103},
            "OUTLET":     {"axis": "j", "index": ny - 1, "normal": np.array([ 0, 1, 0]), "entity": 104},
            "GROUND":     {"axis": "k", "index": 0,      "normal": np.array([ 0, 0,-1]), "entity": 105},
            "TOP":        {"axis": "k", "index": nz - 1, "normal": np.array([ 0, 0, 1]), "entity": 106},
        }

        quad_type = model.mesh.getElementType("Quadrangle", 1)
        next_elem_tag = vol_elem_tags[-1] + 1 if len(vol_elem_tags) else 1

        for wall_name, spec in boundary_specs.items():
            axis = spec["axis"]
            idx = spec["index"]
            desired_normal = spec["normal"]
            surface_tag = model.addDiscreteEntity(2, spec["entity"])

            quads = []
            if axis == "i":
                i = idx
                for k in range(nz - 1):
                    for j in range(ny - 1):
                        quad = [
                            nid(i, j,     k),
                            nid(i, j + 1, k),
                            nid(i, j + 1, k + 1),
                            nid(i, j,     k + 1),
                        ]
                        quads.append(quad)
            elif axis == "j":
                j = idx
                for k in range(nz - 1):
                    for i in range(nx - 1):
                        quad = [
                            nid(i,     j, k),
                            nid(i + 1, j, k),
                            nid(i + 1, j, k + 1),
                            nid(i,     j, k + 1),
                        ]
                        quads.append(quad)
            elif axis == "k":
                k = idx
                for j in range(ny - 1):
                    for i in range(nx - 1):
                        quad = [
                            nid(i,     j,     k),
                            nid(i + 1, j,     k),
                            nid(i + 1, j + 1, k),
                            nid(i,     j + 1, k),
                        ]
                        quads.append(quad)
            else:
                raise ValueError(f"Unknown axis '{axis}' for boundary '{wall_name}'.")

            oriented_quads = []
            desired_normal = desired_normal / np.linalg.norm(desired_normal)
            for quad in quads:
                p1, p2, p3 = points[np.array(quad[:3]) - 1]
                normal = np.cross(p2 - p1, p3 - p1)
                if np.dot(normal, desired_normal) < 0:
                    quad = [quad[0], quad[3], quad[2], quad[1]]
                oriented_quads.append(quad)

            quad_count = len(oriented_quads)
            quad_tags = np.arange(next_elem_tag, next_elem_tag + quad_count, dtype=np.int64)
            next_elem_tag += quad_count

            model.mesh.addElements(
                2,
                surface_tag,
                [quad_type],
                [quad_tags],
                [np.array(oriented_quads, dtype=np.int64).ravel()]
            )

            phys_tag = model.addPhysicalGroup(2, [surface_tag])
            model.setPhysicalName(2, phys_tag, wall_name)

        vol_phys = model.addPhysicalGroup(3, [volume_tag])
        model.setPhysicalName(3, vol_phys, volume_name)

        # Optional: export point and cell data as Gmsh views
        if getattr(grid, "point_values", None):
            for name, values in grid.point_values.items():
                data = np.asarray(values, order="F").ravel(order="F")
                if data.size != total_nodes:
                    raise ValueError(f"Point data '{name}' does not match node count.")
                view_tag = gmsh.view.add(name)
                gmsh.view.addModelData(
                    view_tag, 0, model_name, "NodeData",
                    node_tags, data.tolist(), 0.0, numComponents=1
                )

        if getattr(grid, "cell_values", None):
            total_hexes = len(volume_elements)
            for name, values in grid.cell_values.items():
                data = np.asarray(values, order="F").ravel(order="F")
                if data.size != total_hexes:
                    raise ValueError(f"Cell data '{name}' does not match element count.")
                view_tag = gmsh.view.add(name)
                gmsh.view.addModelData(
                    view_tag, 0, model_name, "ElementData",
                    vol_elem_tags, data.tolist(), 0.0, numComponents=1
                )

        if file_path:
            gmsh.write(file_path)

        return model

    finally:
        if auto_finalize and own_session and gmsh.isInitialized():
            gmsh.finalize()