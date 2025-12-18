from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from domain import Grid3D
from infrastructure.mesh import StructuredHexMesh3d
from infrastructure.topography.topographyUtils import un_rotate_points

@dataclass
class CellFieldExportSpec:
    """
    Describe how a single cell-centered value should be written to OpenFOAM.
    - name: key inside Grid3D.cell_values
    - dimensions: OpenFOAM physical dimensions, e.g. "[0 0 0 0 0 0 0]"
    - default_boundary: fallback boundary condition for every patch
    - boundary_overrides: map of patch name -> {"type": "...", "value": "..."} overrides
    - time_folder: sub-directory under the case (usually "0")
    - field_class: "auto" (infer), "volScalarField", or "volVectorField"
    """
    name: str
    dimensions: str = "[0 0 0 0 0 0 0]"
    default_boundary: str = "zeroGradient"
    boundary_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    time_folder: str = "0"
    field_class: Literal["auto", "volScalarField", "volVectorField"] = "auto"


class OpenFoamStructuredMeshExporter:
    """Export regular hexahedral meshes to OpenFOAM polyMesh files."""

    # Local vertex numbering (matches VTK structured grid convention)
    # v0: (0,0,0); v1: (1,0,0); v2: (1,1,0); v3: (0,1,0);
    # v4: (0,0,1); v5: (1,0,1); v6: (1,1,1); v7: (0,1,1)
    FACE_DEFINITIONS: Tuple[Tuple[str, Tuple[int, int, int, int]], ...] = (
        ("xmin", (0, 3, 7, 4)),
        ("xmax", (1, 5, 6, 2)),
        ("ymin", (0, 4, 5, 1)),
        ("ymax", (3, 2, 6, 7)),
        ("zmin", (0, 1, 2, 3)),
        ("zmax", (4, 5, 6, 7)),
    )

    DEFAULT_PATCHES: OrderedDict[str, Dict[str, str]] = OrderedDict(
        [
            ("zmin", {"name": "ground", "type": "wall"}),
            ("zmax", {"name": "top", "type": "patch"}),
            ("ymin", {"name": "ymin", "type": "patch"}),
            ("ymax", {"name": "ymax", "type": "patch"}),
            ("xmin", {"name": "xmin", "type": "patch"}),
            ("xmax", {"name": "xmax", "type": "patch"}),
        ]
    )

    def __init__(self, patch_map: Optional[Mapping[str, Mapping[str, str]]] = None):
        """
        Args:
            patch_map: Optional overrides for boundary tags.
                       Keys must be among {'xmin','xmax','ymin','ymax','zmin','zmax'}.
                       Example: {"zmax": {"name": "atmosphere", "type": "patch"}}
        """
        self.patch_map = self._build_patch_map(patch_map)

    def export(
            self,
            mesh_or_grid: Union[StructuredHexMesh3d, Grid3D],
            case_dir: Union[str, Path],
            *,
            wind_direction: Optional[float] = None,
            center_x: Optional[float] = None,
            center_y: Optional[float] = None,
            apply_rotation: bool = True,
            cell_field_specs: Optional[Sequence[CellFieldExportSpec]] = None,
        ) -> None:
        """
        Write the OpenFOAM polyMesh for the provided mesh/grid into case_dir/constant/polyMesh.
        """
        grid, wind_direction, center_x, center_y = self._resolve_inputs(
            mesh_or_grid,
            wind_direction,
            center_x,
            center_y,
        )

        case_dir = Path(case_dir)
        poly_mesh_dir = case_dir / "constant" / "polyMesh"
        poly_mesh_dir.mkdir(parents=True, exist_ok=True)

        points = self._build_points(grid, wind_direction, center_x, center_y, apply_rotation)
        faces, owners, neighbour_internal, boundary_blocks = self._build_faces(grid, points)

        self._write_points(poly_mesh_dir / "points", points)
        self._write_faces(poly_mesh_dir / "faces", faces)
        self._write_owner(poly_mesh_dir / "owner", owners)
        self._write_neighbour(poly_mesh_dir / "neighbour", neighbour_internal)
        self._write_boundary(poly_mesh_dir / "boundary", boundary_blocks)

        if cell_field_specs:
            self._write_cell_fields(grid, case_dir, cell_field_specs)


    def _write_cell_fields(
            self,
            grid: Grid3D,
            case_dir: Union[str, Path],
            specs: Sequence[CellFieldExportSpec],
        ) -> None:
        case_dir = Path(case_dir)
        nx, ny, nz = grid.X.shape
        cell_shape = (nx - 1, ny - 1, nz - 1)

        for spec in specs:
            if spec.name not in grid.cell_values:
                raise KeyError(f"Grid3D.cell_values is missing '{spec.name}'.")
            raw = np.asarray(grid.cell_values[spec.name])
            flat = self._flatten_cell_field(raw, cell_shape)
            n_components = 1 if flat.ndim == 1 else flat.shape[1]

            field_class = spec.field_class
            if field_class == "auto":
                field_class = "volScalarField" if n_components == 1 else "volVectorField"

            time_folder = case_dir / spec.time_folder
            time_folder.mkdir(parents=True, exist_ok=True)

            self._write_vol_field(
                filepath=time_folder / spec.name,
                field_class=field_class,
                field_name=spec.name,
                dimensions=spec.dimensions,
                values=flat,
                default_boundary=spec.default_boundary,
                boundary_overrides=spec.boundary_overrides,
            )

    @staticmethod
    def _flatten_cell_field(
            data: np.ndarray,
            cell_shape: Tuple[int, int, int],
        ) -> np.ndarray:
        if data.shape[:3] != cell_shape:
            raise ValueError(
                f"Cell field has spatial shape {data.shape[:3]}, expected {cell_shape}."
            )

        if data.ndim == 3:
            return np.reshape(data, -1, order="F")

        if data.ndim == 4:
            comps = data.shape[3]
            if comps not in (3,):
                raise ValueError(
                    "Only 3-component vector cell fields are supported (got "
                    f"{comps})."
                )
            return np.reshape(data, (-1, comps), order="F")

        raise ValueError("Cell field arrays must be 3D (scalar) or 4D (vector).")   

    def _write_vol_field(
            self,
            filepath: Path,
            *,
            field_class: str,
            field_name: str,
            dimensions: str,
            values: np.ndarray,
            default_boundary: str,
            boundary_overrides: Mapping[str, Mapping[str, str]],
        ) -> None:
        is_scalar = values.ndim == 1
        component_tag = "scalar" if is_scalar else "vector"
        value_formatter = (
            lambda v: f"{v:.12e}"
            if is_scalar
            else f"({v[0]:.12e} {v[1]:.12e} {v[2]:.12e})"
        )

        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header(field_class, field_name))
            fh.write(f"\ndimensions      {dimensions};\n")
            fh.write(f"internalField   nonuniform List<{component_tag}>\n")
            fh.write(f"{values.shape[0]}\n(\n")
            for val in values:
                fh.write(f"    {value_formatter(val)}\n")
            fh.write(")\n;\n\n")
            fh.write("boundaryField\n{\n")
            for tag, patch_cfg in self.patch_map.items():
                patch_name = patch_cfg["name"]
                override = boundary_overrides.get(patch_name, {})
                bc_type = override.get("type", default_boundary)
                fh.write(f"    {patch_name}\n    {{\n")
                fh.write(f"        type            {bc_type};\n")
                if "value" in override:
                    fh.write(f"        value           {override['value']};\n")
                fh.write("    }\n")
            fh.write("}\n")    
    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

    def _resolve_inputs(
        self,
        mesh_or_grid: Union[StructuredHexMesh3d, Grid3D],
        wind_direction: Optional[float],
        center_x: Optional[float],
        center_y: Optional[float],
    ) -> Tuple[Grid3D, float, float, float]:
        if isinstance(mesh_or_grid, StructuredHexMesh3d):
            grid = mesh_or_grid.mesh
            cfg = getattr(mesh_or_grid.config, "ground_mesh", None)
        elif isinstance(mesh_or_grid, Grid3D):
            grid = mesh_or_grid
            cfg = None
        else:
            raise TypeError(
                "mesh_or_grid must be a StructuredHexMesh3d or Grid3D instance."
            )

        if cfg is not None:
            wind_direction = cfg.wind_direction if wind_direction is None else wind_direction
            center_x = cfg.center_x if center_x is None else center_x
            center_y = cfg.center_y if center_y is None else center_y

        return grid, float(wind_direction or 0.0), float(center_x or 0.0), float(center_y or 0.0)

    def _build_points(
        self,
        grid: Grid3D,
        wind_direction: float,
        center_x: float,
        center_y: float,
        apply_rotation: bool,
    ) -> np.ndarray:
        shape = grid.X.shape
        if shape != grid.Y.shape or shape != grid.Z.shape or len(shape) != 3:
            raise ValueError("Grid3D.X, Y, Z must share the same 3D shape.")

        x = np.asarray(grid.X, dtype=np.float64)
        y = np.asarray(grid.Y, dtype=np.float64)
        z = np.asarray(grid.Z, dtype=np.float64)

        points = np.column_stack(
            (
                x.ravel(order="F"),
                y.ravel(order="F"),
                z.ravel(order="F"),
            )
        )

        if apply_rotation and not np.isclose(wind_direction, 0.0):
            un_rotate_points(
                points[:, :2],
                angle_deg=wind_direction,
                center_x=center_x,
                center_y=center_y,
                inplace=True,
            )

        return points

    def _build_faces(
        self,
        grid: Grid3D,
        points: np.ndarray,
    ) -> Tuple[List[Tuple[int, ...]], List[int], List[int], List[Dict[str, int]]]:
        nx, ny, nz = grid.X.shape
        nx_cells, ny_cells, nz_cells = nx - 1, ny - 1, nz - 1

        if min(nx_cells, ny_cells, nz_cells) <= 0:
            raise ValueError("Grid must contain at least one hexahedral cell along each axis.")

        def pid(i: int, j: int, k: int) -> int:
            return i + j * nx + k * nx * ny

        def cid(i: int, j: int, k: int) -> int:
            return i + j * nx_cells + k * nx_cells * ny_cells

        faces: List[Tuple[int, ...]] = []
        owners: List[int] = []
        neighbours: List[int] = []
        patch_labels: List[Optional[str]] = []
        face_lookup: Dict[Tuple[int, ...], int] = {}

        for k in range(nz_cells):
            for j in range(ny_cells):
                for i in range(nx_cells):
                    cell_id = cid(i, j, k)
                    cell_vertices = (
                        pid(i, j, k),
                        pid(i + 1, j, k),
                        pid(i + 1, j + 1, k),
                        pid(i, j + 1, k),
                        pid(i, j, k + 1),
                        pid(i + 1, j, k + 1),
                        pid(i + 1, j + 1, k + 1),
                        pid(i, j + 1, k + 1),
                    )
                    cell_center = points[np.array(cell_vertices)].mean(axis=0)

                    for face_tag, local_corners in self.FACE_DEFINITIONS:
                        face_vertex_ids = [cell_vertices[idx] for idx in local_corners]
                        face_vertex_ids = self._orient_face(face_vertex_ids, points, cell_center)

                        key = tuple(sorted(face_vertex_ids))
                        if key not in face_lookup:
                            face_id = len(faces)
                            face_lookup[key] = face_id
                            faces.append(tuple(face_vertex_ids))
                            owners.append(cell_id)
                            neighbours.append(-1)
                            patch_labels.append(face_tag)
                        else:
                            face_id = face_lookup[key]
                            if neighbours[face_id] != -1:
                                raise ValueError("Non-manifold face detected (shared by >2 cells).")
                            neighbours[face_id] = cell_id
                            patch_labels[face_id] = None  # Interior faces should not belong to a patch.

        # Separate interior / boundary and reorder so internal faces come first.
        internal_ids = [idx for idx, nb in enumerate(neighbours) if nb != -1]
        boundary_ids = [idx for idx, nb in enumerate(neighbours) if nb == -1]

        missing_tags = {
            patch_labels[idx] for idx in boundary_ids if patch_labels[idx] not in self.patch_map
        }
        if missing_tags:
            raise ValueError(f"No patch mapping defined for boundary tags: {missing_tags}")

        ordered_faces: List[Tuple[int, ...]] = []
        ordered_owners: List[int] = []

        # Interior first (order preserved)
        ordered_faces.extend(faces[idx] for idx in internal_ids)
        ordered_owners.extend(owners[idx] for idx in internal_ids)
        neighbour_internal = [neighbours[idx] for idx in internal_ids]

        # Boundary faces grouped patch-by-patch to guarantee contiguous ranges.
        boundary_blocks: List[Dict[str, int]] = []
        current_start = len(ordered_faces)

        for tag, patch_cfg in self.patch_map.items():
            ids = [idx for idx in boundary_ids if patch_labels[idx] == tag]
            ordered_faces.extend(faces[idx] for idx in ids)
            ordered_owners.extend(owners[idx] for idx in ids)

            boundary_blocks.append(
                {
                    "name": patch_cfg["name"],
                    "type": patch_cfg["type"],
                    "nFaces": len(ids),
                    "startFace": current_start,
                }
            )
            current_start += len(ids)

        if len(ordered_faces) != len(faces):
            raise RuntimeError("Face reordering failed (counts do not match).")

        return ordered_faces, ordered_owners, neighbour_internal, boundary_blocks

    @staticmethod
    def _orient_face(
        face_vertex_ids: List[int],
        points: np.ndarray,
        cell_center: np.ndarray,
    ) -> List[int]:
        coords = points[np.array(face_vertex_ids)]
        normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        face_center = coords.mean(axis=0)
        if np.dot(normal, face_center - cell_center) < 0.0:
            return list(reversed(face_vertex_ids))
        return face_vertex_ids

    def _build_patch_map(
        self,
        overrides: Optional[Mapping[str, Mapping[str, str]]],
    ) -> OrderedDict[str, Dict[str, str]]:
        mapping: OrderedDict[str, Dict[str, str]] = OrderedDict(
            (tag, cfg.copy()) for tag, cfg in self.DEFAULT_PATCHES.items()
        )
        if overrides:
            for tag, cfg in overrides.items():
                if tag not in mapping:
                    raise KeyError(f"Unknown boundary tag '{tag}'.")
                merged = mapping[tag].copy()
                merged.update(cfg)
                mapping[tag] = merged
        return mapping

    # -------------------------------------------------------------------------
    # File writers
    # -------------------------------------------------------------------------

    @staticmethod
    def _foam_header(foam_class: str, obj: str) -> str:
        return (
            "/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
            "|  \\\\    /   O peration     | Version:  v2012                                  |\n"
            "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n"
            "{\n"
            f"    version     2.0;\n"
            f"    format      ascii;\n"
            f"    class       {foam_class};\n"
            f"    location    \"constant/polyMesh\";\n"
            f"    object      {obj};\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        )

    def _write_points(self, filepath: Path, points: np.ndarray) -> None:
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header("vectorField", "points"))
            fh.write(f"\n{len(points)}\n(\n")
            for x, y, z in points:
                fh.write(f"    ({x:.12e} {y:.12e} {z:.12e})\n")
            fh.write(")\n")

    def _write_faces(self, filepath: Path, faces: Sequence[Tuple[int, ...]]) -> None:
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header("faceList", "faces"))
            fh.write(f"\n{len(faces)}\n(\n")
            for face in faces:
                indices = " ".join(str(idx) for idx in face)
                fh.write(f"    {len(face)}({indices})\n")
            fh.write(")\n")

    def _write_owner(self, filepath: Path, owners: Sequence[int]) -> None:
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header("labelList", "owner"))
            fh.write(f"\n{len(owners)}\n(\n")
            for owner in owners:
                fh.write(f"    {owner}\n")
            fh.write(")\n")

    def _write_neighbour(self, filepath: Path, neighbours_internal: Sequence[int]) -> None:
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header("labelList", "neighbour"))
            fh.write(f"\n{len(neighbours_internal)}\n(\n")
            for neighbour in neighbours_internal:
                fh.write(f"    {neighbour}\n")
            fh.write(")\n")

    def _write_boundary(
        self,
        filepath: Path,
        boundary_blocks: Sequence[Dict[str, int]],
    ) -> None:
        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._foam_header("polyBoundaryMesh", "boundary"))
            fh.write(f"\n{len(boundary_blocks)}\n(\n")
            for block in boundary_blocks:
                fh.write(f"{block['name']}\n")
                fh.write("{\n")
                fh.write(f"    type            {block['type']};\n")
                fh.write("    inGroups        0();\n")
                fh.write(f"    nFaces          {block['nFaces']};\n")
                fh.write(f"    startFace       {block['startFace']};\n")
                fh.write("}\n")
            fh.write(")\n")