from typing import cast

import numpy as np
import trimesh
from skimage import measure

from brain_region_database.nifti import NiftiImage, Zooms


def nifti_to_polyhedralsurface(image: NiftiImage, simplify: bool = False, decimate_factor: float = 0.5) -> str:
    """
    Convert a NIfTI mask to PostGIS POLYHEDRALSURFACE Z.
    """

    data = image.get_fdata()
    mask = data.astype(bool)

    header = image.header
    if hasattr(header, 'get_zooms'):
        zooms = cast(Zooms, header.get_zooms())
    else:
        zooms = (1.0, 1.0, 1.0)

    print(f"Image shape: {data.shape}")
    print(f"Voxel dimensions: {zooms}")
    print(f"Mask voxels: {np.sum(mask)} / {mask.size}")

    verts, faces = extract_surface_marching_cubes(mask, zooms, image.affine)

    if simplify and len(faces) > 10000:
        verts, faces = simplify_mesh(verts, faces, decimate_factor)

    wkt = mesh_to_polyhedralsurface(verts, faces)

    return wkt


def extract_surface_marching_cubes(
    mask: np.ndarray,
    zooms: Zooms,
    affine: np.ndarray,
    level: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract surface using marching cubes with proper coordinate transformation.
    """

    verts, faces, normals, values = measure.marching_cubes(
        mask.astype(float),
        level=level,
        spacing=zooms,
        allow_degenerate=False
    )

    verts = apply_affine_transform(verts, affine)

    return verts, faces


def apply_affine_transform(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply NIfTI affine transform to convert voxel coordinates to world coordinates.
    """
    ones = np.ones((vertices.shape[0], 1))
    verts_homogeneous = np.hstack([vertices, ones])

    verts_transformed = (affine @ verts_homogeneous.T).T

    return verts_transformed[:, :3]


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    factor: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simplify mesh to reduce polygon count while preserving shape.
    """

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    target_faces = int(len(faces) * factor)

    simplified = mesh.simplify_quadric_decimation(target_faces)

    return simplified.vertices, simplified.faces


def mesh_to_polyhedralsurface(vertices: np.ndarray, faces: np.ndarray) -> str:
    """
    Convert mesh to PostGIS POLYHEDRALSURFACE Z WKT format.
    """

    polygons: list[str] = []

    for face in faces:
        face_vertices = vertices[face]

        if len(face_vertices) > 0:
            closed_face = np.vstack([face_vertices, face_vertices[0:1]])
        else:
            continue

        coords_str = ", ".join(f"({x:.6f} {y:.6f} {z:.6f})" for x, y, z in closed_face)
        polygons.append(f"(({coords_str}))")

    return f"POLYHEDRALSURFACE Z ({', '.join(polygons)})"
