import trimesh
from trimesh.sample import sample_surface_even
import logging


def get_folded_mesh_points(points_3D, faces, sample_points=True, samples=10000):
    logging.getLogger("trimesh").setLevel(logging.ERROR)
    p, tri_faces = points_3D, faces
    sampled_mesh_points = []
    folded_mesh = None
    if sample_points:
        folded_mesh = trimesh.Trimesh(p, tri_faces)
        if tri_faces != []:
            sampled_mesh_points = sample_surface_even(folded_mesh, samples)[0]
        else:
            # if no faces exist, but only edges
            sampled_mesh_points = p
    return sampled_mesh_points, p, folded_mesh
