import trimesh
from geometric import get_folded_mesh_points
import warnings
import numpy as np
from scipy.spatial import distance
from copy import deepcopy
import sys


def shaped_hausdorff_distance_rewarder(target_file, count_interior, board_length, target_transform=None, **kwargs):
    board_size = int((board_length+1)**2)
    target_mesh = trimesh.load(target_file)

    def scale(mesh,auto_scaling=False):
        scale=1.0
        if auto_scaling:
            diff = 1.6*mesh.area-board_size
            while abs(diff)>board_size*0.2:
                if diff>0:
                    scale -= 0.1
                else:
                    scale += 0.05
                v = scale*mesh.vertices
                f = mesh.faces
                mesh = trimesh.Trimesh(v,f)
                diff = 1.6*mesh.area-board_size
        else:
            v = scale*mesh.vertices
            f = mesh.faces
            mesh = trimesh.Trimesh(v,f)
        return mesh

    target_mesh = scale(target_mesh)
    transform = target_transform
    target_mesh = target_mesh.apply_transform(np.asarray(transform))
    if kwargs["auto_mesh_transform"]:
        target_meshes = []
        target_sample_points = []
        for i in range(-board_length//2,board_length//2+1,1):
            new_mesh = deepcopy(target_mesh)
            mesh = auto_transform_mesh(new_mesh,i)
            target_meshes.append(mesh)
            sampled_points = trimesh.sample.sample_surface_even(
                mesh, 10000)[0].astype(np.float32)
            target_sample_points.append(sampled_points)
    else:
        target_sample_points = [trimesh.sample.sample_surface_even(target_mesh, 10000)[0].astype(np.float32)]
        target_meshes = [target_mesh]

    def shaped_hausdorff_distance(vtx_objs, points_3D, faces, reward_history, done):
        # === sample from folded_mesh
        idx = -vtx_objs[0].coords[1]+board_length//2 if len(target_sample_points)>1 else 0
        folded_mesh_points, points, mesh = get_folded_mesh_points(points_3D, faces, sample_points=done)
        if not count_interior:
            d = trimesh.proximity.signed_distance(target_meshes[idx], points)
            filter = np.where(d < 0)
            points = points[filter]
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            d_pat2tar = distance.directed_hausdorff(points, target_sample_points[idx])[0]
            if done:
                d_tar2pat = distance.directed_hausdorff(target_sample_points[idx], folded_mesh_points, seed=200)[0]
                rew = -max(d_pat2tar, d_tar2pat)
            else:
                rew = -d_pat2tar
        rew_min = sum(reward_history)
        if rew > rew_min:
            print(rew, rew_min)
            rew = 0
        else:
            rew = rew - rew_min
        return rew
    return shaped_hausdorff_distance

def auto_transform_mesh(mesh,origin_y_coord):
    verts = mesh.vertices
    verts_x,verts_y,verts_z = verts[:,0],verts[:,1],verts[:,2]
    radius = 0.0
    filter = np.zeros(verts_x.size).astype(bool)
    while not np.any(filter):
        radius += 0.1
        filter_x,filter_y = \
            np.where(abs(verts_x)<radius,True,False),\
            np.where(abs(verts_y-origin_y_coord)<radius,True,False)
        filter = np.logical_and(filter_x,filter_y)
    if np.any(filter):
        z_shift = np.amax(verts_z[filter])
    else:
        z_shift = 0.
    transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-z_shift],[0,0,0,1]])
    mesh.apply_transform(np.asarray(transform))
    return mesh

def bucket_rewarder(board_length, **kwargs):
    def bucket_reward(vtx_objs, points_3D, faces, reward_history, done):
        if done:
            outer_nodes = []
            for vtx in vtx_objs:
                if vtx.is_pseudo or not vtx.is_extended:
                    outer_nodes.append(vtx.name)
            outer_points = points_3D[np.asarray(outer_nodes)]
            min_z = max(np.min(outer_points[:, -1]), 0.0)
            if min_z == 0.0:
                return 0.
            max_z = np.max(outer_points[:, -1])
            points_in_plane = points_3D.copy()[4:]
            points_in_plane[:, -1] = 0.0
            norms = np.linalg.norm(points_in_plane, axis=-1)
            if np.max(norms) < 2 * np.min(norms) and 2 * min_z > max_z:
                return min_z
            else:
                return 0.
        return 0.
    return bucket_reward


def distance_between_colinear_planes(point_1, point_2, normalized_normal):
    vec = point_1 - point_2
    projected_vec = np.dot(vec, normalized_normal) * normalized_normal
    return np.linalg.norm(projected_vec)

def shelf_rewarder(board_length, **kwargs):
    min_area = 0.4
    def shelf_reward(vtx_objs, points_3D, faces, reward_history, done):
        if done:
            normals = []
            for face in faces:
                points = points_3D[face]
                vec1 = points[0] - points[1]
                vec2 = points[2] - points[1]
                normal = np.cross(vec1, vec2)
                normals.append(normal)
            def normalize(normal):
                normalized = normal / np.linalg.norm(normal)
                if normalized[0] < 0:
                    return normalized
                else:
                    return -normalized
            normalized_normals = list(map(normalize, normals))
            # in_plane_indices = [[] for _ in normals]
            # colinear_indices = [[] for _ in normals]
            areas = [np.linalg.norm(normal) * 0.5 for normal in normals]
            in_plane_sets = []
            set_areas = []
            set_normals = []
            for idx_i, area in enumerate(areas):
                if area < min_area: continue
                added_to_set = False
                for set_idx, set in enumerate(in_plane_sets):
                    if added_to_set: break
                    for idx_j in set:
                        if np.dot(normalized_normals[idx_i], normalized_normals[idx_j]) > 0.99999:
                            point_i = points_3D[faces[idx_i][0]]
                            point_j = points_3D[faces[idx_j][0]]
                            distance = distance_between_colinear_planes(point_i, point_j, normalized_normals[idx_i])
                            if distance < 0.5:
                                in_plane_sets[set_idx].append(idx_i)
                                set_areas[set_idx] += area
                                set_normals[set_idx] += normalized_normals[idx_i]
                                added_to_set = True
                                break
                if not added_to_set:
                    in_plane_sets.append([idx_i])
                    set_areas.append(area)
                    set_normals.append(normalized_normals[idx_i])
            parallel_planes = [[idx] for idx in range(len(set_normals))]
            for idx, normal in enumerate(set_normals):
                normalized_normal = normal / float(len(in_plane_sets[idx]))
                set_normals[idx] = normalized_normal
                for idx_j, second_normal in enumerate(set_normals[:idx]):
                    if np.dot(normalized_normal, second_normal) > 0.99999:
                        parallel_planes[idx_j].append(idx)
            max_shelf_area = 0
            for parallel_plane_list in parallel_planes:
                if len(parallel_plane_list) > 2:
                    shelf_areas = [set_areas[idx] for idx in parallel_plane_list]
                    max_shelf_area = max(min(shelf_areas), max_shelf_area)
            return max_shelf_area
        return 0
    return shelf_reward

def table_rewarder(**kwargs):
    def table_reward(vtx_objs, points_3D, faces, reward_history, done):
        if done:
            z_sorted_vertices = np.argsort(np.abs(points_3D[:, -1]))
            max_z_points_z = points_3D[z_sorted_vertices[-4:], -1]
            other_points_z = points_3D[z_sorted_vertices[:-4], -1]
            target_z = 2.5
            return -np.mean(np.abs(np.abs(max_z_points_z) - target_z)) - np.mean(np.abs(other_points_z))
        return 0
    return table_reward



def chair_rewarder(board_length, **kwargs):
    def chair_reward(vtx_objs, points_3D, faces, reward_history, done):
        points_too_far_out = np.any(np.abs(points_3D[:, :-1]) > 4.)
        if points_too_far_out:
            return -board_length
        if done:
            z_sorted_vertices = np.argsort(points_3D[:, -1])
            leg_points = points_3D[z_sorted_vertices[-3:]]
            leg_points_z = leg_points[:, -1]
            leg_points_y = leg_points[:, 1]
            legs_not_on_floor = np.any(np.abs(leg_points_z - leg_points_z[0]) > 0.1)
            legs_only_on_one_side = np.all(leg_points_y > -0.1) or np.all(leg_points_y < 0.1)
            if legs_not_on_floor or legs_only_on_one_side:
                return -board_length / 2.

            backrest_points = []
            for point in points_3D[z_sorted_vertices]:
                if point[-1] >= 0:
                    break
                backrest_points.append(point)
            if not backrest_points:
                backrest_points.append(points_3D[z_sorted_vertices[0]])
            backrest_points = np.asarray(backrest_points)
            target_z_legs = 4.
            target_y_legs = 2.1
            target_z_backrest = -4.
            target_y_backrest = 2.1
            leg_loss = np.mean(np.abs(leg_points_z - target_z_legs)) + \
                       np.mean(np.abs(np.abs(leg_points_y) - target_y_legs))
            backrest_loss = np.abs(backrest_points[0, -1] - target_z_backrest) + \
                            np.mean(np.abs(np.abs(backrest_points[:, 1]) - target_y_backrest))
            rew = - leg_loss - backrest_loss
            return rew
        return 0
    return chair_reward


def packaging_rewarder(board_length, **kwargs):
    board_size = int((board_length+1)**2)

    def packaging_reward(vtx_objs, points_3D, faces, reward_history, done):
        center = np.mean(points_3D, axis=0)
        norm = np.max(np.linalg.norm(points_3D - center, axis=-1))
        volume_proxy = norm ** 3
        mesh = trimesh.Trimesh(points_3D, faces)
        ratio = mesh.area / volume_proxy
        previous_ratio = sum(reward_history)
        return ratio - previous_ratio

    return packaging_reward

#wrapper function
def reward_wrapper(reward_function):
    def wrapped_reward_function(vtx_objs, all_3D_points, triangle_inequality_holds, **kwargs):
        all_3D_points = all_3D_points.swapaxes(0,1)
        if np.any(np.isnan(all_3D_points[triangle_inequality_holds])):
            print('nans', all_3D_points)
        rewards = -np.inf*np.ones(len(triangle_inequality_holds))
        rewards[np.array(triangle_inequality_holds)] = np.array(
            [reward_function(vtx_objs, points_3D,**kwargs) for points_3D in all_3D_points[triangle_inequality_holds]])
        rew = np.max(rewards)
        opt_idx = np.argmax(rewards)
        return rew,opt_idx
    return wrapped_reward_function
