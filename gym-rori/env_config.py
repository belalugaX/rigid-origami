# env initial configs

import numpy as np
from single_vertex import SingleVertex
from kinematic_model_num import Rz,Rx,planar_angle
from copy import deepcopy

def create_base(env,coords,dir):
    d = coords[1]
    if env.symmetry_axis == "point":
        quad_config(env,d=d,dir=dir)
    else:
        simple_config(env,d=d,dir=dir)

def place_source(env, pseudo_v, dir):

    if env.cross_axis_symmetry:
        # take all
        mirrored_pseudo_vertices = []
        for pseudo, msg in env.pseudo_vertices.values():
            mirrored_pseudo_vertices.append(pseudo)
        env.pseudo_vertices = {}
        zero_coords = np.zeros(2)
    else:
        mirrored_pseudo_vertices = [pseudo_v]
        coords = env.vertex_objs[pseudo_v].coords
        if env.symmetry_axis == "point":
            # for vtx in env.vertex_objs:
            #     print(vtx.name, vtx.coords, vtx.is_pseudo)
            # print(env.points_2D, len(env.points_2D))
            # print('----')
            mirrored_pseudo_vertices.append(env.points_2D.index([-coords[0], -coords[1]]))
            zero_coords = np.zeros(2)
        elif env.quad_base:
            zero_coords = np.zeros(2)
        else:
            zero_coords = np.asarray(env.vertex_objs[0].coords)
        for pseudo_vtx in mirrored_pseudo_vertices:
            del env.pseudo_vertices[env.vertex_objs[pseudo_vtx].key]
        # for key in ['x+', 'x-', 'y+', 'y-']:
        #     if key in env.pseudo_vertices and env.pseudo_vertices[key] in mirrored_pseudo_vertices:
        #         del env.pseudo_vertices[key]

    for pseudo_vtx in mirrored_pseudo_vertices:
        pseudo_vtx_obj = env.vertex_objs[pseudo_vtx]
        new_vtx_obj = SingleVertex(env, name=pseudo_vtx, coords=pseudo_vtx_obj.coords, mother_vertex_name=-1,
                                   driving_angle=[0, 0], rbm=dir, new_vertex=False)
        new_vtx_obj.coords_3D = pseudo_vtx_obj.coords_3D
        new_vtx_obj.is_extended = True

        a = pseudo_vtx_obj.neighbor
        vtx_a = env.vertex_objs[a]
        if pseudo_vtx_obj.coords[0] == 0:
            b_coords = [-vtx_a.coords[0], vtx_a.coords[1]]
        else:
            b_coords = [vtx_a.coords[0], -vtx_a.coords[1]]
        b = env.points_2D.index(b_coords)
        vtx_b = env.vertex_objs[b]
        if np.cross(np.asarray(b_coords) - zero_coords, np.asarray(vtx_a.coords) - zero_coords) < 0:
            # switch
            tmp_name, tmp_obj = a, vtx_a
            a, vtx_a = b, vtx_b
            b, vtx_b = tmp_name, tmp_obj

        new_vtx_obj.surrounding_vertices = [a, b]
        new_edges = [[pseudo_vtx, a], [pseudo_vtx, b]]
        env.edge_list.extend(new_edges)
        new_vtx_obj.update_adjacency_state(new_edges)
        vtx_a.surrounding_vertices.append(pseudo_vtx)
        vtx_b.surrounding_vertices.insert(0, pseudo_vtx)
        vtx_a.dihedral_angles = np.vstack(
            (np.array(vtx_a.dihedral_angles).T, dir * env.psi_opt)).T
        vtx_b.dihedral_angles = np.vstack(
            (dir * env.psi_opt, np.array(vtx_b.dihedral_angles).T)).T
        vtx_a.update_adjacency_state([new_edges[0]])
        vtx_b.update_adjacency_state([new_edges[1]])

        vec_1 = env.vertex_objs[vtx_b.surrounding_vertices[1]].coords - vtx_b.coords
        vec_2 = new_vtx_obj.coords-vtx_b.coords
        final_sector = planar_angle(vec_1,vec_2)
        num_valid_folds = np.count_nonzero(env.triangle_inequality_holds)
        init_rot = [None]*len(env.psi_opt)
        diff = len(env.psi_opt) - num_valid_folds
        for i in range(num_valid_folds):
            final_trans = Rx(-vtx_b.dihedral_angles[i+diff,1]).dot(Rz(final_sector))
            init_rot[i+diff] = vtx_b.init_rot[i+diff].dot(final_trans)
        vtx_b.init_rot = init_rot

        env.vertex_objs[pseudo_vtx] = new_vtx_obj



def quad_config(env,d=0,dir=1):
    env.quad_base = True
    if d == 0: d = int(env.TS/2)
    psi_opt = dir*env.psi_opt
    env.vertex_objs.append(SingleVertex(env,0,[0,d],-1,0,rbm=dir))                    # source vertex
    env.vertex_objs.append(SingleVertex(env,1,[-d,0],-1,0))                   # source vertex
    env.vertex_objs.append(SingleVertex(env,2,[0,-d],-1,0))                    # source vertex
    env.vertex_objs.append(SingleVertex(env,3,[d,0],-1,0))                     # source vertex
    env.vertex_objs.append(SingleVertex(env,4,[d,-d],3,
                                        psi_opt,init_rot=Rz(np.pi/2),units=[np.pi])) #np.pi/2
    env.vertex_objs.append(SingleVertex(env,5,[d,d],0,
                                        psi_opt,init_rot=Rz(-np.pi),units=[np.pi/2])) #-np/pi
    env.vertex_objs.append(SingleVertex(env,6,[-d,d],1,
                                        psi_opt,init_rot=Rz(-np.pi/2),units=[np.pi/2])) #-np.pi/2
    env.vertex_objs.append(SingleVertex(env,7,[-d,-d],2,
                                        psi_opt,init_rot=Rz(0),units=[np.pi/2]))
    env.vertex_objs[0].is_extended = True
    env.vertex_objs[1].is_extended = True
    env.vertex_objs[2].is_extended = True
    env.vertex_objs[3].is_extended = True
    env.vertex_objs[0].surrounding_vertices = [6,5]
    env.vertex_objs[1].surrounding_vertices = [7,6]
    env.vertex_objs[2].surrounding_vertices = [4,7]
    env.vertex_objs[3].surrounding_vertices = [5,4]
    env.vertex_objs[4].surrounding_vertices = [3,2]
    env.vertex_objs[5].surrounding_vertices = [0,3]
    env.vertex_objs[6].surrounding_vertices = [1,0]
    env.vertex_objs[7].surrounding_vertices = [2,1]
    env.vertex_objs[0].dihedral_angles = [0,0]
    env.vertex_objs[1].dihedral_angles = [0,0]
    env.vertex_objs[2].dihedral_angles = [0,0]
    env.vertex_objs[3].dihedral_angles = [0,0]
    env.vertex_objs[4].dihedral_angles = np.vstack((psi_opt,psi_opt)).T.reshape((-1,2))
    env.vertex_objs[5].dihedral_angles = np.vstack((psi_opt,psi_opt)).T.reshape((-1,2))
    env.vertex_objs[6].dihedral_angles = np.vstack((psi_opt,psi_opt)).T.reshape((-1,2))
    env.vertex_objs[7].dihedral_angles = np.vstack((psi_opt,psi_opt)).T.reshape((-1,2))
    env.edge_list = [[0,6],[0,5],[1,7],[1,6],[2,4],[2,7],[3,5],[3,4]]
    for vtx in env.vertex_objs:
        vtx.update_adjacency_state(env.edge_list)
    env.vertex_list = [0,1,2,3,4,5,6,7]
    env.triangles = [[5, 6, 7], [4, 5, 7]]


def single_config(env):
    psi_opt = dir*env.psi_opt
#    env.vertex_objs.append(SingleVertex(env,0,[0,0],0,0))                 # source vertex
#    env.vertex_objs.append(SingleVertex(env,1,[d,0],0,env.psi,init_rot=Rz(np.pi)))
    env.vertex_objs.append(SingleVertex(env,0,[0,10],0,0))                 # source vertex
    env.vertex_objs.append(SingleVertex(
        env,1,[0,7],0,
        psi_opt.reshape((-1,1)),
        init_rot=list(np.matmul(Rz(np.pi/2),np.array(list(map(Rx,psi_opt/2)))))
        ))
    env.vertex_objs[0].surrounding_vertices = [1]
    env.vertex_objs[0].dihedral_angles = psi_opt
    env.vertex_objs[0].is_extended = True
    env.vertex_list = [0,1]
    env.edge_list = [[0,1]]                                                    # edges (v_from,v_to)
    for vtx in env.vertex_objs:
        vtx.update_adjacency_state(env.edge_list)


def simple_config(env,d=1,dir=1): #default case: d=0
    # === single line, two vertices to extend
    psi_opt = dir*env.psi_opt
    env.vertex_objs.append(SingleVertex(env, 0, [0, d], 0, 0, rbm=dir))                 # source vertex
    env.vertex_objs.append(SingleVertex(
        env,1,[1,d],0,
        np.array(psi_opt).reshape((-1,1)),
        init_rot=Rz(np.pi)
        ))
    env.vertex_objs.append(SingleVertex(
        env,2,[-1,d],0,
        np.array(psi_opt).reshape((-1,1)),
        init_rot=list(map(Rx,psi_opt))
        ))
    env.vertex_objs[0].surrounding_vertices = [1, 2]
    env.vertex_objs[0].dihedral_angles = [psi_opt, psi_opt]
    env.vertex_objs[0].is_extended = True
    env.vertex_list = [0,1,2]
    env.edge_list = [[0,1],[0,2]]                                          # edges (v_from,v_to)
    for vtx in env.vertex_objs:
        vtx.update_adjacency_state(env.edge_list)


def simple_vert_config(env):
    env.vertex_objs.append(SingleVertex(env,0,[0,0],0,0))                 # source vertex
    env.vertex_objs.append(SingleVertex(
        env,1,[0,1],0,
        np.array(psi_opt).reshape((-1,1)),
        init_rot=list(np.matmul(Rz(-np.pi/2),np.array(list(map(Rx,psi_opt/2)))))
        ))
    env.vertex_objs.append(SingleVertex(
        env,2,[0,-1],0,
        np.array(psi_opt).reshape((-1,1)),
        init_rot=list(np.matmul(Rz(np.pi/2),np.array(list(map(Rx,psi_opt/2)))))
        ))
    env.vertex_objs[0].surrounding_vertices = [1,2]
    env.vertex_objs[0].dihedral_angles = [psi_opt,psi_opt]
    env.vertex_objs[0].is_extended = True
    env.vertex_list = [0,1,2]
    env.edge_list = [[0,1],[0,2]]                                          # edges (v_from,v_to)
    for vtx in env.vertex_objs:
        vtx.update_adjacency_state(env.edge_list)
