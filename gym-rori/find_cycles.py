# find cycles in origami graph

import networkx as nx
from rules import search_bnd_vtx
from kinematic_model_num import planar_angle
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import multiprocessing

def find_cycles(env,vtx2ext_name=0):
    """
    find cycles, i.e. faces in graph
    """
    # raise NotImplementedError # hack to catch find_cycles
    v_objs = env.vertex_objs
    new_faces = []
    faces = []
    # virtual closing edges between extendable vertices
    face_boundary_edges = []

    for v in v_objs:
        if v.is_pseudo: continue
        if v.is_extended:
            c = 0
            for i in range(len(v.surrounding_vertices)):
                # --- clockwise cycle
                f = [v.name]
                v_0 = v
                v_1 = v_objs[v.surrounding_vertices[i]]
                f.append(v_1.name)
                while True:
                    c+=1
                    if c>50:
                        print("CAUGHT IN THE LOOP", "vtx:",v_0.name,"v_i",v_1.name,"vn",v_next,"a:",a,v_1.surrounding_vertices)
                        env.fold_pattern(0.1,0,1,silent=0)
                        break
                    pos = v_1.surrounding_vertices.index(v_0.name)
                    a = -1  # search bnd_vtx by default
                    if len(v_1.surrounding_vertices)>1:
                        if pos < (len(v_1.surrounding_vertices)-1):
                            v_next = v_1.surrounding_vertices[pos+1]
                            v_n = v_objs[v_next]
                            vec_1, vec_2 = get_edge_vectors(v_0,v_1,v_n)
                            a = planar_angle(vec_1,vec_2)
                        elif v_1.is_extended:
                            v_next = v_1.surrounding_vertices[0]
                            v_n = v_objs[v_next]
                            vec_1, vec_2 = get_edge_vectors(v_0,v_1,v_n)
                            a = planar_angle(vec_1,vec_2)
                    if 0.0<a:
                        v_0 = v_1
                        v_1 = v_objs[v_next]
                        if v_1.name == v.name:
                            break
                        f.append(v_1.name)
                    else:
                        v_0,_ = search_bnd_vtx(env,v_1,True)
                        face_boundary_edges.extend([(v_1.name,v_0),(v_0,v_1.name)])
                        v_0 = v_objs[v_0]
                        if v_0.name == v.name:
                            break

                        f.append(v_0.name)
                        v_1 = v_0.surrounding_vertices[0]
                        v_1 = v_objs[v_1]
                        if v_1.name == v.name:
                            break
                        f.append(v_1.name)
                faces.append(f)

    faces_sorted = []
    faces_out = []
    for f in faces:
        e = f.copy()
        e.sort()
        nn = True
        for fs in faces_sorted:
            if e == fs:
                nn=False
                break
        if nn:
            faces_sorted.append(e)
            faces_out.append(f)
            if vtx2ext_name in f: new_faces.append(f)

    env.faces = deepcopy(faces_out)
    env.new_faces = deepcopy(new_faces)
    env.face_boundary_edges = face_boundary_edges
    return faces_out

def get_edge_vectors(v0,v1,vn):
    return np.array(v0.coords)-np.array(v1.coords),\
        np.array(vn.coords)-np.array(v1.coords)
