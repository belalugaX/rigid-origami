from kinematic_model_num import planar_angle, compute_folded_unit
from symmetry_rules import map2symm_mask,map2symm_mask_y
import numpy as np

def get_init_mask(env):
    init_mask = np.zeros((env.board_length+1,env.board_length+1))
    init_mask[1:-1,env.half_board_length] = 1
    if env.symmetry_axis == "point":
        init_mask[env.half_board_length,env.half_board_length] = 0
    return init_mask.T.reshape((-1,))


def get_sel_action_mask(env):
    sel_action_mask = np.zeros(env.max_vertices, dtype=np.float32)
    if np.any(env.interim_state[:, :, -1]) or env.done:
        return np.append(sel_action_mask, [0, 0])
    else:
        for i, vtx in enumerate(env.vertex_objs):
            if vtx.is_pseudo:
                if env.allow_source_action and not vtx.inactive and vtx.coords[0]<=0 and vtx.coords[1]>=0:
                    sel_action_mask[i] = 1
                    if env.cross_axis_symmetry and abs(vtx.coords[0])<vtx.coords[1]:
                        sel_action_mask[i] = 0
                continue
            if not vtx.is_extended and not any(abs(vtx.coords) == env.half_board_length):
                # === adjust mask according to pattern symmetry config
                if env.symmetry:
                    if env.symmetry_axis=="point" and vtx.coords[0]<=0 and vtx.coords[1]>=0:
                        sel_action_mask[i] = 1
                        if env.cross_axis_symmetry and abs(vtx.coords[0])<vtx.coords[1]:
                            sel_action_mask[i] = 0
                    if env.symmetry_axis=="y-axis" and vtx.coords[0]<=0:
                        sel_action_mask[i] = 1
                    if env.symmetry_axis=="x-axis" and vtx.coords[1]>=0:
                        sel_action_mask[i] = 1
                else:
                    sel_action_mask[i] = 1

                # constrain locked-in vertices from selection for extension
                vtx_u_bnd,u_bnd_coords = search_bnd_vtx(env,vtx,True)
                v_u = vtx.surrounding_vertices[-1]
                v_u_vec = np.append(env.points_2D[v_u]-vtx.coords,0)
                u_bnd_vec = u_bnd_coords-np.append(vtx.coords,0)
                alpha_u = planar_angle(v_u_vec,u_bnd_vec)
                vtx_l_bnd,l_bnd_coords = search_bnd_vtx(env,vtx,False)
                v_l = vtx.surrounding_vertices[0]
                v_l_vec = np.append(env.points_2D[v_l]-vtx.coords,0)
                l_bnd_vec = l_bnd_coords-np.append(vtx.coords,0)
                alpha_l = planar_angle(l_bnd_vec,v_l_vec)
                if sel_action_mask[i] == 1 and (alpha_u<0 or alpha_l<0):
                    sel_action_mask[i] = 0

    # === set terminate action
    sel_action_mask[0] = 1 if len(env.decision_chain) > env.min_depth else 0

    # === set done key for select agent
    if np.all(sel_action_mask==0): env.done = True  # ["sel_agent"]
    return np.append(sel_action_mask, [1, 1])


# === allow source action
def mask_source_action(env,single_mask):
    if env.symmetry and np.count_nonzero(single_mask)>=3:
        # === find eligible boundary vertices
        x_max = -env.half_board_length
        boundary_vtx = env.vertex_objs[0]
        for vtx in env.vertex_objs:
            if vtx.is_pseudo: continue
            if (not vtx.is_extended) and vtx.coords[0]<0.1 and vtx.coords[1]>0.1:
                if boundary_vtx.name == 0:
                    boundary_vtx = vtx
                    x_max = vtx.coords[0]
                elif x_max<vtx.coords[0]<-0.1:
                    up_bnd_vtx,up_bnd_vtx_coords = search_bnd_vtx(env,vtx,True)
                    if up_bnd_vtx_coords[0]>0.1:
                        boundary_vtx = vtx
                        x_max = vtx.coords[0]
        _,up_bnd_vtx_coords = search_bnd_vtx(env,boundary_vtx,True)
        if up_bnd_vtx_coords[0]>0.1 and \
            not ([0,up_bnd_vtx_coords[1]] in env.points_2D):
            single_mask[boundary_vtx.board_pos[0],env.half_board_length] = 1
            env.source_boundary_vtx = boundary_vtx
    return single_mask


def get_ext_action_mask(env, v: int, stage=-1):

# ------------------------------------------------------------------------------
#
#   takes:      v (int): name of the "vertex to extend"
#
#   returns:    ext_action_mask: a np.dnarray
#
# ------------------------------------------------------------------------------
    # === init mask all zero
    mask = np.zeros(env.board_size,dtype=np.float32)

    if not np.any(env.interim_state[:,:,-1]) or env.sel_action==0 or env.done:
        return mask[:env.ext_act_space_size]
    else:
        # === set extension counter
        env.ext_cnt = len(env.ext_action_list)
        # === vertex2extend object
        vtx = env.vertex_objs[v]
        # === coordinates of vertex2extend
        v_coords = np.array(vtx.coords)
        # === vector with all coords on board (origin in board center)
        coords_arr = np.hstack((np.array(env.board_coords),np.zeros(len(env.board_coords)).reshape((-1,1))))

        # === intersection mask
        p_arr = np.array(env.points_2D)
        e_arr = np.vstack((np.array(env.edge_list),np.array(env.face_boundary_edges)))
        p_out_list = p_arr[e_arr[:,0]]  # point mapping from edges out
        p_in_list = p_arr[e_arr[:,1]]   # point mapping from edges in
        isec_mask = edge_intersect(v_coords,env.coords_list,p_out_list,p_in_list)

        # === vertex to extend coords to 3D
        v_coords = np.append(v_coords,0)
        # === derive boundary edge directions
        l_bnd_vec = np.append(env.points_2D[vtx.surrounding_vertices[0]],0) - v_coords
        u_bnd_vec = np.append(env.points_2D[vtx.surrounding_vertices[-1]],0) - v_coords

        # === first new vertex valid sector angle mask
        if env.ext_cnt == 0:
            # === update boundary vertices
            env.l_bnd_vtx,bnd_vtx_coords = search_bnd_vtx(env,vtx,False)
            env.u_bnd_vtx = env.l_bnd_vtx
            # === test boundary edge intersection
            isec_mask = np.logical_and(
                isec_mask,
                edge_intersect(bnd_vtx_coords[:2],env.coords_list,p_out_list,p_in_list))
            # === test for sector angles in valid region
            test_angle_arr = planar_angle(coords_arr-v_coords,bnd_vtx_coords-v_coords)
            bnd_test = np.where(test_angle_arr>=0,1,0)
            l_bnd_vec = np.append(env.points_2D[vtx.surrounding_vertices[0]],0) - v_coords
            u_bnd_vec = np.append(env.points_2D[vtx.surrounding_vertices[-1]],0) - v_coords
            angle_test_arr = planar_angle(coords_arr-v_coords,l_bnd_vec)
            angle_test_arr = np.where(angle_test_arr>0,angle_test_arr,2*np.pi+angle_test_arr)
            angle = planar_angle(u_bnd_vec,l_bnd_vec)
            if angle<=0:
                angle_mask = np.where(angle_test_arr<np.pi,1,0)
            else:
                angle_mask = np.where(angle_test_arr<angle,1,0)
            angle_mask = np.where((bnd_test+angle_mask)>1,1,0)

        # === second new vertex valid sector angle mask
        elif env.ext_cnt == 1:
            # === compute upper (CW) boundary vertex
            env.u_bnd_vtx,bnd_vtx_coords = search_bnd_vtx(env,vtx,True)
            # === test boundary edge intersection
            isec_mask = np.logical_and(
                isec_mask,
                edge_intersect(bnd_vtx_coords[:2],env.coords_list,p_out_list,p_in_list))
            # === test for sector angles in valid region
            test_angle_arr = planar_angle(bnd_vtx_coords-v_coords,coords_arr-v_coords)
            bnd_test = np.where(test_angle_arr>=0,1,0)
            l_bnd_vec = np.append(np.array(env.ext_action_list_c[0]),0) - v_coords
            u_bnd_vec = np.append(env.points_2D[vtx.surrounding_vertices[-1]],0) - v_coords
            angle_test_arr = planar_angle(u_bnd_vec,coords_arr-v_coords)
            angle_test_arr = np.where(angle_test_arr>0,angle_test_arr,2*np.pi+angle_test_arr)
            angle = planar_angle(u_bnd_vec,l_bnd_vec)
            if angle<0:
                angle_mask = np.where(angle_test_arr<np.pi,1,0)
            else:
                angle_mask = np.where(angle_test_arr<angle,1,0)
            angle_mask = np.where((bnd_test+angle_mask)>1,1,0)

        # === third new vertex valid sector angle and unit mask
        else:
            # === angle mask ===
            # === compute lower (CCW) boundary vertex
            l_bnd_vec = np.append(np.array(env.ext_action_list_c[0]),0) - v_coords
            # === compute upper (CW) boundary vertex
            u_bnd_vec = np.append(np.array(env.ext_action_list_c[1]),0) - v_coords
            # === test for sector angles in valid region
            angle_test_arr = planar_angle(u_bnd_vec,coords_arr-v_coords)
            angle_test_arr = np.where(angle_test_arr>0,angle_test_arr,2*np.pi+angle_test_arr)
            angle = planar_angle(u_bnd_vec,l_bnd_vec)
            if angle<0:
                angle_mask_1 = np.where(angle_test_arr<np.pi,1,0)
                angle_mask_2 = np.where(2*np.pi+angle-angle_test_arr<np.pi,1,0)
                angle_mask = np.where((angle_mask_1+angle_mask_2)>1,1,0)
            else:
                angle_mask = np.where(angle_test_arr<angle,1,0)

            # === unit_mask ===
            l_sector = planar_angle(l_bnd_vec,np.append(env.points_2D[vtx.surrounding_vertices[0]],0) - v_coords)
            u_sector = planar_angle(np.append(env.points_2D[vtx.surrounding_vertices[-1]],0) - v_coords,u_bnd_vec)
            u_2 = angle_test_arr.reshape((-1,1))
            u_3 = abs(planar_angle(coords_arr-v_coords,l_bnd_vec)).reshape((-1,1))
            u = np.array(compute_first_unit(env,v,u_sector,l_sector)).reshape((1,-1))
            u_1 = np.ones((u_2.size,u.size))
            u_1 = np.multiply(u_1,u)#.reshape((-1,len(env.psi_opt)))
            u_1 = np.where(u_1==np.pi,np.pi-0.00001,u_1)
            u_1 = np.where(u_1>np.pi,2*np.pi-u_1,u_1)
            u_2 = np.where(u_2>np.pi,2*np.pi-u_2,u_2)
            u_3 = np.where(u_3>np.pi,2*np.pi-u_3,u_3)
            if len(env.psi_opt)>1:
                M23 = np.concatenate((np.ones(np.shape(u_2)),u_2,u_3),axis=1)
                M = np.repeat(
                    M23[:, :, np.newaxis], len(env.folding_direction*env.psi_opt), axis=2)
                M[:,0,:] = u_1
                M = np.sort(M,axis=1)
                unit_mask = np.where(M[:,0,:]+M[:,1,:]>=M[:,2,:],1,0)
                if env.optimize_psi:
                    unit_mask = unit_mask[:,-1]
                else:
                    unit_mask = np.where(
                        np.add.reduce(unit_mask,axis=1)==len(env.psi_opt),1,0)
            else:
                M = np.concatenate((u_1,u_2,u_3),axis=1)
                M = np.sort(M,axis=1)
                unit_mask = np.where(M[:,0]+M[:,1]>=M[:,2],1,0)

        if env.ext_cnt == 2:
            mask = np.where((isec_mask+angle_mask+unit_mask)>2,1,0)
        else:
            mask = np.where((isec_mask+angle_mask)>1,1,0)

        # === constrain max edge length
        edge_len_arr = np.sqrt(np.add.reduce((coords_arr-v_coords)**2,axis=1))
        valid_edge = np.where(edge_len_arr<=env.CL_MAX,1,0)
        mask = np.where((mask+valid_edge)>1,1,0)

        # === update env boundary vertices for later reference
        env.l_bnd_vec = l_bnd_vec
        env.u_bnd_vec = u_bnd_vec

        # === mask out extended vertices
        for v_obj in env.vertex_objs:
            if v_obj.is_pseudo: continue
            idx = np.where((env.coords_list == v_obj.coords).all(axis=1))[0][0]
            if env.ext_cnt == 0 and mask[idx] and env.l_bnd_vtx==v_obj.name:
                last_surr_vtx_coords = env.points_2D[v_obj.surrounding_vertices[-1]]
                new_sector_angle = planar_angle(last_surr_vtx_coords-v_obj.coords,env.points_2D[v]-v_obj.coords)
                # -> cannot merge to prevent dead vertex due to geometric constraints (sector angle > pi)
                if new_sector_angle < 0:
                    mask[idx] = 0
                else:
                    mask[idx] = 1
            elif env.ext_cnt == 1 and mask[idx] and env.u_bnd_vtx==v_obj.name:
                first_surr_vtx_coords = np.array(env.points_2D[v_obj.surrounding_vertices[0]])
                cntr_pnt = np.array(v_obj.coords)
                new_sector_angle = planar_angle(np.array(env.points_2D[v])-cntr_pnt,first_surr_vtx_coords-cntr_pnt)
                # -> cannot merge to prevent dead vertex due to geometric constraints (sector angle > pi)
                if new_sector_angle < 0:
                    mask[idx] = 0
                else:
                    mask[idx] = 1
            else:
                mask[idx] = 0

        # === mask out duplicate actions
        for i in env.ext_action_list:
            mask[i] = 0

        mask = np.array(mask)
        env.ext_action_mask = mask

        if env.symmetry:
            bnd_vtx = env.vertex_objs[env.l_bnd_vtx] if env.ext_cnt == 0 else env.vertex_objs[env.u_bnd_vtx]
            if env.symmetry_axis == "point":
                mask = map2symm_mask(mask,env.half_board_length,env.ext_cnt,bnd_vtx,env.vertex_objs[env.l_bnd_vtx],env.vtx2ext_on_axis)
            elif env.symmetry_axis == "y-axis":
                mask = map2symm_mask_y(mask,env.half_board_length,env.ext_cnt,bnd_vtx,env.vtx2ext_on_axis)

        if env.symmetry_axis == "point":
            mask_arr = mask.reshape((env.half_board_length+1,env.half_board_length+1)).T
            if env.cross_axis_symmetry:
                mask_eye = np.identity(env.half_board_length+1).astype(int)
                if env.ext_cnt>0 and env.symmetry_axis=="point" and abs(v_coords[0])==v_coords[1]:
                    # vertex 2 extend on cross symmetry axis
                    if env.ext_cnt==1:
                        # 2nd extension must not! lie on xy-symmetry axis
                        x,y = env.ext_action_list_c[0]
                        r,c = env.half_board_length-y,env.half_board_length+x
                        mask_arr = np.zeros((env.half_board_length+1,env.half_board_length+1))
                        mask_arr[c,r] = 1
                    elif env.ext_cnt==2:
                        # last extension must lie on xy-symmetry axis
                        mask_eye = np.identity(env.half_board_length+1)
                        mask_arr = mask_arr+mask_eye
                        mask_arr[env.half_board_length-1,env.half_board_length-1] = 0
                        mask_arr = np.where(mask_arr>1.1,1,0)
                elif not abs(v_coords[0])==v_coords[1]:
                    # vertex 2 extend not on cross symmetry axis
                    mask_arr = np.tril(mask_arr)
                    mask_arr = np.where(mask_arr-mask_eye==1,1,0)

            if env.ext_cnt>0 and not env.vtx2ext_on_axis == "x":
                mask_arr[env.half_board_length,:] = 0
            if not env.ext_cnt==1 and not env.vtx2ext_on_axis == "y":
                mask_arr[:,env.half_board_length] = 0
            mask = mask_arr.T.reshape((-1,))

        env.ext_action_mask = mask
        return mask



def compute_test_angles(bnd_vtx_coords: np.ndarray, v_coords: np.ndarray, coords_vec: np.ndarray):
    bnd_vec = np.matmul((bnd_vtx_coords - v_coords),np.array([[1,0,0],[0,1,0]]))
    test_vec_arr = coords_vec-np.append(v_coords,0)
    return planar_angle(test_vec_arr,bnd_vec)



def search_bnd_vtx(env, vtx, upper_bound_bln: bool):
# ------------------------------------------------------------------------------
#
#   Search for the next boundary vertex
#
#   takes:      vtx: vertex2extend obj
#               upper_bound_bln:    True for searching upper (CW) boundary vertex,
#                                   False for lower (CCW) boundary vertex
#
#   returns:    bnd_vtx_coords: coordinates of boundary vertex
#
# ------------------------------------------------------------------------------
    if upper_bound_bln:
        m_bnd_vtx = env.vertex_objs[vtx.surrounding_vertices[-1]]
        idx = m_bnd_vtx.surrounding_vertices.index(vtx.name)

        while True:
            if idx == 0:
                bnd_vtx = m_bnd_vtx.surrounding_vertices[-1]
            else:
                bnd_vtx = m_bnd_vtx.surrounding_vertices[idx-1]

            if not env.vertex_objs[bnd_vtx].is_extended:
                bnd_vtx_coords = env.points_2D[bnd_vtx]
                bnd_vtx_obj = env.vertex_objs[bnd_vtx]
                break
            else:
                bnd_vtx_obj = env.vertex_objs[bnd_vtx]
                idx = bnd_vtx_obj.surrounding_vertices.index(m_bnd_vtx.name)
                m_bnd_vtx = bnd_vtx_obj

        bnd_vtx_bnd_vec = np.asarray(env.points_2D[bnd_vtx_obj.surrounding_vertices[-1]])\
            -np.asarray(bnd_vtx_coords)

    else:
        m_bnd_vtx = env.vertex_objs[vtx.surrounding_vertices[0]]
        idx = m_bnd_vtx.surrounding_vertices.index(vtx.name)

        while True:
            if idx + 1 < len(m_bnd_vtx.surrounding_vertices):
                bnd_vtx = m_bnd_vtx.surrounding_vertices[idx+1]
            else:
                bnd_vtx = m_bnd_vtx.surrounding_vertices[0]
            if not env.vertex_objs[bnd_vtx].is_extended:
                bnd_vtx_coords = env.points_2D[bnd_vtx]
                bnd_vtx_obj = env.vertex_objs[bnd_vtx]
                break
            else:
                bnd_vtx_obj = env.vertex_objs[bnd_vtx]
                idx = bnd_vtx_obj.surrounding_vertices.index(m_bnd_vtx.name)
                m_bnd_vtx = bnd_vtx_obj

        bnd_vtx_bnd_vec = np.asarray(env.points_2D[bnd_vtx_obj.surrounding_vertices[0]])\
            -np.asarray(bnd_vtx_coords)

    return bnd_vtx,np.append(bnd_vtx_coords,0)



def edge_intersect(v_coords_l: list, crds_list: list, p1_list: list, p2_list: list):
# ------------------------------------------------------------------------------
#
#   Computation of edge intersection mask
#
#   takes:      v_coords_l: list vertex2extend coords
#               crds_list:  list of 2D-coordinates to test intersection for
#               p1_list:    list of edge starting point coordinates
#               p2_list:    list of edge ending point coordinates
#
#   returns:    intsec_mask: intersection mask (flattened)
#
# ------------------------------------------------------------------------------

    np.seterr(invalid='ignore')
    v_coords = np.array(v_coords_l).reshape(1,2)
    p1_arr = np.array(p1_list)
    p2_arr = np.array(p2_list)
    crds_vec = np.array(crds_list)  # all board coords
    ray_o_vec = v_coords*np.ones(np.shape(p1_arr))  # ray origin vector, contains vtx2ext coords
    # ray dir vec contains all dir vectors from vtx2ext to each point on board
    ray_dir_vec = crds_vec - v_coords*np.ones(np.shape(crds_vec))
    # v1 vec contains the vectors from vtx2ext to each starting point
    v1_vec = ray_o_vec - p1_arr
    # v2 vec contains the vectors from the starting points to their ending points
    # (the lines to test intersection against)
    v2_vec = p2_arr - p1_arr
    v3_vec = np.matmul(ray_dir_vec,np.array([[0,1],[-1,0]]))
    cross = np.cross(v2_vec,v1_vec,axisa=1,axisb=1)
    with np.errstate(divide='ignore'):
        dot_res_v2 = np.matmul(v3_vec,v2_vec.transpose())
        cross_res = np.ones(np.shape(dot_res_v2))*cross
        try:
            t1 = np.divide(cross_res,dot_res_v2)
            t1 = np.where(np.isnan(t1),1,t1)
        except Exception:
            t1 = np.where(np.isnan(t1),1,t1)
            pass

        dot_res_v1 = np.matmul(v3_vec,v1_vec.transpose())
        try:
            t2 = np.divide(dot_res_v1,dot_res_v2)
            t2 = np.where(np.isnan(t2),1,t2)
        except Exception:
            t2 = np.where(np.isnan(t2),1,t2)
            pass

    test_0 = np.where(t1>1.0,1,0) # apply to allow inner face vertices
    test_1 = np.where(t1<=0.,1,0)
    test_21 = np.where(t1==1,1,0)
    test_22 = np.where(t2==1,1,0)
    test_2 = np.where((test_21+test_22)==2,1,0)
    test_3 = np.where(t2<=0.,1,0)
    test_4 = np.where(t2>1.0,1,0)

    test_tensor = np.dstack((test_0,test_1,test_2,test_3,test_4))
    test_arr = np.where(np.any(test_tensor,2),0,1)
    mask = np.where(np.any(test_arr,1),0,1)

    return mask


def compute_first_unit(env, v: int, new_sector: float, l_sector: float):
    # === computes the full unit action mask, masking out all actions,
    # === which would result in a violation of the triangle inequality during folding

    # === load the vertex to be extended
    vtx = env.vertex_objs[v]
    # === collect edge-vectors
    arr = np.array(env.points_2D)[vtx.surrounding_vertices]-np.array(vtx.coords)
    surr_vecs = np.hstack((arr,np.zeros(len(vtx.surrounding_vertices)).reshape(-1,1)))
    surr_vecs_1 = np.roll(surr_vecs,-1,axis=0)
    # === compute sector angles for first unit
    sec_angles = planar_angle(surr_vecs,surr_vecs_1)
    alpha_arr = np.concatenate(([l_sector],sec_angles[:-1],[new_sector]))
    alpha_arr = np.stack((alpha_arr,)*len(env.psi_opt),axis=0)
    # === collect dihedral angles from vertex to be extended
    rho_arr = vtx.dihedral_angles.copy()
    # === resulting first unit
    res = list(map(compute_folded_unit,alpha_arr,rho_arr))
    return res
