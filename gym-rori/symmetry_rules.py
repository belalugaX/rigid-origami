import numpy as np
from copy import deepcopy

# === symmetry rules
def sym_expand(env,symmetry_axis="y-axis",on_axis="0"):
    roll_back = 0
    if symmetry_axis=="y-axis" and not on_axis=="y":
        foldable,state,reward,msg,_,_ = mirror_y(env,env.sel_action,env.ext_action_list_c)
        if not foldable: roll_back = 2
        return foldable,state,reward,msg,roll_back
    elif symmetry_axis=="x-axis" and not on_axis=="x":
        foldable,state,reward,msg,_,_ = mirror_x(env,env.sel_action,env.ext_action_list_c)
        if not foldable: roll_back = 2
        return foldable,state,reward,msg,roll_back
    elif symmetry_axis=="point":
        if on_axis=="x":
            foldable,state,reward,msg,_,_ = mirror_y(env,env.sel_action,env.ext_action_list_c)
            if env.cross_axis_symmetry:
                foldable,state,reward,msg,new_ext_action_list_c,v_new = \
                    rotate_90(env,env.sel_action,env.ext_action_list_c)
                foldable,state,reward,msg,_,_ = mirror_x(env,v_new,new_ext_action_list_c)
            if foldable:
                return foldable,state,reward,msg,0
            else:
                return foldable,state,reward,msg,2
        elif on_axis=="y":
            foldable,state,reward,msg,_,_ = mirror_x(env,env.sel_action,env.ext_action_list_c)
            if env.cross_axis_symmetry:
                foldable,state,reward,msg,new_ext_action_list_c,v_new = \
                    rotate_90(env,env.sel_action,env.ext_action_list_c,ccw=True)
                foldable,state,reward,msg,_,_ = mirror_y(env,v_new,new_ext_action_list_c)
            if foldable:
                return foldable,state,reward,msg,0
            else:
                return foldable,state,reward,msg,2
        else:
            foldable,state,reward,msg,points2mirror,v = mirror_x(env,env.sel_action,env.ext_action_list_c)
            if foldable:
                foldable,state,reward,msg,points2mirror2,v2 = mirror_y(env,env.sel_action,env.ext_action_list_c)
                if foldable:
                    foldable,state,reward,msg,points2mirror3,v3 = mirror_y(env,v,points2mirror)
                    if foldable:

                        if env.cross_axis_symmetry:
                            ret = 5
                            foldable,state,reward,msg,_,_ = mirror_xy(env,env.sel_action,env.ext_action_list_c,quadrant=2)
                            if foldable:
                                ret +=1
                                foldable,state,reward,msg,_,_ = mirror_xy(env,v,points2mirror,quadrant=3)
                            if foldable:
                                ret += 1
                                foldable,state,reward,msg,_,_ = mirror_xy(env,v2,points2mirror2,quadrant=1)
                            if foldable:
                                ret += 1
                                foldable,state,reward,msg,_,_ = mirror_xy(env,v3,points2mirror3,quadrant=4)
                            if foldable:
                                return foldable,state,reward,msg,0
                            else:
                                return foldable,state,reward,msg,ret

                        return foldable,state,reward,msg,0
                    else:
                        return foldable,state,reward,msg,4
                else:
                    return foldable,state,reward,msg,3
            else:
                return foldable,state,reward,msg,2
    else:
        return True,env.state,[0,0],"extended vertex located on a symmetry line",0


def mirror_y(env,v,points2mirror):
    # === orig vtx2ext coords
    v_coords = env.points_2D[v]
    is_on_symaxis = True if v_coords[0]==0 else False
    if not is_on_symaxis:
        # === mirrored vtx2ext name
        v_coords_n = v_coords.copy()
        v_coords_n[0] = -v_coords_n[0]
        v_new = env.points_2D.index(v_coords_n)
        env.vertex_objs[v_new].is_mirrored = True
        ext_action_list_c = [
            points2mirror[1].copy(), 
            points2mirror[0].copy(), 
            points2mirror[2].copy()]
        ext_action_list_c[0][0] = -ext_action_list_c[0][0]
        ext_action_list_c[1][0] = -ext_action_list_c[1][0]
        ext_action_list_c[2][0] = -ext_action_list_c[2][0]
        foldable,state,reward,msg = env.do_ext_action(v_new,env.rbm_action,ext_action_list_c)
        return foldable,state,reward,msg,ext_action_list_c,v_new
    else:
        return True,env.interim_state,[0,0],\
            "no mirroring since extended vertex lies on symmetry axis",points2mirror,v


def rotate_90(env,v,points2rotate,ccw=False):
    ccw = -1 if ccw else 1
    # === orig vtx2ext coords
    v_coords = env.points_2D[v]
    # === mirrored vtx2ext name
    v_coords_n = v_coords.copy()
    v_coords_n = [ccw*v_coords_n[1],-ccw*v_coords_n[0]]
    v_new = env.points_2D.index(v_coords_n)
    env.vertex_objs[v_new].is_mirrored = True
    ext_action_list_c = [
        points2rotate[0].copy(),
        points2rotate[1].copy(),
        points2rotate[2].copy()]
    ext_action_list_c[0] = [ccw*ext_action_list_c[0][1],-ccw*ext_action_list_c[0][0]]
    ext_action_list_c[1] = [ccw*ext_action_list_c[1][1],-ccw*ext_action_list_c[1][0]]
    ext_action_list_c[2] = [ccw*ext_action_list_c[2][1],-ccw*ext_action_list_c[2][0]]
    foldable,state,reward,msg = env.do_ext_action(v_new,env.rbm_action,ext_action_list_c)
    return foldable,state,reward,msg,ext_action_list_c,v_new


def mirror_xy(env,v,points2rotate,quadrant=2):
    v_coords = env.points_2D[v]
    v_coords_n = v_coords.copy()
    x,y = v_coords
    is_on_symaxis = True if abs(x)==abs(y) else False

    if quadrant == 1:
        v_coords_n = [v_coords_n[1],v_coords_n[0]]
    elif quadrant==2:
        v_coords_n = [-v_coords_n[1],-v_coords_n[0]]
    elif quadrant==3:
        v_coords_n = [v_coords_n[1],v_coords_n[0]]
    elif quadrant==4:
        v_coords_n = [-v_coords_n[1],-v_coords_n[0]]
    else:
        ccw = 1

    if not is_on_symaxis:
        # === mirrored vtx2ext name
        v_new = env.points_2D.index(v_coords_n)
        env.vertex_objs[v_new].is_mirrored = True
        ext_action_list_co = [
            points2rotate[0].copy(),
            points2rotate[1].copy(),
            points2rotate[2].copy()]
        ext_action_list_c = deepcopy(ext_action_list_co)
        if quadrant == 4:
            ext_action_list_c[0] = [-ext_action_list_co[1][1],-ext_action_list_co[1][0]]
            ext_action_list_c[1] = [-ext_action_list_co[0][1],-ext_action_list_co[0][0]]
            ext_action_list_c[2] = [-ext_action_list_co[2][1],-ext_action_list_co[2][0]]
        elif quadrant == 3:
            ext_action_list_c[0] = [ext_action_list_co[1][1],ext_action_list_co[1][0]]
            ext_action_list_c[1] = [ext_action_list_co[0][1],ext_action_list_co[0][0]]
            ext_action_list_c[2] = [ext_action_list_co[2][1],ext_action_list_co[2][0]]
        elif quadrant == 1:
            ext_action_list_c[0] = [ext_action_list_co[1][1],ext_action_list_co[1][0]]
            ext_action_list_c[1] = [ext_action_list_co[0][1],ext_action_list_co[0][0]]
            ext_action_list_c[2] = [ext_action_list_co[2][1],ext_action_list_co[2][0]]
        else:
            ext_action_list_c[0] = [-ext_action_list_co[1][1],-ext_action_list_co[1][0]]
            ext_action_list_c[1] = [-ext_action_list_co[0][1],-ext_action_list_co[0][0]]
            ext_action_list_c[2] = [-ext_action_list_co[2][1],-ext_action_list_co[2][0]]
        foldable,state,reward,msg = env.do_ext_action(v_new,env.rbm_action,ext_action_list_c)
        return foldable,state,reward,msg,ext_action_list_c,v_new
    else:
        return True,env.interim_state,[0,0],\
            "no mirroring since extended vertex lies on symmetry axis",points2rotate,v



def mirror_x(env,v,points2mirror):
    # === orig vtx2ext coords
    v_coords = env.points_2D[v]
    is_on_symaxis = True if v_coords[1]==0 else False # v_coords[0]==0 or
    if not is_on_symaxis:
        # === mirrored vtx2ext name
        v_coords_n = v_coords.copy()
        v_coords_n[1] = -v_coords_n[1]
        v_new = env.points_2D.index(v_coords_n)
        env.vertex_objs[v_new].is_mirrored = True
        ext_action_list_c = [
            points2mirror[1].copy(),
            points2mirror[0].copy(),
            points2mirror[2].copy()]
        ext_action_list_c[0][1] = -ext_action_list_c[0][1]
        ext_action_list_c[1][1] = -ext_action_list_c[1][1]
        ext_action_list_c[2][1] = -ext_action_list_c[2][1]
        foldable,state,reward,msg = env.do_ext_action(v_new,env.rbm_action,ext_action_list_c)
        return foldable,state,reward,msg,ext_action_list_c,v_new
    else:
        return True, env.interim_state,env.reward,\
            "no mirroring since extended vertex lies on symmetry axis",points2mirror,v

def map2symm_mask(mask: np.ndarray, size: int, cnt: int, bnd_vtx, l_bnd_vtx=None, sym="0", sym_type="y-axis"):
    mask4sym = mask.copy().reshape((2*size+1,2*size+1)).transpose()
    sym_mask = mask4sym[:size+1,:size+1]
    if sym=="x":
        if cnt==0:
            # === mirror third quadrant mask
            sym_mask[size,:] = 0
            sym_mask[:,size] = 0
            sym_mask = np.flip(mask4sym[size:,:size+1],axis=0).transpose()
        elif cnt==2:
            # === valid final extension on x-symmetry-axis only
            sym_mask[:size,:size+1] = 0
            sym_mask = sym_mask.transpose()
    elif sym=="y":
        if cnt==2:
            # === valid final extension on y-symmetry-axis only
            sym_mask[:size+1,:size] = 0
            sym_mask = sym_mask.transpose()
        else:
            sym_mask[:,size] = 0
            sym_mask[size,:] = 0
            sym_mask = sym_mask.transpose()
    else:
        if cnt==2:
            # do not allow final extension on symmetry axis if vtx2ext not on axis
            sym_mask[size,:] = 0
            sym_mask[:,size] = 0
        elif cnt==1:
            # === y-axis symmetry avoid locked-in vertices
            if bnd_vtx.coords[0] == 0:
                sym_mask[:bnd_vtx.board_pos[0],size] = 0
            elif bnd_vtx.coords[0] < 0:
                sym_mask[:,size] = 0
            elif l_bnd_vtx.coords[1] == 0:
                # === y-axis symmetry avoid locked-in vertices
                sym_mask[size,:] = 0
        elif cnt==0:
            # === x-axis symmetry avoid locked-in vertices
            if bnd_vtx.coords[1] == 0 or cnt==2:
                # do not allow extension on symmetry axis if vtx2ext not on y-axis
                sym_mask[:,size] = 0
                sym_mask[size,:bnd_vtx.board_pos[1]] = 0
            elif bnd_vtx.coords[1] > 0:
                sym_mask[size,:] = 0
        sym_mask = sym_mask.transpose()
    return sym_mask.reshape(-1,)

def map2symm_mask_y(mask: np.ndarray, size: int, cnt: int, bnd_vtx, sym="0", sym_type="y-axis"):
    mask4sym = mask.copy().reshape((2*size+1,2*size+1)).transpose()
    sym_mask = mask4sym
    if sym=="-y":
        if cnt==0:
            # === mirror third quadrant mask
            sym_mask[:,:size+1] = np.flip(mask4sym[:,size:],axis=1)
            # === first and second extension forbidden on symm axis
            sym_mask[:,size] = 0
        elif cnt==2:
            # === valid final extension on x-symmetry-axis only
            sym_mask[:,:size] = 0
    elif sym=="y":
        if cnt==0:
            # === first and second extension forbidden on symm axis
            sym_mask[:,size] = 0
        elif cnt==2:
            # === valid final extension on y-symmetry-axis only
            sym_mask[:,:size] = 0
    else:
        # === y-axis symmetry avoid locked-in vertices
        if bnd_vtx.coords[0] == 0 or cnt==2:
            # do not allow extension on symmetry axis if vtx2ext not on y-axis
            sym_mask[:,size] = 0
            sym_mask[:bnd_vtx.board_pos[0],size] = 0
        elif bnd_vtx.coords[0] < 0:
            sym_mask[:,size] = 0
    sym_mask = sym_mask.transpose()

    # === constrain right half of board -> no actions allowed
    sym_mask[size+1:,:] = 0
    return sym_mask.reshape(-1,)

def map_symm2board_act(sym_act: int, size: int, symmetry_axis: str):
    if symmetry_axis == "point":
        board_act = sym_act + int(sym_act/(size+1))*size
    else:
        board_act = sym_act
    return board_act

def map_board2symm_act(board_act: int, size: int, symmetry_axis: str):
    if symmetry_axis == "point":
        sym_act = board_act - int(board_act/(2*size+1))*size
    else:
        sym_act = board_act
    return sym_act
