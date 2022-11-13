# plotting

import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import networkx as nx
from find_cycles import find_cycles
import numpy as np
import os
from datetime import datetime
from rules import get_ext_action_mask, get_sel_action_mask
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def plot_pattern(env, show_mask=False, silent=True):
    # ----------------------------------------------------------------------
    #
    #   takes:      nothing
    #
    #   returns:    plot of directed graph visualization of pattern
    #
    # ----------------------------------------------------------------------

    fig_g,ax = plt.subplots() #figure()
    fig_g.set_size_inches(8,7)
    ax.set_aspect('equal')
    G = nx.DiGraph()
    labels_list = []
    node_colors = []
    alpha_list = []
    edge_list = env.edge_list.copy()

    # === if activated the board state with valid actions is shown at each
    # === extension step
    def map2point(index):
        return index - int(index/(env.board_length+1))*env.half_board_length

    if show_mask:
        # === adjust mask to board for symmetry shrinked space
        board = np.zeros(env.board_size).reshape((env.board_length+1,env.board_length+1))
        mask = get_ext_action_mask(env,env.sel_action)
        s = int(np.sqrt(mask.size))
        if env.symmetry_axis=="y-axis" and mask.size < env.board_size:
                board[:,:env.half_board_length + 1] = \
                    mask.reshape((env.board_length+1,env.half_board_length + 1))
        else:
            board[:s,:s]= mask.reshape((s,s))

        for i,m in enumerate(board.reshape(-1,)):
            G.add_node(i+len(env.vertex_list),pos=env.coords_list[i],alpha=0.25)
            if env.symmetry_axis == "point":
                j = map2point(i)
            else:
                j = i
            if m:
                node_colors.append('lightgreen')
                labels_list.append(j)
            else:
                node_colors.append('white')
                labels_list.append("")

    for i in env.vertex_list:
        G.add_node(env.vertex_list[i],pos=env.points_2D[i],alpha=1.0)
        if i == env.sel_action:
            node_colors.append('gold')
        elif env.vertex_objs[i].is_pseudo:
            node_colors.append('lightgrey')
        elif env.vertex_objs[i].is_extended:
            if env.vertex_objs[i].rbm == 1:
                node_colors.append('lightblue')
            else:
                node_colors.append('olive')
        elif i==0:
            node_colors.append('grey')
        else:
            node_colors.append('pink')
        labels_list.append(str(i))
        if env.count == 0 and node_colors[-1] == 'pink':
            v_idx = env.board_coords.tolist().index(env.points_2D[i])
            j = map2point(v_idx)
            mask = get_sel_action_mask(env)
            if mask[i]:
                labels_list[-1] = str(i)+"--"+str(j)
                node_colors[-1] = 'lightgreen'

    #edge_colors = []
    #edge_labels = {}

    ax.grid(visible=True, color='grey', linestyle='-', linewidth=0.1)
    ax.set_xticks(np.arange(-env.half_board_length, env.half_board_length+1, 1))
    ax.set_yticks(np.arange(-env.half_board_length, env.half_board_length+1, 1))
    #ax.margins(x=0, y=0)
    #plt.rcParams["figure.autolayout"] = True
    G.add_edges_from(edge_list)
    pos=nx.get_node_attributes(G,'pos')
    labelmap = dict(zip(G.nodes(), labels_list))
    alphamap = dict(zip(G.nodes(), alpha_list))
    nx.draw(G,pos,labels=labelmap,with_labels=True,font_size=8,alpha=0.5,
        node_color=node_colors,edge_color="black",arrows=True,ax=ax)
    plt.axis('on')

    if not show_mask:
        save_dir = env.config["save_dir"]+"/"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        dt = datetime.now()
        plt.savefig(
            save_dir+\
            "total_steps="+str(env.total_step_count)+"_"+\
            dt.strftime("%y-%m-%d_%H-%M-%S.%f")+"_pattern_"+\
            str(env.vertex_list[-1])+"_vertices_rew="+\
            str(round(sum(env.reward_history),2))+".png")
    if silent: plt.close('all')



def plot_polygons(env,
    fold_angle: float,
    show_target=False,
    al=0.25, anim=False,
    silent=True,
    view=[90,-90,23],
    resolution=10,
    psi_opt_index=0):
    # ----------------------------------------------------------------------
    #
    #   takes:      fold_angle: global driving angle psi value for simulation
    #               show_target: boolean 1 show, 0 do not show target
    #
    #   returns:    a surface plot of target shape and the 3D folded mesh
    #
    # ----------------------------------------------------------------------

    fig = plt.figure()
    fig.set_size_inches(8,7)
    ax = a3.Axes3D(fig,zscale='linear')
    ax.figure.canvas.draw()
    ax.view_init(elev=view[0],azim=view[1]) # 30,45
    ax.dist=view[2]
    if show_target:
        point_invalid = np.array([np.any(abs(row)>15) \
            for row in env.sampled_points[:,2]])
        p_tar = env.sampled_points[~point_invalid]
        x_p = p_tar[:,0]
        y_p = p_tar[:,1]
        z_p = p_tar[:,2]
        ax.scatter3D(x_p,y_p,z_p)
    X_MAX = Y_MAX = Z_MAX = 10
    ax.set_xlim3d([-X_MAX, X_MAX])
    ax.set_ylim3d([-Y_MAX, Y_MAX])
    ax.set_zlim3d([-Z_MAX, Z_MAX])
    ax.set_axis_off()

    faces = find_cycles(env)
    # ----- preparation of target surface plot
    # ----- polygon 3D collection assembly
    points = env.all_3D_points.swapaxes(0,1)
    def plot_folded_state(fold_index):
        fcs = []
        p = points[fold_index]
        for face in faces:
            fc = p[face]
            fcs.append(fc)
        tri = a3.art3d.Poly3DCollection(fcs)
        tri.set_color("beige")
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

        if silent:
            save_dir = env.config["save_dir"]+"/gif/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dt = datetime.now()
            plt.savefig(
                save_dir+\
                str(env.total_step_count)+"_"+\
                dt.strftime("%y-%m-%d_%H-%M-%S.%f")+"_polygons_"+\
                str(env.vertex_list[-1])+"_vertices_rew="+\
                str(round(sum(env.reward_history),2))+\
                "_psi="+str(env.folding_direction*env.psi_opt[fold_index])+".png")
            tri.remove()

    if anim==True:
        fold_range = list(range(env.psi_opt.size-psi_opt_index))
        fold_range.reverse()
        for psi_index in fold_range:
            plot_folded_state(psi_index)
    else:
        plot_folded_state(0)
    ax.autoscale(enable=True, axis='both', tight=None)
    plt.close(fig='all')
