from gym_rori.envs.rori_env import RoriEnv
import numpy as np
from build_scripts import build_pattern_from_decision_chain
import os
from copy import deepcopy
import imageio.v2 as imageio
from plotting import plot_polygons


def generate_gif2(env_config,psi_max,pattern,resolution,view=[90,-90,23]):
#    env_config["optimize_psi"] = False
    board_param = deepcopy(env_config["board_length"])
    if env_config["symmetry_axis"]=="point":
        board_param = board_param/2
    board_param += 1
    save_dir = env_config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    env_config["PSI"] = psi_max #abs(psi_max)
    env = RoriEnv(env_config)
    env.best_episode_reward = np.inf
    build_pattern_from_decision_chain(env,pattern,board_param)

    plot_polygons(
        env,
        psi_max,
        anim=True,
        resolution=resolution, 
        psi_opt_index=env.opt_idx)
    del env
    images = []
    filenames = os.listdir(save_dir+'/gif')
    filenames.sort()
    filenames_rev = deepcopy(filenames)
    filenames_rev.reverse()
    filenames = filenames+filenames_rev
    for filename in filenames:
        images.append(imageio.imread(save_dir+'/gif/'+filename))
    imageio.mimsave(save_dir+'/animation.gif', images, fps=5)


def generate_gif(env_config,psi_max,pattern,resolution,view=[90,-90,23]):
    env_config["optimize_psi"] = False
    board_param = deepcopy(env_config["board_length"])
    if env_config["symmetry_axis"]=="point":
        board_param = board_param/2
    board_param += 1
    save_dir = env_config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    env_config["PSI"] = np.sign(psi_max)*0.0001
    env = RoriEnv(env_config)
    env.best_episode_reward = np.inf
    build_pattern_from_decision_chain(env,pattern,board_param)
    env.fold_pattern(0.0001,0,0)
    for i in range(resolution):
        psi_i = psi_max*(i+1)/resolution
        if psi_i==0: psi_i=np.sign(psi_max)*0.0001
        env_config["PSI"] = psi_i
        del env
        env = RoriEnv(env_config)
        env.best_episode_reward = np.inf
        build_pattern_from_decision_chain(env,pattern,board_param)
        env.fold_pattern(psi_i,0,0)
        # if want to test edge length==constant throughout folding
        # p3D = env.points_3D
        # d=[]
        # for edge in env.edge_list:
        #     d.append(np.linalg.norm(p3D[edge[0]]-p3D[edge[1]]))
        # print(d)
    del env

    images = []
    filenames = os.listdir(save_dir+'/gif')
    filenames.sort()
    filenames_rev = deepcopy(filenames)
    filenames_rev.reverse()
    filenames = filenames+filenames_rev
    for filename in filenames:
        images.append(imageio.imread(save_dir+'/gif/'+filename))
    imageio.mimsave(save_dir+'/animation.gif', images, fps=5)
