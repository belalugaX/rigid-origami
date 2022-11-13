#!/usr/bin/env python

import warnings
# ignore warnings
warnings.simplefilter("ignore", category=FutureWarning)

import gym,os
from gym.spaces import Discrete, Box, Dict

import trimesh
from trimesh.exchange.obj import export_obj
from trimesh.exchange.export import export_mesh

import pickle
import numpy as np
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt

from rules import get_sel_action_mask, get_ext_action_mask, mask_source_action, \
    get_init_mask
from find_cycles import find_cycles
from symmetry_rules import sym_expand, map_symm2board_act
from env_config import place_source, create_base, quad_config, simple_config,\
    simple_vert_config, single_config
from build_scripts import build_pattern_from_decision_chain
from plotting import plot_pattern, plot_polygons
from geometric import get_folded_mesh_points
from rewarders import shaped_hausdorff_distance_rewarder,reward_wrapper
import time


class RoriEnv(gym.Env):
    # --------------------------------------------------------------------------
    #
    #   gym simulation environment. Contains the methods to build and simulate
    #   ptu crease patterns from single vertices.
    #
    # --------------------------------------------------------------------------

    @property
    def half_board_length(self):
        return int(self.board_length/2)

    @property
    def board_size(self):
        return int((self.board_length+1)**2)

    @property
    def psi_opt(self):
        # return np.array([(self.PSI-0.001)/(i+1) for i in range(10)])
        eps = 1.
        return np.asarray([float(i + eps) * self.PSI / (9. + eps) for i in reversed(range(10))])

    @property
    def default_config(self):
        return {
            "done": False,
            "rewarder": shaped_hausdorff_distance_rewarder,
            "min_depth": 1,
            "count_interior": True,
            "seed_pattern": [],
            "cross_axis_symmetry": False,
            "cascade_mode": False,
            "mode": "TRAIN",
            "show_agent": False,
            "PSI": np.pi/4-0.001,
            "allow_source_action": True,
            "symmetry": True,
            "symmetry_axis": "y-axis",
            "max_vertices": 100,
            "board_length": 20,
            "base": "plain",
            "CL_MAX": np.inf,
            "TS": 2,
            "target_file": "/Users/jeremia/polybox/_MT/reinforcement-learning-for-rigid-origami/target.obj",
            "target_transform": np.identity(4),
        }

    @property
    def coords_list(self):
        return [[i, j] for i in range(round(-self.half_board_length), round(self.half_board_length + 1))
                for j in range(round(self.half_board_length), round(-self.half_board_length-1), -1)]

    @property
    def board_coords(self):
        return np.array(self.coords_list)

    @property
    def num_vertices(self):
        return np.count_nonzero(self.interim_state[:, :, 0])

    @property
    def ext_count(self):
        return len(self.ext_action_list)

    @property
    def count(self):
        return len(self.ext_action_list)+np.count_nonzero(self.interim_state[:, :, -1])

    @property
    def agent(self):
        agent = "ext_agent" if np.any(self.interim_state[:, :, -1]) else "sel_agent"
        return agent

    @property
    def folding_direction(self):
        return self.vertex_objs[0].rbm if any(self.vertex_objs) else 1

    @property
    def points_3D(self):
        # ----------------------------------------------------------------------
        #
        #   takes:      fold_angle: real value of psi, where -Pi < psi < Pi
        #
        #   returns:    points_3D: a 3-dim point cloud array
        #
        # ----------------------------------------------------------------------
        # return np.stack([vtx.coords_3D[self.opt_idx, :] for vtx in self.vertex_objs])
        return self.all_3D_points[:, self.opt_idx]

    @property
    def all_3D_points(self):
        # ----------------------------------------------------------------------
        #
        #   takes:      fold_angle: real value of psi, where -Pi < psi < Pi
        #
        #   returns:    points_3D: a 3-dim point cloud array
        #
        # ----------------------------------------------------------------------
        points = [vtx.coords_3D for vtx in self.vertex_objs]
        # points.extend([vtx.coords_3D for vtx in self.pseudo_vertex_objs])
        return np.stack(points)

    def __init_properties(self):
        # === init global variables
        self.set_action_mask = np.ones((1, self.board_size))
        # state tensor board,adjacency,rbm
        self.state = np.zeros((self.board_length+1, self.board_length+1, self.max_vertices+2), dtype=np.int16)
        # --- list contains Single Vertex objs
        self.vertex_objs = []
        self.vertex_list = []
        self.pseudo_vertices = {}
        # self.pseudo_vertex_objs = []
        # int vertex names list
        self.vertex_pos_list = []
        # --- 2D points in flat state
        self.points_2D = []
        self.edge_list = []
        self.foldable = True
        self.decision_chain = []
        self.ext_action_list = []
        if self.symmetry and self.symmetry_axis == "point":
            self.ext_act_space_size = (self.half_board_length + 1)**2
        elif self.symmetry and self.symmetry_axis=="y-axis":
            self.ext_act_space_size = (self.board_length+1)*(self.half_board_length + 1)
        else:
            self.ext_act_space_size = self.board_size
        # === bound action space for symmetry
        self.init_ext_action_mask = np.ones(self.ext_act_space_size, dtype=np.float32)

        self.reward_function = self.rewarder(**self.config)

        self.action_space = Discrete(self.ext_act_space_size*2)
        if self.symmetry and self.symmetry_axis == "point":
            self.observation_space = Dict({
                "obs": Box(low=-1,high=1,shape=
                    (self.half_board_length + 1,self.half_board_length + 1,self.max_vertices+2),dtype=int),
                "action_mask": Box(low=0,high=1,shape=((2*self.ext_act_space_size,)),dtype=np.float32)
            })
        elif self.symmetry and self.symmetry_axis=="y-axis":
            self.observation_space = Dict({
                "obs": Box(low=-1,high=1,shape=
                    (self.board_length+1,self.half_board_length + 1,self.max_vertices+2),dtype=np.int),
                "action_mask": Box(low=0,high=1,shape=((2*self.ext_act_space_size,)),dtype=np.float32)
            })
        else:
            self.observation_space = Dict({
                "obs": Box(low=-1,high=1,shape=np.shape(self.state),dtype=np.int),
                "action_mask": Box(low=0,high=1,shape=((2*self.ext_act_space_size,)),dtype=np.float32)
                })

        self.info = {}

        self.l_bnd_vtx = None
        self.u_bnd_vtx = None
        self.source_boundary_vtx = None

        # === create base pattern
        self.triangles = []
        # self.faces = []
        self.face_boundary_edges = [(0,1)]
        self.quad_base = False
        if not self.base == "plain": # TODO
            self.__create_base()
            self.faces = find_cycles(self)

        self.ext_action_list_c = []
        self.interim_state = self.state.copy()

        if len(self.vertex_list) > 0:
            self.sel_action_mask = get_sel_action_mask(self)
        else:
            self.sel_action_mask = get_init_mask(self)

        self.obs_dict = {
            "sel_agent": {  "state": self.interim_state,
                            "action_mask": self.sel_action_mask
            },
        }
        self.vtx2ext_on_axis = "0"

        self.reward_history = [0]
        self.sel_action = None
        self.action = None
        self.rbm_action = None
        self.board_vtx_count_prior_ext = len(self.vertex_list)

        if not self.seed_pattern == []:
            build_pattern_from_decision_chain(self, self.seed_pattern, self.board_length+1)
        self.map2single_agent()

        # store init state
        self.optimize_psi = self.config["optimize_psi"]
        self.opt_idx = 0
        self.triangle_inequality_holds = np.ones(np.size(self.psi_opt)).astype(bool)
        init_state_config = self.get_state_config()
        self.state_config = deepcopy(init_state_config)
        self.init_state_config = deepcopy(init_state_config)
        self.init_obs_dict = deepcopy(self.obs_dict)

    def __init__(self, config={}):
        self.__decode_config(config)
        self.__init_properties()
        self.ep_cnt = 0
        self.best_episode_reward = -np.inf
        self.total_step_count = 0
        self.reset()

    def reset(self):
        state_config = deepcopy(self.init_state_config)
        self.set_state_config(state_config)
        # return self.init_obs_dict
        obs = self.init_obs_dict
        for action in self.start_sequence:
            obs, rew, done, info = self.step(int(action))
        return obs

    def step(self, action):
        self.total_step_count += 1
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        actions : discrete
        Returns : observation, reward, done, info : tuple
        -------
        """

        self.decision_chain.append(action)
        self.action = self.__decode_single_agent_action(action)

        # is_terminate_state
        if self.num_vertices >= self.max_vertices or not self.foldable:
            self.done = True
        elif "sel_agent" in self.action:
            self.done = self.action[self.agent][0] == 0


        if self.mode == "DEBUG": print("ACT:", self.action, self.decision_chain, self.reward_history)

        # === enforcing symmetric patterns if "symmetry" property is set and
        # === map action to board action
        symmetric_action = self.define_symmetry(self.action)
        final_extension = self.count == 3
        self.action_switch(symmetric_action)

        # === display ext agent action mask for debugging
        if self.show_agent:
            plot_pattern(self,show_mask=True, silent=False)
            plt.show()

        any_actions = np.any(self.get_ava_actions(self.interim_state.copy()))
        if not self.foldable or not any_actions: self.done = True
        reward = self.compute_reward(final=self.done, final_extension=final_extension)
        self.reward_history.append(reward)
        self.set_observations()

        if self.done and self.vertex_objs and len(self.decision_chain)>0 and \
            sum(self.reward_history)>(self.best_episode_reward+0.01):
            self.__store_best_pattern()

        self.map2single_agent()
        # print("total steps:",self.total_step_count)
        if np.isnan(self.reward_history[-1]):
            self.fold_pattern(self.psi_opt[self.opt_idx],0,1,silent=0)
        assert not np.isnan(self.reward_history[-1]), "reward is NaN"+str(self.reward_history)
        return self.obs_dict,self.reward_history[-1],self.done,self.info

    def render(self, mode='human'):
        self.fold_pattern(self.psi_opt[0],True)

    def __decode_config(self, config_dict):
        config = {**self.default_config, **config_dict}
        for k, v in config.items():
            setattr(self, k, v)
        self.config = config

    def __decode_single_agent_action(self, action):
        if not isinstance(action,dict):
            rbm = 1 if int(action/self.ext_act_space_size)==0 else -1
            action = action % self.ext_act_space_size
            action = (action, rbm)
            if self.agent == "ext_agent":
                action = {self.agent: action[0]}
            else:
                if self.symmetry_axis=="point":
                    coords_list_sym = self.board_coords.reshape((self.board_length+1,self.board_length+1,2))
                    coords_list_sym = coords_list_sym[:self.half_board_length + 1,:self.half_board_length + 1,:]
                    coords_list_sym = coords_list_sym.reshape((-1,2)).tolist()
                elif self.symmetry_axis=="y-axis":
                    coords_list_sym = self.board_coords.reshape((self.board_length+1,self.board_length+1,2))
                    coords_list_sym = coords_list_sym[:self.half_board_length + 1,:,:]
                    coords_list_sym = coords_list_sym.reshape((-1,2)).tolist()
                else:
                    coords_list_sym = self.coords_list
                act = list(coords_list_sym[action[0]])
                if not act in self.points_2D or \
                    self.points_2D.index(act) > len(self.vertex_list):
                    action = {"source_action": [None, act, rbm]}
                else:
                    act = self.points_2D.index(act)
                    while self.vertex_objs[act].is_pseudo and self.vertex_objs[act].inactive:
                        act = self.vertex_objs[act].name
                    if self.vertex_objs[act].is_pseudo:
                        action = {"source_action": [act, None, rbm]}
                    else:
                        action = {self.agent: (act, rbm)}
        return action

    def __source_action(self, action):
        pseudo_vtx, c, rbm = action["source_action"]
        if pseudo_vtx == None:
            create_base(self,c,rbm)
            self.state_config = self.get_state_config()
        else:
            place_source(self, pseudo_vtx, rbm)
        self.state_config = self.get_state_config()

    def action_switch(self, action):
        if "source_action" in action:
            self.__source_action(action)
        elif self.count == 0:
            self.__select(action["sel_agent"][0], action["sel_agent"][1])
        elif self.count < 3:
            self.__extend(action["ext_agent"])
        elif self.count == 3:
            self.foldable = self.__extend_final(action["ext_agent"])

    def __store_best_pattern(self):
        self.best_episode_reward = sum(self.reward_history)
        self.best_path = deepcopy(self.decision_chain)
        self.save_folded_mesh()
        plot_pattern(self)

    def set_state_config(self,state_conf):
        state_config_ = deepcopy(state_conf)
        self.vertex_list = state_config_["vertex_list"]
        self.vertex_objs = state_config_["vertex_objs"]
        self.triangles = state_config_["triangles"]
        self.pseudo_vertices = state_config_["pseudo_vertices"]
        self.vertex_pos_list = state_config_["vertex_pos_list"]
        self.points_2D = state_config_["points_2D"]
        self.edge_list = state_config_["edge_list"]
        self.foldable = state_config_["foldable"]
        self.ext_action_list = state_config_["ext_action_list"]
        self.ext_action_list_c = state_config_["ext_action_list_c"]
        self.reward_history = state_config_["reward_history"]
        self.decision_chain = state_config_["decision_chain"]
        self.interim_state = state_config_["interim_state"]
        self.state = state_config_["state"]
        self.done = state_config_["done"]
        self.obs_dict = state_config_["obs_dict"]
        self.sel_action = state_config_["sel_action"]
        self.rbm_action = state_config_["rbm_action"]
        self.l_bnd_vtx = state_config_["l_bnd_vtx"]
        self.u_bnd_vtx = state_config_["u_bnd_vtx"]
        self.vtx2ext_on_axis = state_config_["vtx2ext_on_axis"]
        self.board_vtx_count_prior_ext = state_config_["board_vtx_count_prior_ext"]
        self.action = state_config_["action"]
        self.source_boundary_vtx = state_config_["source_boundary_vtx"]
        self.face_boundary_edges = state_config_["face_boundary_edges"]
        self.triangle_inequality_holds = state_config_["triangle_inequality_holds"]
        for v in self.vertex_objs:
            v.env = self

    def get_state_config(self):
        state_config_dict = {}
        def reset_env(vertex_obj):
            vertex_obj.env = None
        def set_env(vertex_obj):
            vertex_obj.env = self
        list(map(reset_env,self.vertex_objs))
        state_config_dict["vertex_objs"] = deepcopy(self.vertex_objs)
        list(map(set_env,self.vertex_objs))
        state_config_dict["vertex_list"] = deepcopy(self.vertex_list)
        state_config_dict["triangles"] = deepcopy(self.triangles)
        state_config_dict["pseudo_vertices"] = deepcopy(self.pseudo_vertices)
        state_config_dict["vertex_pos_list"] = deepcopy(self.vertex_pos_list)
        state_config_dict["points_2D"] = deepcopy(self.points_2D)
        state_config_dict["edge_list"] = deepcopy(self.edge_list)
        state_config_dict["foldable"] = deepcopy(self.foldable)
        state_config_dict["ext_action_list"] = deepcopy(self.ext_action_list)
        state_config_dict["ext_action_list_c"] = deepcopy(self.ext_action_list_c)
        state_config_dict["count"] = deepcopy(self.count)
        state_config_dict["reward_history"] = deepcopy(self.reward_history)
        state_config_dict["decision_chain"] = deepcopy(self.decision_chain)
        state_config_dict["interim_state"] = self.interim_state.copy()
        state_config_dict["state"] = self.state.copy()
        state_config_dict["done"] = deepcopy(self.done)
        state_config_dict["obs_dict"] = deepcopy(self.obs_dict)
        state_config_dict["sel_action"] = deepcopy(self.sel_action)
        state_config_dict["rbm_action"] = deepcopy(self.rbm_action)
        state_config_dict["l_bnd_vtx"] = deepcopy(self.l_bnd_vtx)
        state_config_dict["u_bnd_vtx"] = deepcopy(self.u_bnd_vtx)
        state_config_dict["vtx2ext_on_axis"] = deepcopy(self.vtx2ext_on_axis)
        state_config_dict["board_vtx_count_prior_ext"] = \
            deepcopy(self.board_vtx_count_prior_ext)
        state_config_dict["action"] = deepcopy(self.action)
        state_config_dict["source_boundary_vtx"] = deepcopy(self.source_boundary_vtx)
        state_config_dict["face_boundary_edges"] = deepcopy(self.face_boundary_edges)
        state_config_dict["triangle_inequality_holds"] = deepcopy(self.triangle_inequality_holds)
        return state_config_dict

    def map2single_agent(self):
        action_mask = self.obs_dict[self.agent]["action_mask"]
        if len(self.vertex_list) == 0:
            single_mask = action_mask.reshape(
                (self.board_length + 1,self.board_length + 1)).T
            if self.symmetry_axis == "point":
                single_mask = single_mask[:self.half_board_length + 1,:self.half_board_length + 1]
            elif self.symmetry_axis == "y-axis":
                single_mask = single_mask[:,:self.half_board_length + 1]
            single_mask = single_mask.T.reshape((-1,))
            single_mask = np.append(single_mask,single_mask)
        elif self.agent=="sel_agent":
            valid_actions = [self.points_2D[v] \
                for v in self.vertex_list if action_mask[v]==1]
            single_mask = [coords in valid_actions for coords in self.coords_list]
            single_mask = np.where(np.array(single_mask,dtype=bool),1.0,0.0).astype(np.float32)
            single_mask = single_mask.reshape((self.board_length+1,self.board_length+1))
            single_mask = single_mask.transpose()
            if self.symmetry_axis=="point":
                single_mask = single_mask[:self.half_board_length + 1,:self.half_board_length + 1]
                single_mask = single_mask.transpose()
                single_mask = single_mask.reshape((-1,))
            elif self.symmetry_axis=="y-axis":
                single_mask = single_mask[:self.board_length+1,:self.half_board_length + 1]
            single_mask = single_mask.transpose()
            single_mask = single_mask.reshape((-1,))
            single_mask = np.append(single_mask,single_mask)
        else:
            if self.symmetry_axis=="y-axis":
                single_mask = action_mask[:(self.board_length+1)*(self.half_board_length + 1)]
                single_mask = np.append(single_mask,np.zeros(single_mask.size))
            else:
                single_mask = action_mask
                single_mask = np.append(single_mask,np.zeros(self.ext_act_space_size))
        single_mask = single_mask.astype(np.float32)
        if self.symmetry and self.symmetry_axis=="point":
            obs = self.obs_dict[self.agent]["state"][:self.half_board_length + 1,:self.half_board_length + 1,:].copy()
        elif self.symmetry and self.symmetry_axis=="y-axis":
            obs = self.obs_dict[self.agent]["state"][:,:self.half_board_length + 1,:].copy()
        else:
            obs = self.obs_dict[self.agent]["state"].copy()
        self.obs_dict =  {  "obs":obs,
                            "action_mask":single_mask,}

    def set_observations(self):
        if self.done:
            self.ep_cnt += 1
            if self.mode == "DEBUG":
               print(  "TOTAL EPISODES:",self.ep_cnt,
                        "EPISODE REWARD:",sum(self.reward_history[1:]))
            self.obs_dict = {
                "sel_agent": {
                    "state": self.interim_state,
                    "action_mask": get_sel_action_mask(self),},
                "ext_agent": {
                    "state": self.interim_state,
                    "action_mask": get_ext_action_mask(self, self.sel_action),}}
        elif np.any(self.interim_state[:,:,-1]):
            action_mask = get_ext_action_mask(self, self.sel_action)
            self.obs_dict = {
                "ext_agent": {
                    "state": self.interim_state,
                    "action_mask": action_mask,}}
        else:
            action_mask = get_sel_action_mask(self)
            self.obs_dict = {
                "sel_agent": {
                    "state": self.interim_state,
                    "action_mask": action_mask,}}


    def __extend_final(self,action):
        # === third and final extension -> log in extension triplet

        self.reward_ext = 0
        self.ext_action_list.append(action)
        self.ext_action_list_c = self.__vec2coords(self.ext_action_list)
        self.board_vtx_count_prior_ext = len(self.vertex_list)
        # === perform SingleVertex extension
        foldable, state, reward, msg = self.do_ext_action(self.sel_action, self.rbm_action, self.ext_action_list_c)

        # === reset states
        self.state = state.copy()
        self.interim_state = state.copy()

        # === symmetrical expansion if env.symmetry is set
        if self.symmetry and foldable:
            foldable,state,reward,msg,roll_back = sym_expand(self, self.symmetry_axis, self.vtx2ext_on_axis)


        if self.mode == "DEBUG":
           print(foldable, msg)

        if foldable:
            # === reset states
            self.state = state.copy()
            self.ext_action_list = []
            self.ext_action_list_c = []
            self.state[:,:,-1] = 0
            self.interim_state = state.copy()
            self.state_config = self.get_state_config()
        else:
            last_valid_state_config = self.state_config
            self.set_state_config(last_valid_state_config)
        return foldable

    def __extend(self,action):
        # === first and second extension
        self.ext_action_list.append(action)
        self.ext_action_list_c = self.__vec2coords(self.ext_action_list)
        if self.symmetry and self.count == 2:
            if self.vtx2ext_on_axis=="x" and \
                    (self.symmetry_axis == "point" or self.symmetry_axis == "x-axis"):
                new_act_coords = deepcopy(self.ext_action_list_c[0])
                new_act_coords[1] = -new_act_coords[1]
                self.ext_action_list_c.insert(0,new_act_coords)
                act = self.coords_list.index(new_act_coords)
                self.ext_action_list.insert(0,act)
                self.reward_history.append(self.reward_history[-1])
            elif self.vtx2ext_on_axis=="y" and\
                    (self.symmetry_axis == "point" or self.symmetry_axis == "y-axis"):
                new_act_coords = deepcopy(self.ext_action_list_c[0])
                new_act_coords[0] = -new_act_coords[0]
                self.ext_action_list_c.append(new_act_coords)
                act = self.coords_list.index(new_act_coords)
                self.ext_action_list.append(act)
                self.reward_history.append(self.reward_history[-1])
            elif self.vtx2ext_on_axis=="-y" and\
                    (self.symmetry_axis == "point" or self.symmetry_axis == "y-axis"):
                new_act_coords = deepcopy(self.ext_action_list_c[0])
                new_act_coords[0] = -new_act_coords[0]
                self.ext_action_list_c.insert(0,new_act_coords)
                act = self.coords_list.index(new_act_coords)
                self.ext_action_list.insert(0,act)
                self.reward_history.append(self.reward_history[-1])
        for act in self.ext_action_list:
            self.interim_state = self.get_interim_state(self.sel_action,act)

    def __select(self, sel_action, rbm_action):
        # === 0th step: select vertex for extension
        self.board_vtx_count_prior_ext = len(self.vertex_list)

        # === logging sel action
        self.sel_action = sel_action

        # === indicate selected vertex in last slice of state tensor
        self.sel_board_pos = self.vertex_objs[self.sel_action].board_pos
        self.interim_state[self.sel_board_pos[0], self.sel_board_pos[1], -1] = 1

        # === logging rbm action
        self.rbm_action = rbm_action
        self.interim_state[self.sel_board_pos[0], self.sel_board_pos[1], 0] = rbm_action

        # === reset extension lists
        self.ext_action_list = []
        self.ext_action_list_c = []

    def define_symmetry(self, action):
        if self.symmetry:
            if "sel_agent" in action:
                self.vtx2ext_on_axis = "0"
                act = action["sel_agent"][0]
                vtx_coords = self.vertex_objs[act].coords
                if vtx_coords[0] == 0:
                    ind_vec = vtx_coords - self.vertex_objs[0].coords
                    if ind_vec[1]>0:
                        self.vtx2ext_on_axis = "y"
                    elif ind_vec[1]<0:
                        self.vtx2ext_on_axis = "-y"
                elif vtx_coords[1] == 0:
                    self.vtx2ext_on_axis = "x"
            elif "source_action" in action:
                pass
            else:
                action["ext_agent"] = map_symm2board_act(action["ext_agent"], self.half_board_length, self.symmetry_axis)
        return action

    def get_interim_state(self, sel_action: int, ext_action: int):
        state = self.interim_state.copy()
        v_pos = self.vertex_objs[sel_action].board_pos
        new_vtx_pos = self.map2board(ext_action)
        # === set pos on board
        state[new_vtx_pos[0],new_vtx_pos[1],0] = 1
        if new_vtx_pos in self.vertex_pos_list:
            vtx2merge = self.vertex_pos_list.index(new_vtx_pos)
            # --- update vertex2extend in interim_state tensor
            state[v_pos[0],v_pos[1],vtx2merge+1] = 1
            # --- update vertex2merge in interim_state tensor
            state[new_vtx_pos[0],new_vtx_pos[1],sel_action+1] = -1
        else:
            new_vtx_name = np.count_nonzero(state[:,:,0])-1
            # --- update vertex2extend adjacency in interim_state tensor
            state[v_pos[0],v_pos[1],new_vtx_name+1] = 1
            # --- update new vertex adjacency in interim_state tensor
            state[new_vtx_pos[0],new_vtx_pos[1],sel_action+1] = -1
        return state

    def map2board(self, pos1D: int):
        return [np.mod(pos1D,self.board_length+1),int(pos1D/(self.board_length+1))]

    def board2vec(self,board_indice):
        board = np.zeros((self.board_length+1,self.board_length+1))
        board[board_indice[0],board_indice[1]] = 1
        return board.T.reshape(-1,).tolist().index(1)

    def __vec2coords(self, p1D: int):
        # ----------------------------------------------------------------------
        #
        #   takes:      p1D: 1-dim "action_space" point coordinates
        #                   (col-wise sliced board)
        #
        #   returns:    coords: list with mapped to 2-dim coordinates,
        #                   with origin at board center
        #
        # ----------------------------------------------------------------------

        if isinstance(p1D,int):
            pts1D = [p1D]
        else:
            pts1D = p1D

        new_coords = [self.coords_list[i] for i in pts1D]
        return new_coords

    def do_ext_action(self, v: int, rbm: int, new_points: list):
        # ----------------------------------------------------------------------
        #
        #   takes:      v: vertex to extend number
        #               rbm: rigid body mode
        #               new_points: three 2-dim coordinates of new extension points
        #
        #   returns:    state: n x n x n x 2 environment state tensor
        #               [rew_1,rew_2]: float scalar rewards
        #
        # ----------------------------------------------------------------------
        while self.vertex_objs[v].is_pseudo and self.vertex_objs[v].inactive:
            v = self.vertex_objs[v].name
        # === terminate if max number of vertices reached
        if len(self.vertex_list) <= self.max_vertices-3:
            foldable, msg = self.vertex_objs[v].extend(rbm, new_points)
        else:
            foldable = False  # True
            msg = "broken symmetry due to max vertex number reached"
            self.done = True
        reward = [0,0]
        return foldable, self.state, reward, msg
    """
    def sort_points2ext(self, v: int, new_points: list):
        # sorting new nodes coordinates for extension
        new_vec = np.array(new_points) - np.array(self.points_2D[v])
        test_angles = []
        for vec in new_vec.tolist():
            print("KM ------------------")
            a = km.planar_angle(self.u_bnd_vec.tolist(),vec)
            if a<0: a = 2*np.pi+a
            test_angles.append(a)
        test_angles_zip = list(zip(test_angles,range(3)))
        test_angles_zip.sort(reverse=True)
        test_angles.sort(reverse=True)
        new_points_sorted = []
        indices = []
        for a,i in test_angles_zip:
            new_points_sorted.append(new_points[i])

        return [new_points_sorted[0],new_points_sorted[2],new_points_sorted[1]]
    """
    def compute_reward(self, final=False, final_extension=False):
        # ----------------------------------------------------------------------
        #
        #   takes:      points3D: 3-dim point cloud as a function of psi
        #
        #   ->          computes the optimum fold angle psi for shape approximation
        #
        #   returns:    reward_1: max HausdorffDistance HD between sampled points
        #               of target mesh and folded mesh
        #
        # ----------------------------------------------------------------------

        # === length of taken decision chain
        # trace = len(self.decision_chain)

        # === compute reward
        if (not self.foldable) or (self.done and (0<len(self.ext_action_list)<3)):
            # === ist Var gesetzt? TODO
            # === non foldable rewards
            rew = -self.board_length
        elif not final and not final_extension:  # (not final) and trace > 0 and (not self.count==3):
            # === intermediate rewards
            rew = 0
        elif self.optimize_psi:
            # wrapped reward function here
            reward_func = reward_wrapper(self.reward_function)
            rew,opt_idx = reward_func(self.vertex_objs, self.all_3D_points,
                self.triangle_inequality_holds, faces=self.triangles,  # faces=self.faces,
                reward_history=self.reward_history, done=self.done)
            self.opt_idx = opt_idx
        else:
            rew = self.reward_function(self.vertex_objs, self.points_3D, self.triangles,  # self.faces,
                                       self.reward_history, self.done)
        return rew

    def save_folded_mesh(self, all=False):
        points_3D = self.all_3D_points.swapaxes(0,1)
        if all:
            indices = list(range(len(self.psi_opt) - 1, self.opt_idx - 1, -1))
        else:
            indices = [self.opt_idx]
        for idx in indices:
            _, _, folded_mesh = get_folded_mesh_points(points_3D[idx], self.triangles)  #self.faces)
            save_dir = str(self.config["save_dir"])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dt = datetime.now()
            filename = save_dir+"/"+\
                dt.strftime("%y-%m-%d_%H-%M-%S.%f")+\
                "_numvertices="+str(len(self.vertex_list))+\
                "_psi="+str(self.folding_direction*self.psi_opt[idx])+\
                "_rew="+str(round(sum(self.reward_history),2))+\
                "_ep="+str(self.ep_cnt)+\
                "_totalsteps="+str(self.total_step_count)+".obj"
            if all:
                filename = save_dir + "/folding_motion" +str(idx) +\
                           "_at_psi=" + str(self.folding_direction*self.psi_opt[idx]) + ".obj"
            trimesh.exchange.export.export_mesh(folded_mesh, filename)
        fold_angle = self.folding_direction*self.psi_opt[self.opt_idx]
        pattern = list([self.decision_chain,round(sum(self.reward_history),2),fold_angle])
        self.store_pattern_and_foldangle_values(save_dir,pattern,fold_angle)

    def store_pattern_and_foldangle_values(self,save_dir,pattern,fold_angle):
        with open(save_dir+"/.patterns.ob", 'rb') as f:
            data = pickle.load(f)
        f.close()
        data.append(pattern)
        with open(save_dir+"/.patterns.ob", 'wb') as f:
            pickle.dump(data,f)
        f.close()


    def fold_pattern(self, fold_angle=0.0001, show_target=1,
        pattern=False, alpha=0.25, anim=False, silent=True, view=[90,-90,23]):
        # ----------------------------------------------------------------------
        #
        #   takes:      fold_angle: global driving angle psi value for simulation
        #               show_target: boolean 1 show, 0 do not show target
        #
        #   returns:    two plot windows:
        #                   (1) crease pattern as directed graph
        #                   (2) 3D folding simulation of mesh
        #
        # ----------------------------------------------------------------------

        if pattern:
            plot_pattern(self,silent=silent)
            plot_polygons(self,fold_angle,show_target,alpha,anim,silent=silent)
        else:
            plot_polygons(self,fold_angle,show_target,alpha,anim,silent)

        if not silent:
            plt.show()
        else:
            plt.close(fig='all')

    def __create_base(self):
        if self.base == "quad":
            quad_config(self)
        elif self.base == "simple":
            simple_config(self)
        elif self.base == "simple_vert":
            simple_vert_config(self)
        elif self.base=="single":
            single_config(self)

    def get_ava_actions(self,state):
        if np.any(state[:,:,-1]):
            mask = get_ext_action_mask(self, self.sel_action)
        else:
            mask = get_sel_action_mask(self)
        if np.any(self.interim_state[:,:,-2]):
            self.done = True
            mask = np.zeros(np.shape(mask))
        return mask
