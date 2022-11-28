#!/usr/bin/env python
# encoding: utf-8

from gym_rori.envs.rori_env import RoriEnv
from ray.tune.registry import register_env

import os,ray,gym,argparse,pickle
import warnings

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    from ray import tune
    from ray.rllib.models import ModelCatalog
    #from PPO.ResNet_PPO import ActorCriticModel
    from ray.rllib.utils import try_import_tf
    from ray.rllib.models.tf.tf_modelv2 import TFModelV2
    from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
    from ray.rllib.utils.typing import ModelConfigDict
    import tensorflow as tf

    import numpy as np
    from time import strftime
    from gif_generator import generate_gif, generate_gif2
    from mcts import MCTS, RootParentNode, Node, DummyModel
    import dfts, bfts
    from copy import deepcopy


def main():

    parser = argparse.ArgumentParser(description='Process environment config.')
    
    parser.add_argument("--name", default="0",type=str)
    parser.add_argument("--num-steps", default=500, type=int)
    parser.add_argument("--objective", choices=[
        'shape-approx', 
        'packaging', 
        'chair', 
        'table', 
        'shelf', 
        'bucket'],
        default='shape-approx')
    parser.add_argument("--search-algorithm", 
        choices=['RDM', 'DFTS', 'BFTS', 'MCTS', 'PPO', 'evolution', 'human'], 
        default="RDM", 
        type=str)
    parser.add_argument("--base", 
        choices=["plain", "quad", "single", "simple", "simple_vert"], 
        default="plain", 
        type=str)
    parser.add_argument("--start-sequence", nargs='*', default=[])
    parser.add_argument("--seed-pattern-size", default=2, type=int)
    parser.add_argument("--board-length", default=12, type=int)
    parser.add_argument("--num-symmetries", default=2, type=int)
    parser.add_argument("--max-vertices", default=100, type=int)
    parser.add_argument("--psi", default=3.14, type=float)
    parser.add_argument("--optimize-psi", action='store_true', default=True)
    parser.add_argument("--cl-max", default=np.inf, type=float)
    parser.add_argument("--seed", default=16711, type=int) # branching factor for tree searches
    parser.add_argument("--allow-source-action", action='store_true', default=False)

    parser.add_argument("--target", default="target.obj", type=str)
    parser.add_argument("--target-transform", nargs='+', default=[0, 0, 0], type=float)
    parser.add_argument("--auto-mesh-transform", action='store_true', default=False)
    parser.add_argument("--count-interior", action='store_true', default=False)

    parser.add_argument("--mode", default="TRAIN", type=str)
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--local-dir", default=os.getcwd(), type=str)

    parser.add_argument("--bf", default=10, type=int) # branching factor for tree searches

    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--training-iteration", default=100, type=int)
    parser.add_argument("--ray-num-cpus", default=1, type=int)
    parser.add_argument("--ray-num-gpus", default=0, type=int)
    
    parser.add_argument("--anim-view", nargs='+', default=[90, -90, 23], type=float)


    args = parser.parse_args()

    # round board size to even number
    if args.board_length % 2 == 1:
        args.board_length -= 1 
        

    env_config = {}
    local_dir = args.local_dir

    # disable prints
    #if not args.mode == "DEBUG":
    #    sys.stdout = open(os.devnull, 'w')

    if not os.path.exists(local_dir+"/results"):
        os.makedirs(local_dir+"/results")
    if args.name == "0":
        name = strftime("%Y-%m-%d_%H-%M-%S_exp_")+str(args.search_algorithm)+"_"+args.objective
    else:
        name = args.name
    save_dir = local_dir+"/results/"+name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not args.resume:
        data = []
        with open(save_dir+"/.patterns.ob", 'wb') as f:
            pickle.dump(data, f)
        with open(save_dir+"/.env_state.ob", 'wb') as f:
            pickle.dump(data, f)


    symmetry = True
    if args.num_symmetries == 0:
        symmetry = False
        symmetry_axis = "0"
        cross_symmetry_axis = False
    elif args.num_symmetries == 1:
        symmetry_axis = "y-axis"
        cross_symmetry_axis = False
    elif args.num_symmetries == 2:
        symmetry_axis = "point"
        cross_symmetry_axis = False
    else:
        symmetry_axis = "point"
        cross_symmetry_axis = True

    translational_transform = np.array([item for item in args.target_transform])
    target_transform = np.identity(4)
    target_transform[:-1, -1] = translational_transform
    view = np.array([item for item in args.anim_view])
    if args.objective == 'shape-approx':
        from rewarders import shaped_hausdorff_distance_rewarder
        rewarder = shaped_hausdorff_distance_rewarder
    elif args.objective == 'packaging':
        from rewarders import packaging_rewarder
        rewarder = packaging_rewarder
    elif args.objective == 'chair':
        from rewarders import chair_rewarder
        rewarder = chair_rewarder
    elif args.objective == 'table':
        from rewarders import table_rewarder
        rewarder = table_rewarder
    elif args.objective == 'shelf':
        from rewarders import shelf_rewarder
        rewarder = shelf_rewarder
    else:
        from rewarders import bucket_rewarder
        rewarder = bucket_rewarder

    # fill environment config dict
    env_config.update({
        # "rewarder": shaped_hausdorff_distance_rewarder if args.shape_approx else packaging_rewarder,
        "rewarder": rewarder,
        "local_dir": local_dir,
        "save_dir": save_dir,
        "target_file": local_dir + '/' + args.target,
        "PSI": args.psi,
        "TS": args.seed_pattern_size,
        "base": args.base,
        "intermediate_reward": True,
        "max_vertices": args.max_vertices,
        "mode": args.mode,
        "CL_MAX": args.cl_max,
        "symmetry_axis": symmetry_axis,
        "cross_axis_symmetry": cross_symmetry_axis,
        "symmetry": symmetry,
        "target_transform": target_transform,
        "board_length": args.board_length,
        "optimize_psi": args.optimize_psi,
        "count_interior": args.count_interior,
        "allow_source_action": args.allow_source_action,
        "auto_mesh_transform": args.auto_mesh_transform,
        "start_sequence": args.start_sequence,
    })

    if args.search_algorithm == "PPO":
        env = run_PPO(env_config, args)
    elif args.search_algorithm == "human":
        env = run_human(env_config, args)
    elif args.search_algorithm == "evolution":
        env = run_evolution(env_config, args)
    elif args.search_algorithm == "MCTS":
        env = run_mcts(env_config, args)
    elif args.search_algorithm == "DFTS":
        env = run_dfts(env_config, args)
    elif args.search_algorithm == "BFTS":
        env = run_bfts(env_config, args)
    else:
        env = run_RDM(env_config, args)

    with open(save_dir+'/.patterns.ob', 'rb') as file:
        patterns = pickle.load(file)
        rew_max = -np.inf
        for pat, rew, fold_angle in patterns:
            if rew > rew_max:
                rew_max = rew
                best_pattern = pat
                opt_fold_angle = fold_angle

    print("Found pattern of highest return: ", rew_max,
          "with optimal fold angle:", str(opt_fold_angle),
          "in " + env_config["local_dir"]+"/results")

    resolution = 10
    generate_gif2(env.config, opt_fold_angle, best_pattern, resolution, view=view)


def run_human(env_config, args):
    # human play

    # build environment
    # env_config.update({'show_agent': True})
    env = RoriEnv(env_config)
    # observations
    obs = env.reset()
    done = False
    ep_reward = 0
    total_step_count = 0

    while total_step_count < args.num_steps:
        ava_actions = obs["action_mask"]
        discrete_actions = np.arange(ava_actions.size)
        total_step_count += 1
        if done or not np.any(ava_actions):
            print('done', ep_reward)
            print('foldable?', env.foldable)
            if ep_reward > env.best_episode_reward:  # and env_config["mode"] == "DEBUG":
                print("JUST SEEN BEST EPISODE", ep_reward, env.decision_chain)
            print('Do you want to save all fold angles?')
            answer = str(input())
            if answer in ('y', 'yes'):
                env.save_folded_mesh(all=True)

            # reset
            env.best_episode_reward = -np.inf
            done = False
            ep_reward = 0
            env.reset()
            obs = env.obs_dict
        else:
            # sample action
            print(env.triangle_inequality_holds)
            print(env.pseudo_vertices)
            print(np.sum(np.abs(env.interim_state) + np.abs(env.state), axis=-1))
            if env.symmetry_axis == "point":
                length = args.board_length // 2 + 1
                print(np.arange(length**2).reshape(length, length).transpose())
            else:
                length = args.board_length + 1
                print(np.arange(length**2).reshape(length, length).transpose())
            print(discrete_actions[ava_actions.astype(bool)])
            action = int(input())
            if action == -1:
                break
            # take step
            obs, rew, done, info = env.step(action)
            # count total episode reward
            ep_reward += rew
    return env


def run_dfts(env_config,args):
    np.random.seed(args.seed)
    bf_max = args.bf # max branching factor
    env = RoriEnv(env_config)
    save_dir = str(env.config["save_dir"])
    if args.resume:
        with open(save_dir+"/.env_state.ob", 'rb') as f:
            state_config,new_seed,env_best_reward,env_step_count,ctr = pickle.load(f)
            np.random.set_state(new_seed)
            f.close()
        env.set_state_config(state_config)
        env.best_episode_reward = env_best_reward
        env.total_step_count = env_step_count
    else:
        state_config = env.get_state_config()
        ctr = 0
    while env.total_step_count < args.num_steps:
        print("TOTAL STEPS:",env.total_step_count)
        ctr += 1
        max_steps = ctr*args.num_steps/100 # max steps per trial
        root = dfts.Node(0,env.obs_dict,0,False,"rootparent",env)
        best_reward = dfts.tree_search(root,max_steps,bf_max)
        env.reset()
        data = [
            env.get_state_config(),
            np.random.get_state(),
            env.best_episode_reward,
            env.total_step_count,
            ctr]
        # data.append(env.get_state_config())
        # data.append(np.random.get_state())
        # data.append(env.best_episode_reward)
        # data.append(env.total_step_count)
        # data.append(ctr)
        with open(save_dir+"/.env_state.ob", 'wb') as f:
            pickle.dump(data,f)
            f.close()
    return env


def run_bfts(env_config, args):
    np.random.seed(args.seed)
    bf_max = args.bf
    best_reward = -np.inf
    env = RoriEnv(env_config)
    save_dir = str(env.config["save_dir"])
    if args.resume:
        with open(save_dir+"/.env_state.ob", 'rb') as f:
            state_config,new_seed,env_best_reward,env_step_count,ctr = pickle.load(f)
            np.random.set_state(new_seed)
            f.close()
        env.set_state_config(state_config)
        env.best_episode_reward = env_best_reward
        env.total_step_count = env_step_count
    else:
        state_config = env.get_state_config()
        ctr = 0
    while env.total_step_count < args.num_steps:
        print("TOTAL STEPS:",env.total_step_count)
        ctr += 1
        max_steps = ctr*args.num_steps/100 # max steps per trial
        root = bfts.Node(0,env.obs_dict,0,False,"rootparent",env)
        best_reward = bfts.tree_search(root,max_steps,best_reward,bf_max)
        env.reset()
        data = [
            env.get_state_config(),
            np.random.get_state(),
            env.best_episode_reward,
            env.total_step_count,
            ctr]
        with open(save_dir+"/.env_state.ob", 'wb') as f:
            pickle.dump(data,f)
            f.close()
    return env


def run_mcts(env_config,args,mcts_new_config={}):
    np.random.seed(args.seed)
    mcts_config={
        "temperature":1.5,
        "puct_coefficient": 1.0,
        "add_dirichlet_noise": True,
        "dirichlet_noise": 0.03,
        "dirichlet_epsilon": 0.25,
        "num_simulations": 100,
        "alphazero":True,
        "alpha_zero_tree_policy": True,
        "follow_best_path": False,
        "preserve_tree": True,
        "argmax_tree_policy": True,
        "use_value_net": False}
    mcts_config.update(mcts_new_config)
    env = RoriEnv(env_config)
    model = DummyModel()
    while env.total_step_count<args.num_steps:
        mcts = MCTS(model,mcts_config)
        root_parent = RootParentNode(env)
        state_config = env.get_state_config()
        root_node = Node(
            action=0,
            obs = deepcopy(env.obs_dict),
            done=False,
            reward=0,
            state=deepcopy(state_config),
            mcts=mcts,
            parent=root_parent)

        tree_policy, action, next_node = mcts.compute_action(root_node)
        actions=[]
        mcts_config["iter"] = 1
        actions.append(action)
        while not (next_node.done or not np.any(next_node.valid_actions)) and \
            env.total_step_count<args.num_steps:
            mcts_config["iter"] += 1
            if not mcts_config["preserve_tree"]:
                del mcts
                mcts = MCTS(model,mcts_config)
                root_node = Node(
                    action=deepcopy(next_node.action),
                    obs=deepcopy(next_node.env.obs_dict),
                    done=False,
                    reward=deepcopy(next_node.reward),
                    state=deepcopy(next_node.state),
                    mcts=mcts,
                    parent=root_parent)
                _,action,next_node = mcts.compute_action(root_node)
            else:
                del next_node.parent
                next_node.parent = root_parent
                _,action,next_node = mcts.compute_action(next_node)
            actions.append(action)
        del mcts
        env.reset()
    return env


def run_evolution(env_config, args):
    np.random.seed(args.seed)
    population_size = 128
    env = RoriEnv(env_config)

    def run_episode(raw_agent):
        # env, agent = env_and_agent
        obs = env.reset()
        ava_actions = obs["action_mask"]
        ep_reward = 0
        step = 0
        done = False
        action_seq = []
        while not done or np.any(ava_actions):
            if np.any(obs['obs'][:, :, -1]) or step == 0:
                agent = raw_agent[:len(raw_agent)//2]
            else:
                agent = raw_agent[len(raw_agent)//2:]
            masked_agent = np.where(ava_actions, agent, -np.inf)
            action = np.argmax(masked_agent)
            # take step
            obs, rew, done, info = env.step(action)
            ava_actions = obs["action_mask"]
            # count total episode reward
            ep_reward += rew
            step += 1
            action_seq.append(action)
        return ep_reward, step, action_seq
    # envs = np.asarray([RoriEnv(env_config) for _ in range(population_size)])
    # population = np.asarray([np.random.randn(envs[0].action_space.n) for _ in range(population_size)])
    num_actions = env.action_space.n * 2  # separate for selection and expansion phase
    population = np.asarray([np.random.randn(num_actions) for _ in range(population_size)])
    total_steps = 0
    # with Pool(population_size) as p:
    while total_steps < args.num_steps:
        # result = list(map(run_episode, zip(envs, population)))
        result = list(map(run_episode, population))
        returns, steps, action_seqs = zip(*result)
        returns = np.asarray(returns)
        for idx, action_seq in enumerate(action_seqs):
            if action_seq in action_seqs[:idx]:
                returns[idx] = -np.inf
        print(returns)
        total_steps += sum(steps)
        print(total_steps)
        tmp = np.argsort(returns)[::-1]
        top_agents = tmp[:population_size//4]
        print(sum(returns[top_agents]))
        parents = population[top_agents] + 0.1 * np.random.randn(population_size//4, num_actions)
        offspring1 = parents + 0.5 * np.random.randn(population_size//4, num_actions)
        offspring2 = parents + 1 * np.random.randn(population_size//4, num_actions)
        new_comers = np.random.randn(population_size//4, num_actions)
        population = np.concatenate([parents, offspring1, offspring2, new_comers])
    return env


def run_RDM(env_config,args):
    # uniform random policy
    np.random.seed(args.seed)

    # build environment
    env = RoriEnv(env_config)

    # observations
    obs = env.reset()
    done = False
    ep_reward = 0
    total_step_count = 0
    best_ep_reward = - np.inf

    print(
        "================================================================= \n",
        "Starting origami game with parameters: \n",
        "OBJECTIVE:", args.objective, "\n", 
        "TOTAL-ENVIRONMENT-INTERACTIONS:",args.num_steps, "\n",
        "BOARD-SIZE:", args.board_length+1, "x", args.board_length+1, "\n",
        "SYMMETRY-AXES:", args.num_symmetries, "\n",
        "RANDOM-SEED:", args.seed, "\n",
        "================================================================= \n",
        )

    while total_step_count < args.num_steps:
        if total_step_count % 1000 == 0:
            print('Environment interactions:', total_step_count, 'of', args.num_steps)
        ava_actions = obs["action_mask"]
        discrete_actions = np.arange(ava_actions.size)
        total_step_count += 1
        if done or not np.any(ava_actions):
            if ep_reward > best_ep_reward:
                print(
                    "Just seen best episode at STEP", 
                    total_step_count,
                    "with EPISODE-REWARD", 
                    round(ep_reward, 3), 
                    "and STEP-SEQUENCE", 
                    env.decision_chain)
            best_ep_reward = max(ep_reward, best_ep_reward)
            # reset
            done = False
            ep_reward = 0
            env.reset()
            obs = env.obs_dict
        else:
            # sample action
            action = np.random.choice(discrete_actions[ava_actions.astype(bool)])
            # take step
            obs, rew, done, info = env.step(action)
            # count total episode reward
            ep_reward += rew
            # if ep_reward < best_ep_reward:
            #     done = True
    return env


def run_PPO(env_config,args):
    ray.init(num_cpus=args.ray_num_cpus, num_gpus=args.ray_num_gpus,
             ignore_reinit_error=True, local_mode=False)

    # -- register custom action mask models in RLlib model catalog
    ModelCatalog.register_custom_model(
        "rori_model", ROriMLPActionMaskModel)

    # register the custom environment
    register_env("Rori-v0", RoriEnv) #lambda config: RoriEnv(config))

    tune.run(
        "PPO",
        stop={
            "training_iteration": args.training_iteration,
            "timesteps_total": args.num_steps,
        },
        local_dir = env_config["save_dir"],
        name = strftime("%Y-%m-%d_%H-%M-%S_exp"),
        checkpoint_freq = 1000,
        checkpoint_at_end=True,
        resume=False,
        config = {
            "env": "Rori-v0",
            "batch_mode": "complete_episodes",
            "num_gpus": 0,
            "env_config": env_config,
            "num_workers": args.num_workers,
            "horizon": None,
            "entropy_coeff": 0.01,
            "evaluation_num_episodes": 1000,
            "rollout_fragment_length": 500,
            "sgd_minibatch_size": 64,
            "train_batch_size": 2050,
            "explore": True,
            "framework":"tf",
            "log_level":"DEBUG",
#            "lr": 1e-6,
            "seed": args.seed,
            "model": {
                "custom_model": "rori_model",}
        })
    return RoriEnv(env_config)


class ROriMLPActionMaskModel(TFModelV2):

    def __init__(   self,
                    obs_space: gym.spaces.Space,
                    action_space: gym.spaces.Space,
                    num_outputs: int,
                    model_config: ModelConfigDict,
                    name: str,
                    **customized_model_kwargs):

        super(ROriMLPActionMaskModel, self).__init__(
            obs_space,
            action_space, num_outputs, model_config, name,
            **customized_model_kwargs)

        self.mlp_model = FullyConnectedNetwork(
            obs_space,
            action_space, num_outputs,
            model_config, name + "_sel_action_masking")
        self.register_variables(self.mlp_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        state_obs = input_dict["obs"]["obs"]
        shape = state_obs.get_shape().as_list()
        dim = np.prod(shape[1:])
        state_obs = tf.reshape(state_obs, [-1, dim])
        input_state = tf.concat([state_obs,action_mask],-1)


        # compute the action output from the fully connected network(s)
        model_output,_ = self.mlp_model({"obs": input_state})

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        return model_output+inf_mask, state

    def value_function(self):
        return self.mlp_model.value_function()



if __name__ == "__main__":
    main()
