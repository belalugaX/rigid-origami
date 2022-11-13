import numpy as np
from copy import deepcopy
import time as t
def tree_search(node,max_steps,bf_max):
    current = node
    count = 0
    local_dir = node.env.config["local_dir"]
    best_reward = current.env.best_episode_reward
    while (not current=="rootparent") and current.env.total_step_count<max_steps:
        if np.any(current.action_mask) and not current.done and \
            current.bf<=bf_max and current.reward>best_reward:
            current = current.get_child()
        elif current.reward<=best_reward:
            current.action_mask = (current.action_mask*0).astype(bool)
            act = deepcopy(current.action)
            if current.parent=="rootparent":
                break
            else:
                current = current.parent
                del current.children[act]
                current.action_mask[act] = False
                current.env.set_state_config(current.state_config)
        else:
            best_reward = current.env.best_episode_reward
            act = deepcopy(current.action)
            current = current.parent
            if not current=="rootparent":
                current.action_mask[act] = False
                current.env.set_state_config(current.state_config)
    return best_reward


class Node():
    def __init__(self,action,obs,reward,done,parent,env):
        self.env = env
        self.action = action
        self.obs = deepcopy(obs)
        self.parent = parent
        self.children = {}
        self.action_mask = deepcopy(obs["action_mask"].astype(bool))
        self.potential_actions = np.arange(self.action_mask.size)
        self.reward = deepcopy(sum(env.reward_history))
        state_config = env.get_state_config()
        self.state_config = deepcopy(state_config)
        self.done = deepcopy(done)
        self.bf = 0


    def get_child(self):
        # sample action
        act = np.random.choice(
            self.potential_actions[self.action_mask])
        obs_dict,_,done,info = self.env.step(act)
        rew = sum(self.env.reward_history)
        child = Node(act,obs_dict,rew,done,self,self.env)
        self.children[act] = child
        self.bf += 1
        return child
