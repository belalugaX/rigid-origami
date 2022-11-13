import numpy as np
from copy import deepcopy

def tree_search(node,max_steps,best_reward,bf_max):
    current = node
    count = 0
    local_dir = node.env.config["local_dir"]
    while (not current=="rootparent") and current.env.total_step_count<max_steps:
        best_reward = current.env.best_episode_reward
        if np.any(current.action_mask) and (not current.done) and \
            current.bf<=bf_max and current.reward>=best_reward-0.1:
            child = current.get_child()
            current.env.set_state_config(current.state_config)
        elif current.done and current.reward>best_reward:
            current.action_mask = (current.action_mask*0).astype(bool)
            with open(local_dir+"/patterns.txt", 'a') as f:
                f.write("TOTAL STEP COUNT="+str(current.env.total_step_count)+"_best_rew="+str(current.reward))
                f.writelines('\n')
            act = deepcopy(current.action)
            current = current.parent
            del current.children[act]
            current.action_mask[act] = False
            if current=="rootparent":
                break
            else:
                current.env.set_state_config(current.state_config)
        elif current.reward<best_reward:
            current.action_mask = (current.action_mask*0).astype(bool)
            act = deepcopy(current.action)
            if current.parent=="rootparent":
                break
            else:
                current = current.parent
                del current.children[act]
                current.action_mask[act] = False
                current.env.set_state_config(current.state_config)
        elif any(current.children) and \
            np.any(current.child_rewards[current.action_mask]>=best_reward-0.1):
            current = current.sample_best_child()
            current.env.set_state_config(current.state_config)
        else:
            act = deepcopy(current.action)
            current = current.parent
            if current=="rootparent":
                break
            else:
                current.action_mask[act] = False
                del current.children[act]
                current.env.set_state_config(current.state_config)
    return best_reward


class Node():
    def __init__(self,action,obs,reward,done,parent,env):
        self.env = env
        self.action = deepcopy(action)
        self.obs = deepcopy(obs)
        self.parent = parent
        self.children = {}
        self.action_mask = deepcopy(obs["action_mask"].astype(bool))
        self.potential_actions = np.arange(self.action_mask.size)
        self.reward = deepcopy(sum(env.reward_history))
        state_config = env.get_state_config()
        self.state_config = deepcopy(state_config)
        self.done = deepcopy(done)
        self.child_rewards = np.ones(self.action_mask.size)-env.board_length
        self.bf = 0


    def get_child(self):
        # sample action
        act = np.random.choice(
            self.potential_actions[self.action_mask])
        obs_dict,_,done,info = self.env.step(act)
        rew = deepcopy(sum(self.env.reward_history))
        child = Node(act,obs_dict,rew,done,self,self.env)
        self.children[act] = child
        self.bf += 1
        self.child_rewards[act] = rew
        return child

    def sample_best_child(self):
        self.child_rewards[~self.action_mask] = -np.inf
        child_rewards = self.child_rewards[self.action_mask]
        if np.all(child_rewards==0):
            choice = np.random.choice(list(self.children.keys()))
            best_child = self.children[choice]
        else:
            best_action = np.argmax(self.child_rewards)
            best_child = self.children[best_action]
        return best_child
