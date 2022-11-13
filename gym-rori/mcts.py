"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math
import numpy as np
from copy import deepcopy

class Node:
    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        self.env = parent.env
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros(
            [self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)  # N
        self.valid_actions = obs["action_mask"].astype(np.bool).copy()

        self.reward = sum(self.env.reward_history)
        self.done = deepcopy(done)
        self.state = deepcopy(state)
        self.obs = deepcopy(obs)

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * self.child_priors / (
            1 + self.child_number_visits)

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        return np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state_config(self.state)
            obs, reward, done, _ = self.env.step(action)
            next_state = self.env.get_state_config()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts)
        else:
            self.env.set_state_config(self.children[action].state)
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self,model,mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

    def norm_reward(self,reward, min_reward):
        norm_reward = 2*reward/min_reward+1
        return max(norm_reward,-1)


    def simulate(self,leaf):
        discrete_actions = np.arange(leaf.valid_actions.size)
        valid_sim_actions = discrete_actions[leaf.valid_actions]
        valid_priors = leaf.child_priors[leaf.valid_actions]
        p = valid_priors/valid_priors.sum()
        action = np.random.choice(valid_sim_actions, p=p)
        obs,rew,done,info = leaf.env.step(action)
        while not done:
            ava_actions = obs["action_mask"]
            action = np.random.choice(discrete_actions[ava_actions.astype(bool)])
            obs,rew,done,info = leaf.env.step(action)
        sim_reward = sum(leaf.env.reward_history)
        value = self.norm_reward(sim_reward,leaf.env.board_length)
        return value

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = self.norm_reward(leaf.reward,node.env.board_length)
            else:
                child_priors, _ = self.model.compute_priors_and_value(
                    leaf.obs)
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size)
                leaf.expand(child_priors)
                if np.any(leaf.valid_actions):
                    value = self.simulate(leaf)
                else:
                    leaf.done = True
                    leaf.reward = -leaf.env.board_length
                    value = -1
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(
                np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]


class DummyModel():
    def compute_priors_and_value(self,obs):
        action_mask = obs["action_mask"]
        state = obs["obs"]
        priors =  np.ones(action_mask.size)/np.count_nonzero(action_mask)
        return priors,0
