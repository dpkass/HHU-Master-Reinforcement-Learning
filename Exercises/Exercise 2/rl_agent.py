# author: Jan Robine
# date:   2023-04-25
# course: reinforcement learning

import copy
import numpy as np


class Agent:

    def __init__(self, env):
        self.env = env

    def policy(self, state):
        # Needs to be implemented by a subclass
        raise NotImplementedError()
    
    def value(self, state):
        # Needs to be implemented by a subclass
        raise NotImplementedError()


# Agent with a uniform policy (i.e. random actions).
class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        # Copy space so we don't change the state of the environment's RNG
        self.action_space = copy.deepcopy(env.action_space)

    def policy(self, state):
        # Sample a random action
        return self.action_space.sample()
    
    def value(self, state):
        # This agent does not support values
        return None


# Base class for tabular agents, i.e., for environments
# with small discrete state and action spaces.
class TabularAgent(Agent):

    @property
    def num_states(self):
        return self.env.observation_space.n

    @property
    def num_actions(self):
        return self.env.action_space.n


# Agent with a random policy and random values, only for testing purposes.
class RandomValueAgent(TabularAgent):

    def __init__(self, env):
        super().__init__(env)
        # Generate random values
        self.v = np.random.randn(self.num_states)

    def policy(self, state):
        # Sample a random action
        return np.random.choice(self.num_actions)

    def value(self, state):
        return self.v[state]
