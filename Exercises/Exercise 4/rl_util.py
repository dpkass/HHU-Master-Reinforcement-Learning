# author: Jan Robine
# date:   2023-04-03
# course: reinforcement learning

import copy
import gymnasium as gym
import numpy as np



def rollout(env, agent, max_steps=None):
    states = []         # Trajectory of states
    actions = []        # Trajectory of actions
    rewards = []        # Trajectory of rewards
    terminated = False  # Episode completed in terminal state
    truncated = False   # Episode truncated after max_steps

    # Reset the environment
    state, info = env.reset()
    states.append(state)

    while not (terminated or truncated):
        # Select action and perform an environment step
        action = agent.policy(state)
        state, reward, terminated, truncated, info = env.step(action)

        # Add data to trajectories
        actions.append(action)
        rewards.append(reward)
        states.append(state)

        # Truncate episode if necessary
        if max_steps is not None and len(rewards) >= max_steps:
            truncated = True

    # Return the episode tuple
    episode = {'states':  states, 
               'actions': actions, 
               'rewards': rewards, 
               'terminated': terminated, 
               'truncated':  truncated}
    return episode
