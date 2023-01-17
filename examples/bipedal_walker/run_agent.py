import sys
import gym
import argparse

import torch
import numpy as np

sys.path.append('../../..')
from machine_learning.agent import Agent

class DiscreteBipedalWalkerEnv:
    def __init__(self, render_mode='rgb_array'):
        self._env = gym.make('BipedalWalker-v3', render_mode=render_mode)

        self._dof = 4
        self._discretization = 2

        self._discrete_action_space = np.linspace([-1.0] * self._dof, [1.0] * self._dof, self._discretization)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        action = get_action_from_discrete(action, self._discrete_action_space)
        return self._env.step(action)

def get_action_from_discrete(action, discrete_action_space):
    dof = discrete_action_space.shape[1]
    discretization = discrete_action_space.shape[0]

    final_action = np.zeros(4, dtype=np.float32)
    for joint in range(dof):
        exp = joint
        power = discretization**exp

        action_space_idx = (action // power) % discretization
        final_action[joint] = discrete_action_space[action_space_idx][joint]

    return final_action

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains/evaluates a learning model playing the Snake game')

    parser.add_argument('-f', '--filepath', default='model.pth', help='An optional argument for specifying file name of the model to store or load')
    parser.add_argument('-e', '--eval', action='store_true', help='An optional argument for specifying the program to run in evaluation mode instead of training mode')
    parser.add_argument('-g', '--gui', action='store_true', help='An optional argument for specifying the program to run with a GUI')

    args = parser.parse_args()

    return args.filepath, args.eval, args.gui

def main():

    # Read command-line arguments
    filepath, eval, gui = parse_arguments()

    n_episodes = 300

    # Construct the environment
    env = DiscreteBipedalWalkerEnv(render_mode='human' if gui else "rgb_array")
    state, _ = env.reset()

    # Get number of actions and state observations
    n_actions = 2
    n_observations = len(state)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize an agent
    agent = Agent(n_observations, n_actions, device=device)

    if eval:
        # Load existing model
        agent.policy_net.load(filepath)

    try:
        for i_episode in range(n_episodes):

            score = agent.play_episode(env, eval)
            print('Game:', i_episode, 'Score:', score)
    except KeyboardInterrupt:
        print('\tStopping training/evaluation...')

    if not eval:
        # Save trained model
        agent.policy_net.save(filepath)

    print('Complete')

main()