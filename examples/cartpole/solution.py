import gym
import torch

import argparse

import sys
sys.path.append('../../..')

from machine_learning.agent import Agent

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
    env = gym.make('CartPole-v1')
    state, _ = env.reset()

    # Get number of actions and state observations
    n_actions = env.action_space.n
    n_observations = len(state)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize an agent
    agent = Agent(n_observations, n_actions, device=device)


    if eval:
        # Load existing model
        agent.policy_net.load(filepath)

    for i_episode in range(n_episodes):

        score = agent.play_episode(env, eval)
        print('Game:', i_episode, 'Score:', score)

    if not eval:
        # Save trained model
        agent.policy_net.save(filepath)

    print('Complete')

main()