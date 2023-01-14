import sys
import yaml
import gym
import argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../..')
from machine_learning.agent import Agent

def get_action_from_discrete(discrete_action, discrete_action_space):
    action_idx = np.argmax(discrete_action)

    dof = discrete_action_space.shape[1]
    discretization = discrete_action_space.shape[0]

    action = np.zeros(4, dtype=np.float32)
    for joint in range(dof):
        exp = joint
        power = discretization**exp

        action_space_idx = (action_idx // power) % discretization
        action[joint] = discrete_action_space[action_space_idx][joint]

    return action

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains/evaluates a learning model playing the Snake game')

    parser.add_argument('-f', '--filename', default='model.pth', help='An optional argument for specifying file name of the model to store or load')
    parser.add_argument('-e', '--eval', action='store_true', help='An optional argument for specifying the program to run in evaluation mode instead of training mode')
    parser.add_argument('-g', '--gui', action='store_true', help='An optional argument for specifying the program to run with a GUI')

    args = parser.parse_args()

    return args.filename, args.eval, args.gui

def train(num_observation, torque_limit, discretization, dof, gui=False, filename='./model.pth'):
    discrete_action_space = np.linspace([-torque_limit] * dof, [torque_limit] * dof, discretization)

    # Construct environment
    env = None
    if gui:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3')

    # Construct an agent
    agent = Agent([num_observation, 40, 40, discretization**dof], batch_size=5000, lr=1e-2)
    agent.set_gamma(0.9)

    decay_rate = 0.999

    # Used to store cost over episodes
    costs = []

    try:
        # Run training episodes
        num_episodes = 5000
        for episode in range(num_episodes):
            old_state = env.reset()
            old_state = old_state[0]

            score = 0

            done = False
            while not done:
                # Make discrete action from model
                discrete_action = agent.get_action(old_state)
                action = get_action_from_discrete(discrete_action, discrete_action_space)

                new_state, reward, term, trunc, _ = env.step(action=action)
                score = reward

                done = term or trunc

                # Train with and record in replay buffer
                # agent.train_short_memory(old_state, discrete_action, reward, new_state, done)
                agent.remember(old_state, discrete_action, reward, new_state, term)

            agent.epsilon_decay(decay_rate)

            # Update weights
            cost = agent.train_long_memory()
            costs.append(cost)

            print(f'Game {episode}, Score: {score}, Cost: {cost}')
        
    except KeyboardInterrupt:
        pass

    print('Saving model to file...')
    agent.save(filename)

    # Generate graph of model metrics
    plt.plot(costs)
    plt.show()

def evaluate(num_observation, torque_limit, discretization, dof, gui=False, filename='./model.pth'):
    discrete_action_space = np.linspace([-torque_limit] * dof, [torque_limit] * dof, discretization)

    # Construct environment
    if gui:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3')

    # Construct an agent
    agent = Agent([num_observation, 1024, discretization**dof])

    agent.load(filename)
    agent.epsilon_decay(0.0)

    # Used to store cost over episodes
    scores = []

    try:
        # Run training episodes
        num_episodes = 100
        for episode in range(num_episodes):
            old_state = env.reset()
            old_state = old_state[0]

            score = 0

            done = False
            while not done:
                # Make discrete action from model
                discrete_action = agent.get_action(old_state)
                action = get_action_from_discrete(discrete_action, discrete_action_space)

                _, reward, term, trunc, _ = env.step(action=action)
                score = reward

                done = term or trunc

            scores.append(score)
            print(f'Game {episode}, Score: {score}')
        
    except KeyboardInterrupt:
        pass

    # Generate graph of model metrics
    plt.plot(scores)
    plt.show()


def main():

    num_observation = 24
    torque_limit = 0.2
    discretization = 3
    dof = 4

    # Parse CLI arguments
    filename, eval, with_gui = parse_arguments()
    

    if eval:
        evaluate(num_observation, torque_limit, discretization, dof, with_gui, filename=filename)
    else:
        train(num_observation, torque_limit, discretization, dof, with_gui, filename=filename)

if __name__ == '__main__':
    main()