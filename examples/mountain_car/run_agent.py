import sys
import gym
import argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../..')
from machine_learning.agent import Agent

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains/evaluates a learning model playing the Snake game')

    parser.add_argument('-f', '--filename', default='model.pth', help='An optional argument for specifying file name of the model to store or load')
    parser.add_argument('-e', '--eval', action='store_true', help='An optional argument for specifying the program to run in evaluation mode instead of training mode')
    parser.add_argument('-g', '--gui', action='store_true', help='An optional argument for specifying the program to run with a GUI')

    args = parser.parse_args()

    return args.filename, args.eval, args.gui

def train(num_observation, discretization, dof, gui=False, filename='./model.pth'):

    # Construct environment
    env = None
    if gui:
        env = gym.make('MountainCar-v0', render_mode='human')
    else:
        env = gym.make('MountainCar-v0')

    # Construct an agent
    agent = Agent([num_observation, 24, 48, discretization**dof], batch_size=1000)
    agent.set_gamma(0.99)

    agent.set_epsilon(1.0)
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
                action = agent.get_action(old_state)
                action_idx = np.argmax(action)

                new_state, reward, term, trunc, _ = env.step(action=action_idx)
                score = reward

                done = term or trunc

                # Train with and record in replay buffer
                # agent.train_short_memory(old_state, action, reward, new_state, done)
                agent.remember(old_state, action, reward, new_state, term)

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

def evaluate(num_observation, discretization, dof, gui=False, filename='./model.pth'):

    # Construct environment
    if gui:
        env = gym.make('MountainCar-v0', render_mode='human')
    else:
        env = gym.make('MountainCar-v0')

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
                action = agent.get_action(old_state)
                action_idx = np.argmax(action)

                _, reward, term, trunc, _ = env.step(action=action_idx)
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

    num_observation = 2
    discretization = 3
    dof = 1

    # Parse CLI arguments
    filename, eval, with_gui = parse_arguments()
    

    if eval:
        evaluate(num_observation, discretization, dof, with_gui, filename=filename)
    else:
        train(num_observation, discretization, dof, with_gui, filename=filename)

if __name__ == '__main__':
    main()