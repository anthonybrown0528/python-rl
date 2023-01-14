import sys
import gym
import argparse

import torch
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

def train(env, num_observation, discretization, dof, gui=False, filename='./model.pth'):

    # Construct an agent
    agent = Agent([num_observation, 128, 128, 128, discretization**dof], batch_size=128, memory_size=10000, lr=1e-4)
    agent.set_gamma(0.99)

    agent.set_epsilon(1.0)
    decay_rate = 0.99

    # Used to store cost over episodes
    costs = []

    try:
        # Run training episodes
        num_episodes = 1000
        for episode in range(num_episodes):
            old_state, _ = env.reset()

            score = 0

            done = False
            while not done:
                # Make discrete action from model
                action = agent.get_action(old_state)
                action_idx = np.argmax(action)

                new_state, reward, term, trunc, _ = env.step(action=action_idx)
                score += reward

                pole_angle = abs(new_state[2])
                reward += 1 / (pole_angle + 0.01)

                reward = reward - 5 if done else reward

                # print(f'Pole angle: {pole_angle} Reward: {reward}')

                done = term or trunc

                # Train with and record in replay buffer
                agent.train_short_memory(old_state, action, score, new_state, term)
                agent.remember(old_state, action, reward, new_state, term)

            agent.epsilon_decay(decay_rate)

            # Update weights
            cost = 0
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

def evaluate(env, num_observation, discretization, dof, gui=False, filename='./model.pth'):

    # Construct an agent
    agent = Agent([num_observation, 1024, discretization**dof])

    agent.load(filename)

    agent.set_min_epsilon(0.0)
    agent.epsilon_decay(0.0)

    # Used to store cost over episodes
    scores = []

    try:
        # Run training episodes
        num_episodes = 1000
        for episode in range(num_episodes):
            old_state, _ = env.reset()

            score = 0

            done = False
            while not done:
                # Make discrete action from model
                action = agent.get_action(old_state)
                action_idx = np.argmax(action)

                _, reward, term, trunc, _ = env.step(action=action_idx)
                score += reward

                done = term or trunc

            scores.append(score)
            print(f'Game {episode}, Score: {score}')
        
    except KeyboardInterrupt:
        pass

    # Generate graph of model metrics
    plt.plot(scores)
    plt.show()


def main():

    num_observation = 4
    discretization = 2
    dof = 1

    # Parse CLI arguments
    filename, eval, with_gui = parse_arguments()
    

    if eval:
        env = gym.make('CartPole-v1', render_mode='human') if with_gui else gym.make('CartPole-v1')
        evaluate(env, num_observation, discretization, dof, with_gui, filename=filename)
    else:
        env = gym.make('CartPole-v1', render_mode='human') if with_gui else gym.make('CartPole-v1')
        train(env, num_observation, discretization, dof, with_gui, filename=filename)

if __name__ == '__main__':
    main()