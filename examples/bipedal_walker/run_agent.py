import sys
import gym

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
        exp = joint + 1
        power = discretization**exp

        action_idx = (action_idx // power) % discretization
        action[joint] = discrete_action_space[action_idx][joint]

    return action

def train(num_observation, discretization, dof, gui=False):
    discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)

    # Construct environment
    env = None
    if gui:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3')

    # Construct an agent
    agent = Agent([num_observation, 1024, discretization**dof])
    agent.network.epsilon = 200

    # Used to store cost over episodes
    costs = []

    try:
        # Run training episodes
        num_episodes = 300
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

                # Record in replay buffer
                agent.remember(old_state, discrete_action, reward, new_state, done)

                # Visualize the environment
                if gui:
                    env.render()

            agent.network.epsilon -= 1

            # Update weights
            cost = agent.train_long_memory()
            costs.append(cost)

            print(f'Game {episode}, Score: {score}, Cost: {cost}')
        
    except KeyboardInterrupt:
        pass

    print('Saving model to file...')
    agent.network.save()

    # Generate graph of model metrics
    plt.plot(costs)
    plt.show()

def evaluate(num_observation, discretization, dof, gui=False):
    discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)

    # Construct environment
    if gui:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3')

    # Construct an agent
    agent = Agent([num_observation, 1024, discretization**dof])

    agent.network.load()
    agent.network.epsilon = 0

    agent.network.model.eval()
    agent.network.model.zero_grad()

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

                new_state, reward, term, trunc, _ = env.step(action=action)
                score = reward

                done = term or trunc

                # Record in replay buffer
                agent.remember(old_state, discrete_action, reward, new_state, done)

                # Visualize the environment
                if gui:
                    env.render()


            # Update weights
            cost = agent.train_long_memory()
            scores.append(score)

            print(f'Game {episode}, Score: {score}, Cost: {cost}')
        
    except KeyboardInterrupt:
        pass

    # Generate graph of model metrics
    plt.plot(scores)
    plt.show()


def main():
    eval = False
    with_gui = False

    num_observation = 24
    discretization = 10
    dof = 4

    if eval:
        evaluate(num_observation, discretization, dof, with_gui)
    else:
        train(num_observation, discretization, dof, with_gui)

if __name__ == '__main__':
    main()