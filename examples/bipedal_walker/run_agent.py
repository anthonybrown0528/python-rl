import sys
import gym

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

def train():
    num_observation = 24
    discretization = 5
    dof = 4

    discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)

    # Construct environment
    env = gym.make('BipedalWalker-v3', render_mode='human')

    # Construct an agent
    agent = Agent([num_observation, 1024, discretization**dof])
    agent.network.epsilon = 50

    # Run training episodes
    num_episodes = 300
    for episode in range(num_episodes):
        old_state = env.reset()
        old_state = old_state[0]

        high_score = 0
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
            env.render()

        if score > high_score:
            high_score = score
            agent.network.save()

        # Update weights
        agent.train_long_memory()
        print(f'Game {episode} Score: {score}')


def main():
    train()

if __name__ == '__main__':
    main()