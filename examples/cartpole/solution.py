import gym
import math
import random

import torch

import sys
sys.path.append('../../..')

from machine_learning.dqn import DQN
from machine_learning.agent import Agent

env = gym.make('CartPole-v1')
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, _ = env.reset()
n_observations = len(state)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

agent = Agent(n_observations, n_actions, GAMMA, BATCH_SIZE, MEMORY_SIZE, LR, device=device)

def evaluate(num_episodes):
    agent.policy_net.load('model/model.pth')
    steps_done = 0

    for _ in range(num_episodes):
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = agent.select_action(state, steps_done, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                break


    print('Complete')

def train(num_episodes):
    steps_done = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        score = 0
        while not done:
            action = agent.select_action(state, steps_done, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            score += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push((state, action, next_state, reward))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                print(f'Game {i_episode} Score: {score + 1}')
                break


    agent.policy_net.save('model/model.pth')
    print('Complete')

train(300)