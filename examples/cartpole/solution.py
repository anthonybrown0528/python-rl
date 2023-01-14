import gym
import torch

import sys
sys.path.append('../../..')

from machine_learning.agent import Agent

env = gym.make('CartPole-v1')
state, _ = env.reset()

# Get number of actions and state observations
n_actions = env.action_space.n
n_observations = len(state)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(n_observations, n_actions, device=device)

def play_episode(eval=False):
    # Initialize the environment and get it's state
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    score = 0
    done = False
    while not done:
        action = agent.select_action(state, agent.get_steps_done(), env)
        observation, reward, terminated, truncated, _ = env.step(action.item())

        score += reward

        reward = torch.tensor([reward], device=device)

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if not eval:

            # Store the transition in memory
            agent.memory.push((state, action, next_state, reward, terminated))
            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

        # Move to the next state
        state = next_state

        done = terminated or truncated
        agent.set_steps_done(agent.get_steps_done() + 1)

    return score


def evaluate(num_episodes):
    agent.policy_net.load('model/model.pth')
    
    for i_episode in range(num_episodes):

        score = play_episode(True)
        print('Game:', i_episode, 'Score:', score)


    print('Complete')

def train(num_episodes):
    for i_episode in range(num_episodes):

        score = play_episode()
        print('Game:', i_episode, 'Score:', score)

    agent.policy_net.save('model/model.pth')
    print('Complete')

evaluate(300)