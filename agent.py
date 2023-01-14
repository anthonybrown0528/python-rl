import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
from collections import namedtuple

import numpy as np


from machine_learning.dqn import DQN
from machine_learning.replay_buffer import ReplayBuffer

class Agent:
    
    def __init__(self, n_observations, n_actions, gamma=0.99, batch_size=128, memory_size=10000, lr=1e-4, device="cpu"):
        self._device = device
        self._gamma = gamma # discount rate

        self._batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)

        self.policy_net = DQN(n_observations, n_actions).to(self._device)
        self.target_net = DQN(n_observations, n_actions).to(self._device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.MSELoss()

    def set_gamma(self, gamma):
        self._gamma = gamma

    def get_gamma(self) -> float:
        return self._gamma

    def remember(self, state, action, reward, next_state, done):
        self._memory.add((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def optimize_model(self):
        if len(self.memory) < self._batch_size:
            return

        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        transitions = self.memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state, steps_done, env):
        # global steps_done
        sample = random.random()

        eps_threshold = DQN.EPS_START * math.exp(-steps_done / DQN.EPS_DECAY)
        eps_threshold = max(eps_threshold, DQN.EPS_END)

        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # we pick action with the larger expected reward.
                return torch.argmax(self.policy_net.forward(state)).reshape((1, 1))
        else:
            return torch.tensor(env.action_space.sample(), device=self._device, dtype=torch.long).reshape((1, 1))

    def set_epsilon(self, epsilon):
        self._network.set_epsilon(epsilon)

    def set_min_epsilon(self, min_epsilon):
        self._network.set_min_epsilon(min_epsilon)

    def epsilon_decay(self, decay_rate):
        new_epsilon = self._network.get_epsilon() * decay_rate
        self._network.set_epsilon(new_epsilon)

    def save(self, file_name='model.pth'):
        self._network.save(file_name=file_name)
    
    def load(self, file_name='model.pth'):
        self._network.load(file_name=file_name)