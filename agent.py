import torch
import numpy as np

from machine_learning.dqn import DQN
from machine_learning.replay_buffer import ReplayBuffer

class Agent:
    
    def __init__(self, layers, batch_size=1000, memory_size=1000000, lr=0.001, eval=False):
        assert batch_size <= memory_size

        self._gamma = 0.90 # discount rate
        self._batch_size = batch_size
        self._memory = ReplayBuffer(maxlen=memory_size)
        self._network = DQN(layers=layers, lr=lr, gamma=self._gamma)

        # Disable exploration to evaluate model
        if eval:
            self._network.epsilon = 0

    def set_gamma(self, gamma):
        self._gamma = gamma

    def get_gamma(self) -> float:
        return self._gamma

    def remember(self, state, action, reward, next_state, done):
        self._memory.add((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        mini_sample = self._memory.get_training_samples(size=self._batch_size)

        # end training prematurely if there are no memory samples
        if len(mini_sample) == 0:
            return 0

        # extract data for training
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # train the model and retrieve the sum of the loss function
        cost = self._network.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
        cost /= len(mini_sample)

        return cost

    def train_short_memory(self, state, action, reward, next_state, done):
        self._network.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        return self._network.get_action(state)

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