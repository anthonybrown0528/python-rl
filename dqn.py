import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os

from machine_learning.model import Model


class DQN(nn.Module):

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.model = nn.Sequential()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.model.append(self.layer1)
        self.model.append(nn.ReLU())

        self.model.append(self.layer2)
        self.model.append(nn.ReLU())

        self.model.append(self.layer3)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model.forward(x)

    def save(self, file_name='model.pth'):
        torch.save(self.model, file_name)

    def load(self, file_name='model.pth'):
        self.model = torch.load(file_name)
