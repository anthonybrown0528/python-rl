import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os

from machine_learning.model import Model


class DQN(Model):

    def __init__(self, layers, lr, gamma):
        super().__init__()

        # Initialize an empty list of layers for the neural network
        self._model = nn.Sequential()

        # Making the code device-agnostic
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(device)

        # build the learning model with the described layers
        self._build_model(layers=layers)

        self._input_vars = layers[0]
        self._output_vars = layers[-1]
        self._lr = lr
        self._gamma = gamma
        self._epsilon = 0.99
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._criterion = nn.MSELoss()

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
    
    def get_epsilon(self) -> float:
        return self._epsilon

    def _build_model(self, layers):
        # Add input layer and hidden layers to the neural network
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1])

            self._model.append(linear)
            self._model.append(nn.ReLU())
        self._model.append(nn.Linear(layers[-2], layers[-1]))

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # 1: predicted Q values with the current and next state
        Q_this = self._model.forward(state)
        Q_next = self._model.forward(next_state)

        # Copy of predictions to update with more accuate Q-values
        # Used to compare with actual prediction
        target = Q_this.clone()

        # Extract indices of actions
        action_idc = torch.argmax(action, dim=1)
        action_idc = torch.unsqueeze(action_idc, -1)

        # Update estimates of Q-values
        final_Q = reward
        mid_Q = reward + self._gamma * torch.max(Q_next, dim=1).values

        # Include predicted discounted reward if not done
        Q_new = torch.where(done, final_Q, mid_Q)
        Q_new = torch.unsqueeze(Q_new, -1)

        # Update target with better Q-value estimates
        target = torch.scatter(input=target, dim=1, index=action_idc, src=Q_new)

        # Reset tensor gradients to improve runtime performance
        self._optimizer.zero_grad()
        
        # Compute loss
        loss = self._criterion(Q_this, target)
        loss.backward()

        self._optimizer.step()
        return float(loss)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        final_move = [0] * self._output_vars
        if random.random() < self._epsilon:
            move = random.randint(0, self._output_vars - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self._model.forward(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return np.array(final_move, dtype=np.float32)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self._model, file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if not os.path.exists(file_name):
            raise FileNotFoundError()

        self._model = torch.load(file_name)
