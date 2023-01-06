import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os


class DQN:

    def __init__(self, layers, lr, gamma):
        super().__init__()

        # Initialize an empty list of layers for the neural network
        self.model = nn.Sequential()
        # Add input layer and hidden layers to the neural network
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1])

            self.model.append(linear)
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(layers[-2], layers[-1]))

        self.input_vars = layers[0]
        self.output_vars = layers[-1]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

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
        Q_this = self.model.forward(state)
        Q_next = self.model.forward(next_state)

        # Copy of predictions to update with more accuate Q-values
        # Used to compare with actual prediction
        target = Q_this.clone()

        # Extract indices of actions
        action_idc = torch.argmax(action, dim=1)
        action_idc = torch.unsqueeze(action_idc, -1)

        # Update estimates of Q-values
        final_Q = reward
        mid_Q = reward + self.gamma * torch.max(Q_next, dim=1).values

        # Include predicted discounted reward if not done
        Q_new = torch.where(done, final_Q, mid_Q)
        Q_new = torch.unsqueeze(Q_new, -1)

        # Update target with better Q-value estimates
        target = torch.scatter(input=target, dim=1, index=action_idc, src=Q_new)

        # Reset tensor gradients to improve runtime performance
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.criterion(Q_this, target)
        loss.backward()

        self.optimizer.step()
        return float(loss)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        final_move = [0] * self.output_vars
        if random.random() < self.epsilon:
            move = random.randint(0, self.output_vars - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return np.array(final_move, dtype=np.float32)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model, file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if not os.path.exists(file_name):
            raise FileNotFoundError()

        self.model = torch.load(file_name)
