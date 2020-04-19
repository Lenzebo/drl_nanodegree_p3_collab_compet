import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNetwork(nn.Module):
    """
    Basic implementation of a fully connected neural network with 2 hidden layers
    """
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):
    """
    Basic implementation of a policy for PPO having two heads:
     - actor to propose an action (mean with gaussian noise)
     - critic to judge about the value of reaching a state
    """

    def __init__(self, state_size, action_size, hidden_size, device, seed):
        super(PPOPolicyNetwork, self).__init__()
        self.actor = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh)
        self.critic = FullyConnectedNetwork(state_size, 1, hidden_size)
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)
        self.device = device

    def forward(self, obs, action=None):
        if isinstance(obs, (np.ndarray, np.generic) ):
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs = obs.to(self.device)

        a = self.actor(obs)
        v = self.critic(obs)

        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        else:
            action = action.to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v

