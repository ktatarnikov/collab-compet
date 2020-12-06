from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self,
                 state_size: Tuple,
                 action_size: int,
                 seed: int,
                 hidden1: int = 512,
                 hidden2: int = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.linear1 = nn.Linear(state_size, hidden1)
        self.batch1 = nn.BatchNorm1d(hidden1)

        self.linear2 = nn.Linear(in_features=hidden1, out_features=hidden2)
        self.batch2 = nn.BatchNorm1d(hidden2)

        self.linear3 = nn.Linear(in_features=hidden2, out_features=action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*self.hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*self.hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-1e-3, 1e-3)
        self.linear1.bias.data.fill_(0.5)
        self.linear2.bias.data.fill_(0.5)
        self.linear3.bias.data.fill_(0.5)

    def hidden_init(self, layer: nn.Module):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        x = state
        x = F.relu(self.batch1(self.linear1(x)))
        x = F.relu(self.batch2(self.linear2(x)))
        x = torch.tanh(self.linear3(x))
        return x
