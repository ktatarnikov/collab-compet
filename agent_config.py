from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch


class AgentConfig:
    def __init__(self, device: torch.device, buffer_size: int, batch_size: int,
                 gamma: float, tau: float, lr_actor: float, lr_critic: float,
                 eps: float, update_every: int, agent_count: int,
                 state_size: int, action_size: int, random_seed: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps = eps
        self.update_every = update_every
        self.agent_count = agent_count
        self.random_seed = random_seed
