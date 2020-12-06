import copy
import random

import numpy as np

import torch


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self,
                 size: int,
                 seed: int,
                 mu: float = 0.,
                 theta: float = 0.15,
                 sigma: float = 0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.size = size

    def reset(self) -> None:
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""

        x = self.state
        dx = self.theta * (self.mu - x)
        dx += self.sigma * np.random.randn(self.size)
        self.state = x + dx

        return torch.Tensor(self.state)
