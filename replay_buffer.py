import random
from collections import deque, namedtuple
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size: int, buffer_size: int, batch_size: int,
                 device: str):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state: Any, action: Any, reward: Any, next_state: Any,
            done: Any) -> None:
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences
                       if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences
                       if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
