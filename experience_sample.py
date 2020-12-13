from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch


class AgentExperienceSample:
    def __init__(self, device: str):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.device = device

    def add_agent_experience(self, states, actions, rewards, next_states,
                             dones) -> None:
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)

    def get_states(self) -> np.ndarray:
        return torch.stack(self.states).to(self.device)

    def get_next_states(self) -> np.ndarray:
        return torch.stack(self.next_states).to(self.device)

    def get_rewards(self, agent_idx: int) -> np.ndarray:
        return self.rewards[agent_idx].to(self.device)

    def get_dones(self, agent_idx: int) -> np.ndarray:
        return self.dones[agent_idx].to(self.device)

    def get_full_states(self) -> torch.Tensor:
        return torch.stack([
            torch.cat(state) for state in zip(*(s for s in self.states))
        ]).to(self.device)

    def get_full_next_states(self) -> torch.Tensor:
        return torch.stack([
            torch.cat(next_state)
            for next_state in zip(*(ns for ns in self.next_states))
        ]).to(self.device)

    def get_full_actions(self) -> torch.Tensor:
        return torch.stack([
            torch.cat(action) for action in zip(*(a for a in self.actions))
        ]).to(self.device)
