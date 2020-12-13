from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from actor_network import ActorNetwork
from agent_config import AgentConfig
from critic_network import CriticNetwork
from experience_sample import AgentExperienceSample
from noise import OUNoise
from replay_buffer import ReplayBuffer
from torch import nn


class DDPGAgent:
    def __init__(self, config: AgentConfig):
        self.device = config.device
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNetwork(config.state_size, config.action_size,
                                        config.random_seed).to(self.device)
        self.actor_target = ActorNetwork(config.state_size, config.action_size,
                                         config.random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = CriticNetwork(
            config.agent_count * config.state_size, config.action_size,
            config.agent_count, config.random_seed).to(self.device)
        self.critic_target = CriticNetwork(
            config.agent_count * config.state_size, config.action_size,
            config.agent_count, config.random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=config.lr_critic)

        # Replay memory
        self.memory = ReplayBuffer(config.action_size, config.buffer_size,
                                   config.batch_size, self.device)

    def step(self, states, actions, rewards, next_states, dones) -> None:
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(
                states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

    def act_local(self, states, noise: Optional[OUNoise],
                  eps: float) -> torch.Tensor:
        """Returns actions for given state as per current policy."""
        states = states.to(self.device)

        if len(np.shape(states)) == 1:
            states = states.reshape(1, -1)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states)
        self.actor_local.train()
        action = action.to("cpu")
        if noise:
            action = action + eps * noise.sample()
            action = np.clip(action, -1.0, 1.0)

        return action.squeeze()

    def reset(self):
        self.noise.reset()

    def learn_critic(self, experience_sample: AgentExperienceSample,
                     gamma: float, agent_index: int,
                     actions_next_full: torch.Tensor):

        next_states_full = experience_sample.get_full_next_states()
        actions_full = experience_sample.get_full_actions()
        states_full = experience_sample.get_full_states()
        # ---------------------------- update critic ---------------------------- #
        Q_targets_next = self.critic_target(next_states_full,
                                            actions_next_full)

        current_agent_rewards = experience_sample.get_rewards(agent_index)
        current_agent_dones = experience_sample.get_dones(agent_index)

        Q_targets = current_agent_rewards + gamma * Q_targets_next * (
            1 - current_agent_dones)

        Q_expected = self.critic_local(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

    def learn_actor(self, states_full: torch.Tensor,
                    actions_local: torch.Tensor):
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_local(states_full, actions_local)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
        self.actor_optimizer.step()

    def soft_updates(self) -> None:
        self.soft_update(self.critic_local, self.critic_target,
                         self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module,
                    tau: float):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)

    def save(self, checkpoint_path: str, agent_idx: int) -> None:
        torch.save(self.actor_local.state_dict(),
                   f'{checkpoint_path}/agent_{agent_idx}_actor.pth')
        torch.save(self.critic_local.state_dict(),
                   f'{checkpoint_path}/agent_{agent_idx}_critic.pth')

    def load(self, checkpoint_path: str, agent_idx: int) -> None:
        self.actor_local.load_state_dict(
            torch.load(f'{checkpoint_path}/agent_{agent_idx}_actor.pth'))
        self.critic_local.load_state_dict(
            torch.load(f'{checkpoint_path}/agent_{agent_idx}_critic.pth'))
