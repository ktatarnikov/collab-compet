from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
from agent_config import AgentConfig
from ddpg_agent import DDPGAgent
from experience_sample import AgentExperienceSample
from noise import OUNoise


class MultiAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.device = config.device

        self.agents = [
            DDPGAgent(config=config) for _ in range(config.agent_count)
        ]
        # Noise process
        self.noise = OUNoise(config.action_size, config.random_seed)
        self.t_step = 0
        self.eps = config.eps
        self.eps_decay = 0.9995
        self.eps_end = 0.01

    def step(self, states, actions, rewards, next_states, dones):
        for agent_index in range(len(self.agents)):
            agent = self.agents[agent_index]
            agent.step(states, actions, rewards, next_states, dones)

        self.t_step += 1
        if self.t_step > self.config.batch_size and len(
                self.agents[0].memory) > self.config.batch_size:
            if self.t_step % self.config.update_every == 0:
                for _ in range(10):
                    experience = self.sample_experience()
                    self.learn(experience)

    def act(self,
            states: np.ndarray,
            add_noise: bool = True) -> Sequence[torch.Tensor]:
        state = torch.from_numpy(states).float().to(self.config.device)
        actions = []
        for idx in range(len(self.agents)):
            agent = self.agents[idx]
            noise = self.noise if add_noise else None
            agent_action = agent.act_local(state[idx], noise, self.eps)
            actions.append(agent_action.numpy())
        self.eps = max(self.eps_decay * self.eps, self.eps_end)
        return np.stack(actions, axis=0)

    def sample_experience(self) -> AgentExperienceSample:
        experience = AgentExperienceSample(self.device)
        for agent_index in range(len(self.agents)):
            agent = self.agents[agent_index]
            states, actions, rewards, next_states, dones = agent.memory.sample(
            )
            experience.add_agent_experience(states, actions, rewards,
                                            next_states, dones)

        return experience

    def learn(self, experience_sample: AgentExperienceSample) -> None:
        actions_next = self.get_target_actions(
            experience_sample.get_next_states())

        for agent_idx in range(len(self.agents)):
            agent = self.agents[agent_idx]
            agent.learn_critic(experience_sample, self.config.gamma, agent_idx,
                               actions_next)

        for agent_idx in range(len(self.agents)):
            agent = self.agents[agent_idx]
            actions_local = self.get_local_actions(
                experience_sample.get_states())
            agent.learn_actor(experience_sample.get_full_states(),
                              actions_local)
            agent.soft_updates()

    def get_target_actions(self, states: torch.Tensor) -> torch.Tensor:
        actions = []
        for agent_idx in range(len(self.agents)):
            agent = self.agents[agent_idx]
            state = states[agent_idx]
            agent_actions = agent.actor_target(state)
            actions.append(agent_actions)

        return torch.cat(actions, dim=-1)

    def get_local_actions(self,
                          states: Sequence[torch.Tensor]) -> torch.Tensor:
        actions = []
        for agent_idx in range(len(self.agents)):
            agent = self.agents[agent_idx]
            state = states[agent_idx]
            agent_actions = agent.actor_local(state)
            actions.append(agent_actions)

        return torch.cat(actions, dim=-1)

    def reset(self):
        self.noise.reset()

    def save(self, checkpoint_path: str) -> None:
        for agent_index in range(len(self.agents)):
            agent = self.agents[agent_index]
            agent.save(checkpoint_path, agent_index)

    def load(self, checkpoint_path: str) -> None:
        for agent_index in range(len(self.agents)):
            agent = self.agents[agent_index]
            agent.load(checkpoint_path, agent_index)
