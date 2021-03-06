from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

import torch
from agent_config import AgentConfig
from multi_agent import MultiAgent
from unityagents import UnityEnvironment

torch.set_num_threads(1)


class TrainRunner:
    def __init__(self, config: AgentConfig, env_path: str,
                 checkpoint_path: str):
        self.env = UnityEnvironment(file_name=env_path)
        self.config = config
        self.brain_name = self.env.brain_names[0]
        self.agent = MultiAgent(config)
        self.checkpoint_path = checkpoint_path

    def run(self, n_episodes: int = 1700, max_t: int = 500) -> Sequence[float]:
        """
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
        """

        scores_deque = deque(maxlen=100)
        score_average = 0
        scores = []
        scores_stddev = []
        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            score = np.zeros(self.config.agent_count)
            self.agent.reset()
            for t in range(max_t):
                actions = self.agent.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations  # get next state (for each agent)
                rewards = env_info.rewards  # get reward (for each agent)
                dones = env_info.local_done  # see if episode finished
                score += rewards
                self.agent.step(states, actions, rewards, next_states, dones)
                states = next_states  # roll over states to next time step
                if np.any(dones):  # exit loop if episode finished
                    break

            scores.append(np.max(score))
            scores_deque.append(np.max(score))
            score_average = np.mean(scores_deque)
            score_stddev = np.std(scores_deque)
            scores_stddev.append(score_stddev)

            if i_episode % 50 == 0:
                self.agent.save(self.checkpoint_path)
                print(
                    f'\rEpisode {i_episode}\tAverage Score: {score_average:.2f}'
                )
            if score_average > 1.0:
                break

        return scores

    def close(self) -> None:
        self.env.close()
