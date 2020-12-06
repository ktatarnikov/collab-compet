from collections import deque

import numpy as np

import torch
from agent_config import AgentConfig
from multi_agent import MultiAgent
from unityagents import UnityEnvironment

torch.set_num_threads(1)


class TestRunner:
    def __init__(self, config: AgentConfig, int, env_path: str,
                 checkpoint_path: str):
        self.config = config
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = MultiAgent(config)
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        self.agent.load(self.checkpoint_path)

        scores_deque = deque(maxlen=100)
        score_average = 0

        for i_episode in range(10):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations  # get the current state
            self.agent.reset()

            score = np.zeros(self.config.agent_count)
            for j in range(1000):
                action = self.agent.act(state, add_noise=False)

                # send the action to the environment
                env_info = self.env.step(action)[self.brain_name]
                state = env_info.vector_observations  # get the next state
                reward = env_info.rewards  # get the reward
                done = env_info.local_done  # see if episode has finished
                score += reward

                if np.any(done):
                    break

            scores_deque.append(np.max(score))
            score_average = np.mean(scores_deque)
            score_stddev = np.std(scores_deque)
            print(
                f'Episode {i_episode} - Average Score: {score_average:.2f}, StdDev: {score_stddev}'
            )

        self.env.close()
