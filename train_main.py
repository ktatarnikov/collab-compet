from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
from agent_config import AgentConfig
from plots import Plots
from train_runner import TrainRunner

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_config = AgentConfig(device=device,
                               buffer_size=int(1e6),
                               batch_size=64,
                               gamma=0.99,
                               tau=1e-3,
                               lr_actor=5e-4,
                               lr_critic=5e-4,
                               eps=1.0,
                               update_every=20,
                               agent_count=2,
                               state_size=24,
                               action_size=2,
                               random_seed=42)
    trainer = TrainRunner(config=agent_config,
                          env_path="./Tennis_Linux/Tennis.x86_64",
                          checkpoint_path="./checkpoints")
    scores = trainer.run()
    trainer.close()
    Plots().plot_scores(scores)
