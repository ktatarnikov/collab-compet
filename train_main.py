from typing import Any, Callable, Dict, Optional, Sequence, Union

from plots import Plots
from train_runner import TrainRunner

if __name__ == '__main__':
    trainer = TrainRunner(2, "./Tennis_Linux/Tennis.x86_64", "./checkpoints")
    scores = trainer.run()
    trainer.close()
    Plots().plot_scores(scores)
