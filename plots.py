from typing import Any, Callable, Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


class Plots:
    def plot_scores(self, scores: Sequence[float]) -> None:
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
