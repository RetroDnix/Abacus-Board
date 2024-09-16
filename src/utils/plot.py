import matplotlib.pyplot as plt
from typing import Any, Dict, List
import math

def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def gen_loss_plot(trainer_log: List[Dict[str, Any]]):
    r"""
    Plots loss curves in LlamaBoard.
    """
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig