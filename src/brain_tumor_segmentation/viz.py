"""Plotting helpers for training curves and volume agreement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr


def plot_history(history, metric: str = "loss") -> None:
    """Plot train vs validation curves for a Keras History metric."""
    plt.figure(figsize=(16, 6))
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Train", "Validation"])
    plt.tight_layout()


def plot_volume_agreement(volumes_gt: np.ndarray, volumes_pred: np.ndarray) -> float:
    """Scatter + regression and Bland–Altman plots; returns Pearson r."""
    volumes_gt = np.asarray(volumes_gt).ravel()
    volumes_pred = np.asarray(volumes_pred).ravel()

    slope, intercept, _, _, _ = linregress(volumes_gt, volumes_pred)
    line = slope * volumes_gt + intercept
    r, _ = pearsonr(volumes_gt, volumes_pred)

    plt.figure()
    plt.scatter(volumes_gt, volumes_pred)
    plt.plot(
        volumes_gt,
        line,
        "r",
        label=f"y={slope:.2f}x+{intercept:.2f}, r={r:.2f}",
    )
    plt.xlabel("Ground-truth volume")
    plt.ylabel("Predicted volume")
    plt.legend(loc="best")
    plt.tight_layout()

    sm.graphics.mean_diff_plot(volumes_pred, volumes_gt)
    return float(r)
