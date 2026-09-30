from typing import Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_prediction_truth_error(
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis'
) -> Figure:
    # Convert to numpy
    prediction_np = prediction.detach().cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    ground_truth_np = ground_truth.detach().cpu().numpy() if isinstance(ground_truth, torch.Tensor) else ground_truth
    difference_np = prediction_np - ground_truth_np

    # Create compact figure layout
    fig = plt.figure(figsize=(8, 3.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.02, wspace=0.05)

    ax_pred = fig.add_subplot(grid[0, 0])
    ax_truth = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[0, 2])
    cax_shared = fig.add_subplot(grid[1, 0:2])
    cax_err = fig.add_subplot(grid[1, 2])

    if title is not None:
        fig.suptitle(title, fontsize=10, y=0.98)

    # Prediction
    im_pred = ax_pred.imshow(prediction_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_pred.set_title("Prediction", fontsize=8)
    ax_pred.axis('off')

    # Ground truth
    im_truth = ax_truth.imshow(ground_truth_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_truth.set_title("Ground Truth", fontsize=8)
    ax_truth.axis('off')

    # Difference
    im_err = ax_err.imshow(difference_np, cmap=cmap)
    ax_err.set_title("Difference", fontsize=8)
    ax_err.axis('off')

    # Colorbar for prediction and ground truth (shared)
    cbar_shared = fig.colorbar(im_pred, cax=cax_shared, orientation='horizontal', pad=0.01)
    cbar_shared.ax.tick_params(labelsize=6, pad=1)

    # Colorbar for difference
    cbar_err = fig.colorbar(im_err, cax=cax_err, orientation='horizontal', pad=0.01)
    cbar_err.ax.tick_params(labelsize=6, pad=1)

    return fig
