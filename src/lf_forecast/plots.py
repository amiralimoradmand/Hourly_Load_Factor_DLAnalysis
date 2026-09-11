from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.train_loss, label="train")
    ax.plot(history.val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(
    target_start_datetime: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    horizon: int,
    out_path: str | Path,
    max_days: int | None = None,
) -> None:
    """Stitch consecutive day-ahead forecast blocks into one continuous line
    per model and compare against the actual continuous series."""
    order = np.argsort(target_start_datetime)
    y_true = y_true[order]
    predictions = {name: arr[order] for name, arr in predictions.items()}

    if max_days is not None:
        y_true = y_true[:max_days]
        predictions = {name: arr[:max_days] for name, arr in predictions.items()}

    actual_flat = y_true.reshape(-1)
    x = np.arange(len(actual_flat))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, actual_flat, label="actual", color="black", linewidth=1.5)
    for name, arr in predictions.items():
        ax.plot(x, arr.reshape(-1), label=name, linewidth=1.0, alpha=0.85)

    for day_boundary in range(0, len(actual_flat), horizon):
        ax.axvline(day_boundary, color="grey", linewidth=0.3, alpha=0.4)

    ax.set_xlabel("test-set hour (stitched day-ahead forecasts)")
    ax.set_ylabel("load factor")
    ax.set_title("Predicted vs. actual load factor on the test set")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
