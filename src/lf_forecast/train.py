from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, device: str = "cpu") -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(X)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
    return total_loss / n_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: str = "cpu",
) -> History:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = History()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}"
            f"{'  *' if improved else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)
