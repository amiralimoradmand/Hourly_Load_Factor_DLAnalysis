from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .features import Windows


class WindowDataset(Dataset):
    def __init__(self, windows: Windows):
        self.X = torch.from_numpy(windows.X)
        self.y = torch.from_numpy(windows.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
