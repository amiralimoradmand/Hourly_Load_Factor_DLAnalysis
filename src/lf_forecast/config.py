"""Central configuration for the load-factor forecasting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    raw_csv: str = "data/raw/Tetuan City power consumption.csv"
    processed_csv: str = "data/processed/load_factor_hourly.csv"
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implied as 1 - train_frac - val_frac
    window: int = 168  # hours of history fed to the model (7 days)
    horizon: int = 24  # hours to forecast ahead (1 day)


@dataclass
class ModelConfig:
    model_type: str = "lstm"  # "lstm" or "gru"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def resolve(path: str) -> Path:
    """Resolve a project-relative path to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
