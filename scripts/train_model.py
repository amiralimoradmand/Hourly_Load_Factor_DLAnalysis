"""Train an LSTM or GRU day-ahead load-factor forecaster.

Usage:
    py -3 scripts/train_model.py --model lstm
    py -3 scripts/train_model.py --model gru --hidden-size 128 --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.config import Config, resolve
from lf_forecast.datasets import WindowDataset
from lf_forecast.features import FEATURE_COLUMNS, prepare_splits
from lf_forecast.metrics import all_metrics
from lf_forecast.models import build_model
from lf_forecast.plots import plot_loss_curves
from lf_forecast.train import predict, train_model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["lstm", "gru"], default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.model is not None:
        cfg.model.model_type = args.model
    if args.hidden_size is not None:
        cfg.model.hidden_size = args.hidden_size
    if args.num_layers is not None:
        cfg.model.num_layers = args.num_layers
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.patience is not None:
        cfg.train.patience = args.patience
    if args.seed is not None:
        cfg.train.seed = args.seed

    set_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={cfg.model.model_type}  device={device}")

    df = pd.read_csv(resolve(cfg.data.processed_csv), parse_dates=["datetime", "date"])
    splits = prepare_splits(
        df,
        window=cfg.data.window,
        horizon=cfg.data.horizon,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )
    print(
        f"train windows={len(splits['train'].X)}  "
        f"val windows={len(splits['val'].X)}  test windows={len(splits['test'].X)}"
    )

    train_loader = DataLoader(WindowDataset(splits["train"]), batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(splits["val"]), batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(splits["test"]), batch_size=cfg.train.batch_size, shuffle=False)

    model = build_model(
        cfg.model.model_type,
        input_size=len(FEATURE_COLUMNS),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        horizon=cfg.data.horizon,
        dropout=cfg.model.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        device=device,
    )

    y_pred, y_true = predict(model, test_loader, device=device)
    metrics = all_metrics(y_true, y_pred)
    print(f"test metrics ({cfg.model.model_type}):", metrics)

    name = cfg.model.model_type
    ckpt_path = resolve(f"outputs/checkpoints/{name}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    cfg_path = resolve(f"outputs/checkpoints/{name}_config.yaml")
    cfg.save(cfg_path)

    metrics_path = resolve(f"outputs/metrics/{name}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {"metrics": metrics, "n_params": n_params, "epochs_trained": len(history.train_loss)},
            f,
            indent=2,
        )

    plot_loss_curves(
        history,
        title=f"{name.upper()} training curve",
        out_path=resolve(f"outputs/figures/{name}_loss_curve.png"),
    )

    pred_path = resolve(f"outputs/predictions/{name}_test_preds.npz")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        target_start_datetime=splits["test"].target_start_datetime,
    )

    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss curve to outputs/figures/{name}_loss_curve.png")
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
