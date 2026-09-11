"""Evaluate the non-deep-learning baselines (seasonal-naive, Holt-Winters
exponential smoothing) on the held-out test windows and save metrics.

Usage:
    py -3 scripts/run_baselines.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.baselines import holt_winters_predict, seasonal_naive_predict
from lf_forecast.config import Config, resolve
from lf_forecast.features import prepare_splits
from lf_forecast.metrics import all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()

    df = pd.read_csv(resolve(cfg.data.processed_csv), parse_dates=["datetime", "date"])
    splits = prepare_splits(
        df,
        window=cfg.data.window,
        horizon=cfg.data.horizon,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )
    test = splits["test"]
    print(f"Evaluating baselines on {len(test.X)} non-overlapping test windows "
          f"({cfg.data.horizon}h-ahead each)")

    results = {}

    naive_pred = seasonal_naive_predict(test, cfg.data.horizon)
    results["seasonal_naive"] = all_metrics(test.y, naive_pred)
    print("seasonal_naive:", results["seasonal_naive"])

    print("Fitting Holt-Winters per test window (this takes a minute)...")
    hw_pred = holt_winters_predict(test, cfg.data.horizon)
    results["holt_winters"] = all_metrics(test.y, hw_pred)
    print("holt_winters:", results["holt_winters"])

    out_path = resolve("outputs/metrics/baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline metrics to {out_path}")

    pred_path = resolve("outputs/predictions/baselines_test_preds.npz")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_path,
        y_true=test.y,
        seasonal_naive=naive_pred,
        holt_winters=hw_pred,
        target_start_datetime=test.target_start_datetime,
    )
    print(f"Saved baseline predictions to {pred_path}")


if __name__ == "__main__":
    main()
