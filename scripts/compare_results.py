"""Aggregate baseline + LSTM + GRU metrics into one comparison table and
build the stitched predicted-vs-actual plot on the test set.

Usage:
    py -3 scripts/compare_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.config import Config, resolve
from lf_forecast.metrics import all_metrics
from lf_forecast.plots import plot_predicted_vs_actual


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    cfg = Config()
    horizon = cfg.data.horizon

    baselines_metrics = load_json(resolve("outputs/metrics/baselines.json"))
    baselines_preds = np.load(resolve("outputs/predictions/baselines_test_preds.npz"))

    rows = [
        ("Seasonal-naive", baselines_metrics["seasonal_naive"]),
        ("Holt-Winters (ETS)", baselines_metrics["holt_winters"]),
    ]

    predictions_for_plot = {
        "seasonal_naive": baselines_preds["seasonal_naive"],
        "holt_winters": baselines_preds["holt_winters"],
    }
    y_true = baselines_preds["y_true"]
    target_start = baselines_preds["target_start_datetime"]

    for name, label in [("lstm", "LSTM"), ("gru", "GRU")]:
        metrics_path = resolve(f"outputs/metrics/{name}.json")
        pred_path = resolve(f"outputs/predictions/{name}_test_preds.npz")
        if not metrics_path.exists() or not pred_path.exists():
            print(f"skipping {label}: run `py -3 scripts/train_model.py --model {name}` first")
            continue
        m = load_json(metrics_path)
        rows.append((label, m["metrics"]))
        preds = np.load(pred_path)
        predictions_for_plot[name] = preds["y_pred"]

    naive_rmse = baselines_metrics["seasonal_naive"]["rmse"]

    header = f"| {'Model':<20} | {'RMSE':>8} | {'MAE':>8} | {'MAPE (%)':>9} | {'vs. seasonal-naive':>19} |"
    sep = f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*21}|"
    lines = [header, sep]
    for label, m in rows:
        improvement = 100 * (naive_rmse - m["rmse"]) / naive_rmse
        lines.append(
            f"| {label:<20} | {m['rmse']:>8.4f} | {m['mae']:>8.4f} | {m['mape']:>9.2f} | "
            f"{improvement:>+18.1f}% |"
        )
    table = "\n".join(lines)
    print(table)

    out_md = resolve("outputs/metrics/comparison.md")
    out_md.write_text(table + "\n")
    print(f"\nSaved comparison table to {out_md}")

    plot_predicted_vs_actual(
        target_start_datetime=target_start,
        y_true=y_true,
        predictions=predictions_for_plot,
        horizon=horizon,
        out_path=resolve("outputs/figures/predicted_vs_actual.png"),
        max_days=14,
    )
    print("Saved predicted-vs-actual plot to outputs/figures/predicted_vs_actual.png")


if __name__ == "__main__":
    main()
