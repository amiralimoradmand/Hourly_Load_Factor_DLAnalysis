"""Reconstruct the continuous hourly load-factor series from the raw UCI
Tetouan power-consumption dataset and save it to data/processed/.

Usage:
    py -3 scripts/prepare_data.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.config import Config, resolve
from lf_forecast.data_prep import build_hourly_load_factor, save_processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()

    raw_csv = resolve(cfg.data.raw_csv)
    out_csv = resolve(cfg.data.processed_csv)

    print(f"Reading raw data from {raw_csv}")
    hourly = build_hourly_load_factor(raw_csv)
    print(f"Reconstructed {len(hourly)} continuous hourly rows "
          f"({hourly['date'].nunique()} days x 24h)")
    print(hourly["load_factor"].describe())

    save_processed(hourly, out_csv)
    print(f"Saved processed series to {out_csv}")


if __name__ == "__main__":
    main()
