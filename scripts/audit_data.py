"""Data audit: compares the original random-subsample CSV against the
reconstructed continuous hourly series, and reports on missing values,
chronological continuity, and the hour-of-day cycle.

Usage:
    py -3 scripts/audit_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.config import Config, resolve


def audit_subsample(path: Path) -> None:
    print("=" * 70)
    print(f"ORIGINAL SUBSAMPLE: {path}")
    print("=" * 70)
    df = pd.read_csv(path)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    print(f"rows: {len(df)}, unique dates: {df['date_parsed'].nunique()}")
    print(f"missing values: {df.isna().sum().sum()}")
    counts = df.groupby("date_parsed").size()
    print(f"dates with all 24 hours present: {(counts == 24).sum()} / {len(counts)}")
    print(f"mean hours sampled per date: {counts.mean():.2f}")
    df["datetime"] = df["date_parsed"] + pd.to_timedelta(df["hour"], unit="h")
    diffs = df.sort_values("datetime")["datetime"].diff().dropna()
    frac_consecutive = (diffs == pd.Timedelta(hours=1)).mean()
    print(f"fraction of sorted row-pairs exactly 1h apart: {frac_consecutive:.3f}")
    print("-> NOT usable for windowed sequence forecasting as-is "
          "(no date has a complete 24h profile).")
    print()


def audit_processed(path: Path) -> None:
    print("=" * 70)
    print(f"RECONSTRUCTED CONTINUOUS SERIES: {path}")
    print("=" * 70)
    df = pd.read_csv(path, parse_dates=["datetime", "date"])
    print(f"rows: {len(df)}, unique dates: {df['date'].nunique()}")
    print(f"missing values: {df.isna().sum().sum()}")
    diffs = df["datetime"].diff().dropna()
    print(f"all gaps == 1h: {(diffs == pd.Timedelta(hours=1)).all()}")
    print(f"load_factor range: [{df['load_factor'].min():.4f}, {df['load_factor'].max():.4f}]")
    print()
    print("mean load_factor by hour-of-day (confirms the daily cycle):")
    print(df.groupby("hour")["load_factor"].mean().round(3).to_string())
    print()


def main() -> None:
    cfg = Config()
    audit_subsample(resolve("data/reference/hourly_load_factor_subsample.csv"))
    processed_path = resolve(cfg.data.processed_csv)
    if processed_path.exists():
        audit_processed(processed_path)
    else:
        print(f"{processed_path} not found yet - run scripts/prepare_data.py first")


if __name__ == "__main__":
    main()
