"""Feature engineering and windowing for the sequence models.

Each hourly timestep is represented by:
    [load_factor, hour_sin, hour_cos, dow_sin, dow_cos]

Sliding windows of length `window` are used to predict the next `horizon`
load_factor values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["load_factor", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def add_cyclical_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    dow = pd.to_datetime(df["date"]).dt.dayofweek  # Monday=0
    df["day_of_week"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


@dataclass
class Windows:
    X: np.ndarray  # (n_windows, window, n_features)
    y: np.ndarray  # (n_windows, horizon)
    target_start_datetime: np.ndarray  # (n_windows,) first timestamp being predicted


def make_windows(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    stride: int = 1,
) -> Windows:
    """Build sliding (input, target) windows over the full continuous series.

    Window i uses rows [i, i+window) as input and predicts load_factor for
    rows [i+window, i+window+horizon).
    """
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = df["load_factor"].to_numpy(dtype=np.float32)
    datetimes = df["datetime"].to_numpy()

    n = len(df)
    last_start = n - window - horizon
    starts = np.arange(0, last_start + 1, stride)

    X = np.stack([values[s : s + window] for s in starts])
    y = np.stack([targets[s + window : s + window + horizon] for s in starts])
    target_start = datetimes[starts + window]

    return Windows(X=X, y=y, target_start_datetime=target_start)


def time_split_dates(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[np.datetime64, np.datetime64]:
    """Return (val_start, test_start) datetimes that split the unique dates
    chronologically into contiguous train/val/test blocks."""
    unique_dates = np.sort(pd.to_datetime(df["date"]).unique())
    n = len(unique_dates)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))
    val_start = unique_dates[train_end_idx]
    test_start = unique_dates[val_end_idx]
    return val_start, test_start


def split_windows(
    windows: Windows, val_start: np.datetime64, test_start: np.datetime64
) -> dict[str, Windows]:
    ts = windows.target_start_datetime
    train_mask = ts < val_start
    val_mask = (ts >= val_start) & (ts < test_start)
    test_mask = ts >= test_start

    def subset(mask: np.ndarray) -> Windows:
        return Windows(
            X=windows.X[mask], y=windows.y[mask], target_start_datetime=ts[mask]
        )

    return {"train": subset(train_mask), "val": subset(val_mask), "test": subset(test_mask)}


def prepare_splits(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, Windows]:
    """End-to-end: cyclical features -> dense sliding windows -> chronological
    split -> non-overlapping subsample for val/test so reported metrics
    reflect independent day-ahead forecasts rather than autocorrelated
    overlapping ones. Train stays dense (stride=1) to maximize training
    signal for the RNNs."""
    feat_df = add_cyclical_features(df)
    dense = make_windows(feat_df, window=window, horizon=horizon, stride=1)
    val_start, test_start = time_split_dates(feat_df, train_frac, val_frac)
    splits = split_windows(dense, val_start, test_start)

    splits["val"] = select_non_overlapping(splits["val"], stride=horizon)
    splits["test"] = select_non_overlapping(splits["test"], stride=horizon)
    return splits


def select_non_overlapping(windows: Windows, stride: int) -> Windows:
    """Subsample windows so the prediction horizons no longer overlap.

    Dense (stride=1) windows are good for training (more samples), but bad
    for a held-out comparison: adjacent windows share almost all of their
    target hours, so their errors are highly autocorrelated and a classical
    model would need one fit per window. Evaluating every `stride`-th
    window instead gives one independent day-ahead forecast per block and
    keeps classical-model fitting tractable.
    """
    order = np.argsort(windows.target_start_datetime)
    idx = order[::stride]
    return Windows(
        X=windows.X[idx],
        y=windows.y[idx],
        target_start_datetime=windows.target_start_datetime[idx],
    )
