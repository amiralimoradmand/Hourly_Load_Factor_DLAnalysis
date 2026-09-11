"""Build a continuous hourly load-factor series from the raw UCI Tetouan
power-consumption dataset (10-minute resolution, 3 distribution zones).

LF(d, h) = P_bar(d, h) / max_h' P_bar(d, h')

where P_bar(d, h) is city-wide (all zones summed) average power in hour h
of day d.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ZONE_COLUMNS = [
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
]


def load_raw(raw_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df["datetime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_hourly_load_factor(raw_csv: str | Path) -> pd.DataFrame:
    """Return a tidy dataframe with columns [datetime, date, hour, load_factor],
    one row per hour, fully continuous (no gaps)."""
    df = load_raw(raw_csv)
    df["total_power"] = df[ZONE_COLUMNS].sum(axis=1)

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    hourly = (
        df.groupby(["date", "hour"])["total_power"]
        .mean()
        .reset_index()
        .sort_values(["date", "hour"])
        .reset_index(drop=True)
    )

    daily_max = hourly.groupby("date")["total_power"].transform("max")
    hourly["load_factor"] = hourly["total_power"] / daily_max

    hourly["datetime"] = pd.to_datetime(hourly["date"]) + pd.to_timedelta(
        hourly["hour"], unit="h"
    )
    hourly = hourly.sort_values("datetime").reset_index(drop=True)

    # sanity check: fully continuous, one row per hour
    expected = pd.date_range(
        hourly["datetime"].iloc[0], hourly["datetime"].iloc[-1], freq="h"
    )
    assert len(hourly) == len(expected), "gap detected in reconstructed hourly series"
    assert (hourly["datetime"].values == expected.values).all()

    return hourly[["datetime", "date", "hour", "total_power", "load_factor"]]


def save_processed(hourly: pd.DataFrame, out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False)
