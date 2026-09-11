"""Download the raw UCI 'Power Consumption of Tetouan City' dataset
(10-minute resolution, 3 distribution zones, full year 2017).

Source: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city

Usage:
    py -3 scripts/download_data.py
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lf_forecast.config import resolve

URL = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"


def main() -> None:
    raw_dir = resolve("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "tetouan.zip"

    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    print(f"Extracting to {raw_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    print("Done. Run scripts/prepare_data.py next.")


if __name__ == "__main__":
    main()
