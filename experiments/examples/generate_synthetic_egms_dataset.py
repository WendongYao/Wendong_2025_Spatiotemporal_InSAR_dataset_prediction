"""
Generate a compact synthetic EGMS-like CSV for repository test cases.

The production manuscript workflow uses an external EGMS CSV that is not
redistributed in the repository. This script creates a shareable CSV that
matches the column layout expected by the experiment code:

- metadata columns at indices 0..10
- 300 history columns at indices 11..310
- one unused placeholder column at index 311
- one target column at index 312
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataframe(nx: int, ny: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 19000.0, nx)
    ys = np.linspace(0.0, 19000.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    easting = xx.reshape(-1)
    northing = yy.reshape(-1)
    n_points = easting.size

    x_norm = (easting - easting.mean()) / max(np.ptp(easting), 1.0)
    y_norm = (northing - northing.mean()) / max(np.ptp(northing), 1.0)
    radius = np.sqrt(x_norm**2 + y_norm**2)

    time_axis = np.arange(301, dtype=np.float32)
    phase = 2.0 * np.pi * (0.35 * x_norm + 0.2 * y_norm)
    trend = (0.015 + 0.025 * (x_norm + 0.8))[:, None] * time_axis[None, :]
    seasonal = 0.55 * np.sin((2.0 * np.pi / 36.0) * time_axis[None, :] + phase[:, None])
    hotspot = np.exp(-((x_norm - 0.12) ** 2 + (y_norm + 0.08) ** 2) / 0.015)[:, None]
    transient = hotspot * (0.9 / (1.0 + np.exp(-(time_axis[None, :] - 220.0) / 12.0)))
    longwave = 0.25 * np.cos((2.0 * np.pi / 120.0) * time_axis[None, :] + 0.5 * phase[:, None])
    noise = rng.normal(loc=0.0, scale=0.035, size=(n_points, time_axis.size)).astype(np.float32)
    values = trend + seasonal + transient + longwave + noise

    quality = np.clip(0.98 - 0.22 * radius + rng.normal(0.0, 0.015, size=n_points), 0.45, 0.99)

    data: dict[str, np.ndarray] = {
        "point_id": np.arange(n_points, dtype=np.int32),
        "easting": easting.astype(np.float32),
        "northing": northing.astype(np.float32),
        "los_placeholder": np.zeros(n_points, dtype=np.float32),
        "quality_score": quality.astype(np.float32),
        "meta_05": (0.2 + 0.1 * x_norm).astype(np.float32),
        "meta_06": (0.15 + 0.1 * y_norm).astype(np.float32),
        "meta_07": (radius).astype(np.float32),
        "meta_08": np.sin(phase).astype(np.float32),
        "meta_09": np.cos(phase).astype(np.float32),
        "meta_10": np.ones(n_points, dtype=np.float32),
    }

    for idx in range(300):
        data[f"hist_{idx:03d}"] = values[:, idx].astype(np.float32)

    data["unused_311"] = values[:, 299].astype(np.float32)
    data["target_300"] = values[:, 300].astype(np.float32)
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic EGMS-like CSV for test cases.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path. Defaults to examples/synthetic_egms_small.csv")
    parser.add_argument("--nx", type=int, default=20, help="Number of points in the easting direction.")
    parser.add_argument("--ny", type=int, default=20, help="Number of points in the northing direction.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for the synthetic generator.")
    args = parser.parse_args()

    default_output = Path(__file__).resolve().parent / "synthetic_egms_small.csv"
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.nx, args.ny, args.seed)
    df.to_csv(output_path, index=False)
    print(
        {
            "output_path": str(output_path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }
    )


if __name__ == "__main__":
    main()
