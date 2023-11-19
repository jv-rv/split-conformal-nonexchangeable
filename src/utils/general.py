"""General utilities."""

import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics.pairwise import haversine_distances


def get_dir(path: str) -> str:
    """Create a directory given a path."""
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    elif os.path.isfile(path):
        return f"Cannot create directory {path}. File with same name exists."
    return path


def get_month_day() -> list[tuple[int, int]]:
    """Generate list with month-day pairs assuming a non-leap year."""
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return [(month + 1, day + 1) for month in range(12) for day in range(days[month])]


def get_haversine_dist_matrix() -> pd.DataFrame:
    """Calculate haversine distance between (latitude, longitude) pairs."""
    res = []
    for month, day in get_month_day():
        lat_lon = pd.read_csv(
            f"data/processed/climatology/{month}-{day}.csv",
            usecols=["lat", "lon"],
        ).drop_duplicates()
        res.append(lat_lon)

    for i in range(1, len(res)):
        assert res[0].equals(res[i])

    dist = haversine_distances(
        lat_lon.apply(np.radians).to_records(index=False).tolist(),
    )

    idx = lat_lon.set_index(["lat", "lon"]).index

    return pd.DataFrame(dist, index=idx, columns=idx)


def cummean(arr: NDArray) -> NDArray:
    """Return cumulative mean of given data."""
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def cumstd(arr: NDArray) -> NDArray:
    """Return cumulative standard deviation of given data."""
    return np.sqrt(cummean(arr ** 2) - cummean(arr) ** 2)


def cummad(arr: NDArray, target_value: float) -> NDArray:
    """Return cumulative mean absolute deviation between given data and target value."""
    return cummean(np.abs(arr - target_value))


def weighted_quantile(arr: NDArray, q: float, weights: NDArray) -> float:
    """Calculate quantile based on weights."""
    idx = np.argsort(arr)
    arr = arr[idx]
    weights = weights[idx]
    weights = np.cumsum(weights)
    weights /= weights[-1]
    wq = arr[np.sum(weights < q)].item()
    return float(wq)
