"""General utilities."""

import os

import numpy as np
from numpy.typing import NDArray


def get_dir(path: str) -> str:
    """Create a directory given a path."""
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    elif os.path.isfile(path):
        return f"Cannot create directory {path}. File with same name exists."
    return path


def cummean(arr: NDArray) -> NDArray:
    """Return cumulative mean of given data."""
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def cumstd(arr: NDArray) -> NDArray:
    """Return cumulative standard deviation of given data."""
    return np.sqrt(cummean(arr ** 2) - cummean(arr) ** 2)


def cummad(arr: NDArray, target_value: float) -> NDArray:
    """Return cumulative mean absolute deviation between given data and target value."""
    return cummean(np.abs(arr - target_value))
