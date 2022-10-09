"""Evaluation of conformal prediction performance based on coverage and interval sizes."""

import numpy as np
from numpy.typing import ArrayLike


def empirical_coverage(
    y_test: ArrayLike,
    y_pred_lower: ArrayLike,
    y_pred_upper: ArrayLike,
) -> float:
    """Calculate average number of times that realized values fell inside prediction intervals."""
    # Ensure data type is numpy array
    y_test = np.asarray(y_test)
    y_pred_lower = np.asarray(y_pred_lower)
    y_pred_upper = np.asarray(y_pred_upper)

    if not (y_pred_upper >= y_pred_lower).all():
        raise ValueError("There is an upper bound strictly smaller than a lower bound.")

    empirical_coverage = ((y_test >= y_pred_lower) * (y_test <= y_pred_upper)).mean().item()

    assert isinstance(empirical_coverage, float)

    return empirical_coverage


def average_interval_size(
    y_pred_lower: ArrayLike,
    y_pred_upper: ArrayLike,
) -> float:
    """Calculate average size of prediction intervals."""
    # Ensure data type is numpy array
    y_pred_lower = np.asarray(y_pred_lower)
    y_pred_upper = np.asarray(y_pred_upper)

    if not (y_pred_upper >= y_pred_lower).all():
        raise ValueError("Upper bounds should be larger than or equal to lower bounds.")

    average_interval_size = (y_pred_upper - y_pred_lower).mean().item()

    assert isinstance(average_interval_size, float)

    return average_interval_size


def coverage_mean_absolute_error(
    theoretical_coverage: float,
    empirical_coverages: ArrayLike,
) -> float:
    """Calculate mean absolute error between empirical coverages and theoretical value."""
    empirical_coverages = np.asarray(empirical_coverages)
    coverage_mae = np.abs(empirical_coverages - theoretical_coverage).mean().item()
    assert isinstance(coverage_mae, float)
    return coverage_mae


def coverage_mean_square_error(
    theoretical_coverage: float,
    empirical_coverages: ArrayLike,
) -> float:
    """Calculate mean squared error between empirical coverages and theoretical value."""
    empirical_coverages = np.asarray(empirical_coverages)
    coverage_mse = ((empirical_coverages - theoretical_coverage) ** 2).mean().item()
    assert isinstance(coverage_mse, float)
    return coverage_mse
