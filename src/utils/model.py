"""Model utilities."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from src.models import GradientBoostingQR, KNNQR, LinearQR, NeuralNetworkQR, RandomForestQR


def conformalized_quantile_regression(
    Model: GradientBoostingQR | KNNQR | LinearQR | NeuralNetworkQR | RandomForestQR,
    alpha: float,
    seed: int | None,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_index: pd.Index,
    cal_index: pd.Index,
    test_index: pd.Index,
) -> tuple[NDArray, NDArray]:
    """Produce valid upper and lower prediction sets for a given miscoverage level alpha."""
    # Split data into training, calibration and test sets
    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_cal, y_cal = X.loc[cal_index], y.loc[cal_index]
    X_test = X.loc[test_index]

    # Convert dataframes to numpy arrays
    X_train = X_train.to_numpy()
    X_cal = X_cal.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_cal = y_cal.to_numpy()

    # Train
    model = Model(alpha=alpha, seed=seed).fit(X_train, y_train)

    # Calibrate
    y_pred_lower_cal, y_pred_upper_cal = model.predict(X_cal)

    if (y_pred_lower_cal > y_pred_upper_cal).any():
        y_pred_lower_cal, y_pred_upper_cal = monotonize_quantile_regression_curves(
            y_pred_lower_cal,
            y_pred_upper_cal,
        )

    scores = np.maximum(y_pred_lower_cal - y_cal, y_cal - y_pred_upper_cal)

    q_hat = np.quantile(scores, 1 - alpha)

    # Make predictions for test data
    y_pred_lower, y_pred_upper = model.predict(X_test)

    if (y_pred_lower > y_pred_upper).any():
        y_pred_lower, y_pred_upper = monotonize_quantile_regression_curves(
            y_pred_lower,
            y_pred_upper,
        )

    # Conformalize prediction sets
    y_pred_lower = y_pred_lower - q_hat
    y_pred_upper = y_pred_upper + q_hat

    if (y_pred_lower > y_pred_upper).any():
        y_pred_lower, y_pred_upper = monotonize_quantile_regression_curves(
            y_pred_lower,
            y_pred_upper,
        )

    return y_pred_lower, y_pred_upper


def monotonize_quantile_regression_curves(
    y_pred_lower: ArrayLike,
    y_pred_upper: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Monotonize curves from arbitrary quantile regression model.

    Swap lower and upper predictions if the former is greater than the latter
    in order to build a proper interval.
    This can be seen as a particular case of the methodology described in
    'Quantile and Probability Curves without Crossing, 2010'.
    """
    y_preds = np.array([y_pred_lower, y_pred_upper]).T
    y_preds = np.sort(y_preds, axis=1)
    y_pred_lower, y_pred_upper = y_preds[:, 0], y_preds[:, 1]
    return y_pred_lower, y_pred_upper


def get_model(
    quantile_model: str,
) -> GradientBoostingQR | KNNQR | LinearQR | NeuralNetworkQR | RandomForestQR:
    """Return quantile regression model based on name."""
    match quantile_model:
        case "boosting":
            Model = GradientBoostingQR
        case "knn":
            Model = KNNQR
        case "linear_regression":
            Model = LinearQR
        case "neural_network":
            Model = NeuralNetworkQR
        case "random_forest":
            Model = RandomForestQR
        case _:
            raise ValueError(f"Quantile regression model {quantile_model} not available.")
    return Model
