"""Conformal Quantile Regressor."""

from typing import Self, Type

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.quantile_regressors import QuantileRegressor


class ConformalQR(QuantileRegressor):
    """Conformal Quantile Regressor."""

    def __init__(
        self,
        Model: Type[QuantileRegressor],
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize CQR with desired base model and miscoverage level."""
        super().__init__(alpha, seed)
        self.Model = Model

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Fit base model."""
        self.model = self.Model(alpha=self.alpha, seed=self.seed).fit(X, y)
        return self

    def calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Calibrate via plug-in prediction interval error (Romano, Patterson, and Candès 2019)."""
        y_pred_lower, y_pred_upper = self.model.predict(X)
        scores = np.maximum(y_pred_lower - y, y - y_pred_upper)
        self.q_hat = np.quantile(scores, 1 - self.alpha)
        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""

        # Generate prediction set using base quantile regressor
        y_pred_lower, y_pred_upper = self.model.predict(X)

        # Conformalize prediction set
        y_pred_lower = y_pred_lower - self.q_hat
        y_pred_upper = y_pred_upper + self.q_hat
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper
