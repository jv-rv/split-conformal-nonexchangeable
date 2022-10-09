"""Linear Quantile Regressor."""

from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import QuantileRegressor


class LinearQR:
    """Linear Quantile Regressor."""

    def __init__(
        self,
        alpha: float,
        seed: int | None,  # dummy for compatibility with other models.
    ) -> None:
        """Initialize model with desired miscoverage level."""
        self.alpha = alpha

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "LinearQR":
        """Train lower and upper quantile models."""
        self.qr_lower = QuantileRegressor(
            quantile=(self.alpha / 2),
            alpha=1,  # L1 penalty
            solver="highs",
        ).fit(X, y)

        self.qr_upper = QuantileRegressor(
            quantile=(1 - self.alpha / 2),
            alpha=1,  # L1 penalty
            solver="highs",
        ).fit(X, y)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""
        y_pred_lower = self.qr_lower.predict(X)
        y_pred_upper = self.qr_upper.predict(X)

        return y_pred_lower, y_pred_upper
