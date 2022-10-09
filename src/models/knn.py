"""k-Nearest Neighbors Quantile Regressor."""

from numpy.typing import ArrayLike, NDArray
from sklearn_quantile import KNeighborsQuantileRegressor


class KNNQR:
    """k-Nearest Neighbors Quantile Regressor."""

    def __init__(
        self,
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize model with desired miscoverage level."""
        self.alpha = alpha
        # Discard dummy seed kept only for compatibility with other models; kNN is deterministic.
        del seed

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "KNNQR":
        """Train quantile regression based on k-nearest neighbors."""
        self.qr = KNeighborsQuantileRegressor(
            q=[self.alpha / 2, 1 - self.alpha / 2],
        ).fit(X, y)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""
        y_pred = self.qr.predict(X)

        y_pred_lower = y_pred[0]
        y_pred_upper = y_pred[1]

        return y_pred_lower, y_pred_upper
