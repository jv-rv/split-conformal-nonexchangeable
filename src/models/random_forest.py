"""Random Forest Quantile Regressor."""

from numpy.typing import ArrayLike, NDArray
from sklearn_quantile import RandomForestQuantileRegressor


class RandomForestQR:
    """Random Forest Quantile Regressor."""

    def __init__(
        self,
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize model with desired miscoverage level."""
        self.alpha = alpha
        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "RandomForestQR":
        """Train Quantile Regression Forest (Meinshausen, 2006)."""
        self.qr = RandomForestQuantileRegressor(
            q=[self.alpha / 2, 1 - self.alpha / 2],
            random_state=self.seed,
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
