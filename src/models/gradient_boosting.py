"""Gradient Boosting Quantile Regressor."""

from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike, NDArray


class GradientBoostingQR:
    """Gradient Boosting Quantile Regressor."""

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
    ) -> "GradientBoostingQR":
        """Train lower and upper quantile models."""
        self.qr_lower = LGBMRegressor(
            alpha=(self.alpha / 2),
            objective="quantile",
            metric="quantile",
            boosting_type="gbdt",
            deterministic=True,
            force_row_wise=True,
            seed=self.seed,
        ).fit(X, y)

        self.qr_upper = LGBMRegressor(
            alpha=(1 - self.alpha / 2),
            objective="quantile",
            metric="quantile",
            boosting_type="gbdt",
            deterministic=True,
            force_row_wise=True,
            seed=self.seed,
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
