"""Quantile regression models."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn import linear_model
from sklearn_quantile import KNeighborsQuantileRegressor, RandomForestQuantileRegressor
from torch.utils.data import DataLoader, Dataset


class QuantileRegressor(ABC):
    """Quantile Regressor Abstract Base Class."""

    def __init__(
        self,
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize model with desired miscoverage level."""
        self.alpha = alpha
        self.seed = seed

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "QuantileRegressor":
        """Train model to estimate quantiles."""

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""

    @staticmethod
    def _monotonize_curves(
        y_pred_lower: ArrayLike,
        y_pred_upper: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Monotonize curves from arbitrary quantile regression model.

        Swap lower and upper predictions if the former is greater than the latter
        in order to build a proper interval.
        This can be seen as a particular case of the methodology described in
        'Quantile and Probability Curves without Crossing, 2010'.
        """
        if (y_pred_lower > y_pred_upper).any():
            y_preds = np.array([y_pred_lower, y_pred_upper]).T
            y_preds = np.sort(y_preds, axis=1)
            y_pred_lower, y_pred_upper = y_preds[:, 0], y_preds[:, 1]
        return y_pred_lower, y_pred_upper


class GradientBoostingQR(QuantileRegressor):
    """Gradient Boosting Quantile Regressor."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "QuantileRegressor":
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
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper


class KNNQR(QuantileRegressor):
    """k-Nearest Neighbors Quantile Regressor."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "QuantileRegressor":
        """Train quantile regression based on k-nearest neighbors."""
        self.qr = KNeighborsQuantileRegressor(
            q=(self.alpha / 2, 1 - self.alpha / 2),
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
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper


class LinearQR(QuantileRegressor):
    """Linear Quantile Regressor."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "QuantileRegressor":
        """Train lower and upper quantile models."""
        self.qr_lower = linear_model.QuantileRegressor(
            quantile=(self.alpha / 2),
            alpha=1,  # L1 penalty
            solver="highs",
        ).fit(X, y)

        self.qr_upper = linear_model.QuantileRegressor(
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
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper


class RandomForestQR(QuantileRegressor):
    """Random Forest Quantile Regressor."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "QuantileRegressor":
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
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper


class NeuralNetworkQR(QuantileRegressor):
    """Neural Network Quantile Regressor."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        alpha: float,
        seed: int,
    ) -> None:
        """Initialize attributes particular to neural networks."""
        super().__init__(alpha, seed)
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.quantiles = [self.alpha / 2, 1 - self.alpha / 2]
        self.batch_size = 64
        torch.manual_seed(self.seed)

    def _train_loop(
        self,
        dataloader: DataLoader,
        model: "NeuralNetwork",
        loss_fn: "PinballLoss",
        optimizer: torch.optim.AdamW,
    ) -> None:
        """Train a single epoch."""
        model.train()
        for X, y in dataloader:
            # Send tensors to desired device
            X = X.to(NeuralNetworkQR.device)
            y = y.to(NeuralNetworkQR.device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Adjust weights via backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit(
        self,
        X: ArrayLike | torch.Tensor,
        y: ArrayLike | torch.Tensor,
    ) -> "QuantileRegressor":
        """Train single model for upper and lower quantile prediction."""
        self.model = self.NeuralNetwork(
            input_size=X.shape[1],
            n_quantiles=2,
        ).to(NeuralNetworkQR.device)

        loss_fn = self.PinballLoss(self.quantiles)

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = self.CustomDataset(X, y)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.epochs):
            self._train_loop(dataloader, self.model, loss_fn, optimizer)

        return self

    def predict(
        self,
        X: ArrayLike | torch.Tensor,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""
        X = torch.as_tensor(X).float().to(NeuralNetworkQR.device)
        y_pred = self.model(X)
        y_pred = y_pred.detach().cpu().numpy()
        y_pred_lower = y_pred[:, 0]
        y_pred_upper = y_pred[:, 1]
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

        return y_pred_lower, y_pred_upper

    class NeuralNetwork(nn.Module):
        """Standard neural network."""

        def __init__(self, input_size: int, n_quantiles: int) -> None:
            """Initialize model with input and output sizes and architecture."""
            super().__init__()
            self.input_size = input_size
            self.n_quantiles = n_quantiles

            self.mlp = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_quantiles),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = self.mlp(x)
            return x

    class PinballLoss(nn.Module):
        """Quantile loss function, also known as pinball loss."""

        def __init__(
            self, quantiles: list[float],
        ) -> None:
            """Initialize loss with quantiles to be estimated."""
            super().__init__()
            self.quantiles = torch.as_tensor(quantiles).to(NeuralNetworkQR.device)

        def forward(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass."""
            assert not target.requires_grad
            assert preds.size(0) == target.size(0)
            assert preds.size(1) == len(self.quantiles)
            assert target.size(1) == 1
            errors = target - preds
            losses = torch.maximum(
                (self.quantiles - 1) * errors,
                self.quantiles * errors,
            )
            loss = torch.sum(losses, dim=1).mean()
            return loss

    class CustomDataset(Dataset):
        """Custom dataset class."""

        def __init__(
            self,
            values: ArrayLike | torch.Tensor,
            labels: ArrayLike | torch.Tensor,
        ):
            """Initialize class by converting data to appropriate tensors."""
            super().__init__()
            self.values = torch.as_tensor(values).float().to(NeuralNetworkQR.device)
            self.labels = torch.as_tensor(labels.reshape(-1, 1)).float().to(NeuralNetworkQR.device)

        def __len__(self) -> int:
            """Length."""
            return len(self.labels)

        def __getitem__(
            self,
            index: ArrayLike,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Get item."""
            return self.values[index], self.labels[index]
