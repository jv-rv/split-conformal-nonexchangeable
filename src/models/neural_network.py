"""Neural Network Quantile Regressor."""

import torch
import torch.nn as nn
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetworkQR:
    """Neural Network Quantile Regressor."""

    def __init__(
        self,
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize model with desired miscoverage level and hyperparameters."""
        self.alpha = alpha
        self.seed = seed
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.quantiles = [alpha / 2, 1 - alpha / 2]
        self.batch_size = 64
        torch.manual_seed(seed)

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
            X, y = X.to(device), y.to(device)

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
    ) -> "NeuralNetworkQR":
        """Train single model for upper and lower quantile prediction."""
        self.model = NeuralNetwork(input_size=X.shape[1], n_quantiles=2).to(device)

        loss_fn = PinballLoss(self.quantiles)

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = CustomDataset(X, y)

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
        X = torch.as_tensor(X).float().to(device)
        y_pred = self.model(X)
        y_pred = y_pred.detach().cpu().numpy()
        y_pred_lower = y_pred[:, 0]
        y_pred_upper = y_pred[:, 1]

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
        self.quantiles = torch.as_tensor(quantiles).to(device)

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
        self.values = torch.as_tensor(values).float().to(device)
        self.labels = torch.as_tensor(labels.reshape(-1, 1)).float().to(device)

    def __len__(self) -> int:
        """Length."""
        return len(self.labels)

    def __getitem__(
        self,
        index: ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item."""
        return self.values[index], self.labels[index]
