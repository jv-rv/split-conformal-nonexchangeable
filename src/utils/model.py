"""Model utilities."""


from typing import Type

from src.models import GradientBoostingQR, KNNQR, LinearQR, NeuralNetworkQR, RandomForestQR


def get_model(
    quantile_model: str,
) -> Type[GradientBoostingQR | KNNQR | LinearQR | NeuralNetworkQR | RandomForestQR]:
    """Return quantile regression model based on name."""
    match quantile_model:
        case "boosting":
            return GradientBoostingQR
        case "knn":
            return KNNQR
        case "linear_regression":
            return LinearQR
        case "neural_network":
            return NeuralNetworkQR
        case "random_forest":
            return RandomForestQR
        case _:
            raise ValueError(f"Quantile regression model {quantile_model} not available.")
