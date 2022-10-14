"""Model utilities."""


from src.models import GradientBoostingQR, KNNQR, LinearQR, NeuralNetworkQR, RandomForestQR
from src.models.quantile_regressors import QuantileRegressor


def get_model(
    quantile_model: str,
) -> QuantileRegressor:
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
