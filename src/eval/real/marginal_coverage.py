"""Marginal coverage evaluation on real data."""

from argparse import ArgumentParser

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.models import GradientBoostingQR, KNNQR, LinearQR, NeuralNetworkQR, RandomForestQR
from src.utils import eval
from src.utils.data import get_data
from src.utils.general import get_dir
from src.utils.model import conformalized_quantile_regression, get_model

parser = ArgumentParser()

# Set common parameters
parser.add_argument("--dataset", type=str)
parser.add_argument("--year", type=int, default=2021)
parser.add_argument("--quantile_model", type=str, default="boosting")
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_cal", type=int, default=500)
parser.add_argument("--n_test", type=int, default=1)
parser.add_argument("--lags", type=int, default=10)
parser.add_argument("--n_jobs", type=int, default=-1)

args = parser.parse_args()


def run(
    X_sub: pd.DataFrame,
    y_sub: pd.DataFrame,
    index_sub: pd.Index,
    Model: GradientBoostingQR | KNNQR | LinearQR | NeuralNetworkQR | RandomForestQR,
    dataset: str,
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    **kwargs: int | float | str,
) -> dict[str, float]:
    """Run a single experiment."""
    del kwargs
    train_index = index_sub[:n_train]
    cal_index = index_sub[n_train : n_train + n_cal]
    test_index = index_sub[n_train + n_cal : n_train + n_cal + n_test]

    y_pred_lower, y_pred_upper = conformalized_quantile_regression(
        Model=Model,
        alpha=alpha,
        seed=0,
        X=X_sub,
        y=y_sub,
        train_index=train_index,
        cal_index=cal_index,
        test_index=test_index,
    )
    y_test = y_sub[test_index]

    empirical_coverage = eval.empirical_coverage(y_test, y_pred_lower, y_pred_upper)
    average_interval_size = eval.average_interval_size(y_pred_lower, y_pred_upper)

    return {
        "empirical_coverage": empirical_coverage,
        "average_interval_size": average_interval_size,
        "prediction_point": test_index.item(),
        "y_pred_lower": y_pred_lower.item(),
        "y_pred_upper": y_pred_upper.item(),
        "y_true": y_test.item(),
    }

def main(
    dataset: str,
    year: int,
    quantile_model: str,
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    lags: int,
    n_jobs: int,
) -> None:
    # Load dataset
    data = get_data(target=dataset, target_gap=1, maxlags=lags, year=year)
    index = data.index
    X = data.drop("target", axis=1)
    y = data["target"]

    # Set maximum number of test points available for evaluation
    max_test_points = len(data) - n_train - n_cal

    # Retrieve quantile regression model
    Model = get_model(quantile_model)

    res = Parallel(n_jobs=n_jobs)(
        delayed(run)(
            X_sub=X.iloc[i: i + n_train + n_cal + n_test],
            y_sub=y.iloc[i: i + n_train + n_cal + n_test],
            index_sub=index[i: i + n_train + n_cal + n_test],
            Model=Model,
            **vars(args),
        ) for i in tqdm(range(max_test_points))
    )

    df = pd.DataFrame(res)

    df["model"] = quantile_model
    df["alpha"] = alpha
    df["n_train"] = n_train
    df["n_cal"] = n_cal
    df["n_test"] = n_test
    df["n_lags"] = lags

    assert df["prediction_point"].is_monotonic_increasing

    outpath = get_dir("eval/results/real/marginal_coverage")

    df.to_csv(
        f"{outpath}/{dataset}-{quantile_model}-alpha_{alpha}-" +
        f"n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
        index=False,
    )


if __name__ == "__main__":
    main(**vars(args))
