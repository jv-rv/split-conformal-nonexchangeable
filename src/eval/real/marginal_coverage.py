"""Marginal coverage evaluation on real data."""

from argparse import ArgumentParser

import pandas as pd
from joblib import delayed, Parallel
from numpy.typing import NDArray

from src.models import ConformalQR
from src.models.quantile_regressors import QuantileRegressor
from src.utils import eval
from src.utils.data import get_data, SequentialSplit
from src.utils.general import get_dir
from src.utils.model import get_model


def run(
    Model: QuantileRegressor,
    X_train: NDArray,
    y_train: NDArray,
    X_cal: NDArray,
    y_cal: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    prediction_point: pd.Timestamp,
    alpha: float,
    **kwargs: int | float | str,
) -> dict[str, float]:
    """Run a single experiment."""

    # Generate prediction intervals
    cqr = ConformalQR(Model, alpha, seed=0)
    cqr.fit(X_train, y_train)
    cqr.calibrate(X_cal, y_cal)
    y_pred_lower, y_pred_upper = cqr.predict(X_test)

    # Evaluate
    empirical_coverage = eval.empirical_coverage(y_test, y_pred_lower, y_pred_upper)
    average_interval_size = eval.average_interval_size(y_pred_lower, y_pred_upper)

    return {
        "empirical_coverage": empirical_coverage,
        "average_interval_size": average_interval_size,
        "prediction_point": prediction_point,
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

    # Split data in features and target
    X = data.drop("target", axis=1).to_numpy()
    y = data["target"].to_numpy()

    # Retrieve quantile regression model
    Model = get_model(quantile_model)

    # Set sequential split sizes
    sizes = (n_train, n_cal, n_test)

    res = Parallel(n_jobs=n_jobs)(
        delayed(run)(
            Model=Model,
            X_train=X[train_index],
            y_train=y[train_index],
            X_cal=X[cal_index],
            y_cal=y[cal_index],
            X_test=X[test_index],
            y_test=y[test_index],
            prediction_point=index[test_index].item(),
            **vars(args),
        ) for train_index, cal_index, test_index in SequentialSplit(sizes).split(X)
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
    parser = ArgumentParser()
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

    main(**vars(args))
