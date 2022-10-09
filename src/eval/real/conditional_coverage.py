"""Conditional coverage evaluation on real data."""

from argparse import ArgumentParser

import numpy as np
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
parser.add_argument("--event", type=str)

args = parser.parse_args()


def run(
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
    y_sub: pd.DataFrame,
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
    train_index = X_train.index
    cal_index = X_cal.index
    test_index = X_test.index

    X_sub = pd.concat([X_train, X_cal, X_test], axis=0)
    y_sub = y_sub[X_sub.index]

    assert (X_sub.index == y_sub.index).all()
    assert X_sub.index.is_unique
    assert X_sub.index.is_monotonic_increasing

    assert len(train_index) == n_train
    assert len(cal_index) <= n_cal
    assert len(test_index) == n_test

    if len(cal_index) == 0:
        return {
            "empirical_coverage": np.nan,
            "average_interval_size": np.nan,
            "prediction_point": np.nan,
            "y_pred_lower": np.nan,
            "y_pred_upper": np.nan,
            "y_true": np.nan,
        }

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
    event: str,
) -> None:
    # Load dataset
    data = get_data(target=dataset, target_gap=1, maxlags=lags, year=year)
    index = data.index
    X = data.drop("target", axis=1)
    y = data["target"]

    # Set event based on previous year data and retrieve test points of interest
    X_prev = get_data(
        target=dataset,
        target_gap=1,
        maxlags=10,
        year=year-1,
    ).drop("target", axis=1)

    match event:
        case "highvol":
            X_event = X.std(axis=1) > X_prev.std(axis=1).quantile(0.8)
        case "lowvol":
            X_event = X.std(axis=1) < X_prev.std(axis=1).quantile(0.2)
        case "uptrend":
            X_event = (X.iloc[:, :2] > 0).all(axis=1)
        case "downtrend":
            X_event = (X.iloc[:, :2] < 0).all(axis=1)

    event_index = index[X_event]

    Model = get_model(quantile_model)

    res = Parallel(n_jobs=n_jobs)(
        delayed(run)(
            X_train=X.loc[:t].iloc[-(n_train + n_cal + n_test) : -(n_cal + n_test)],
            X_cal=X.loc[:t].iloc[-(n_cal + n_test) : -n_test][
                X_event[:t].iloc[-(n_cal + n_test) : -n_test]
            ],
            X_test=X.loc[[t]],
            y_sub=y.loc[:t].iloc[-(n_train + n_cal + n_test):],
            Model=Model,
            **vars(args),
        ) for t in tqdm(event_index) if len(y[:t]) >= n_train + n_cal + n_test
    )

    df = pd.DataFrame(res)

    df["dataset"] = dataset
    df["event"] = event
    df["model"] = quantile_model
    df["alpha"] = alpha
    df["n_train"] = n_train
    df["n_cal"] = n_cal
    df["n_test"] = n_test
    df["n_lags"] = lags

    df = df[df["prediction_point"].notna()]
    assert df["prediction_point"].is_unique
    assert df["prediction_point"].is_monotonic_increasing

    outpath = get_dir("eval/results/real/conditional_coverage")

    df.to_csv(
        f"{outpath}/{event}-{dataset}-{quantile_model}-alpha_{alpha}-" +
        f"n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}-year_{year}.csv",
        index=False,
    )


if __name__ == "__main__":
    main(**vars(args))
