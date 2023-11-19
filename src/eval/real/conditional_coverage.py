"""Conditional coverage evaluation on real data."""

from argparse import ArgumentParser
from typing import Type

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from numpy.typing import NDArray
from tqdm import tqdm

from src.models import ConformalQR
from src.models.quantile_regressors import QuantileRegressor
from src.utils import eval
from src.utils.data import get_data
from src.utils.general import get_dir
from src.utils.model import get_model


def run(
    X_train: NDArray,
    y_train: NDArray,
    X_cal: NDArray,
    y_cal: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    Model: Type[QuantileRegressor],
    prediction_point: pd.Timestamp,
    alpha: float,
    **kwargs: int | float | str,
) -> dict[str, float]:
    """Run a single experiment."""
    del kwargs

    if len(y_cal) == 0:
        return {
            "empirical_coverage": np.nan,
            "average_interval_size": np.nan,
            "prediction_point": np.nan,
            "y_pred_lower": np.nan,
            "y_pred_upper": np.nan,
            "y_true": np.nan,
        }

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
    event: str,
) -> None:
    # Load dataset
    data = get_data(target=dataset, target_gap=1, maxlags=lags, year=year)
    index = data.index

    # Split data in features and target
    X = data.drop("target", axis=1)
    y = data["target"]
    index = data.index

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
            X_train=X.loc[:t].iloc[-(n_train + n_cal + n_test) : -(n_cal + n_test)].to_numpy(),
            y_train=y.loc[:t].iloc[-(n_train + n_cal + n_test) : -(n_cal + n_test)].to_numpy(),
            X_cal=X.loc[:t].iloc[-(n_cal + n_test) : -n_test][
                X_event[:t].iloc[-(n_cal + n_test) : -n_test]
            ].to_numpy(),
            y_cal=y.loc[:t].iloc[-(n_cal + n_test) : -n_test][
                X_event[:t].iloc[-(n_cal + n_test) : -n_test]
            ].to_numpy(),
            X_test=X.loc[[t]].to_numpy(),
            y_test=y.loc[[t]].to_numpy(),
            Model=Model,
            prediction_point=t,
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
    parser.add_argument("--event", type=str)
    args = parser.parse_args()

    main(**vars(args))
