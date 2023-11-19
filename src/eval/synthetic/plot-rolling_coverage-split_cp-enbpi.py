"""Rolling coverage comparison between Split CP and EnbPI for beta-mixing data."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from src.utils.data import get_synthetic
from src.utils.eval import average_interval_size, empirical_coverage
from src.utils.general import get_dir


def split_cp(
    X: NDArray,
    y: NDArray,
    n_train: int,
    n_test: int,
    alpha: float,
) -> pd.DataFrame:
    """Run Split CP online with expanding calibration set.

    Args:
        X: features matrix
        y: target vector
        n_train: number of points used for training
        n_test: number of test points to be probed
        alpha: miscoverage level

    Returns:
        dataframe with target, predictions, lower and upper intervals
    """
    # Subset training data
    X_train, y_train = X[:n_train, :], y[:n_train]

    # Train model
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    res = []

    for i in tqdm(range(n_test)):
        # Set index of current test point
        test_index = -n_test + i

        # Use as much data as possible for calibration
        X_cal = X[n_train:test_index, :]
        y_cal = y[n_train:test_index]

        # Calculate nonconformity scores as absolute residuals
        scores = np.abs(model.predict(X_cal) - y_cal)

        # Make prediction for single test point
        y_pred = model.predict(X[[test_index]]).item()

        # Generate prediction interval
        d = np.quantile(scores, 1 - alpha)
        y_pred_lower = y_pred - d
        y_pred_upper = y_pred + d

        res.append({
            "target": y[test_index],
            "pred": y_pred,
            "lower": y_pred_lower,
            "upper": y_pred_upper,
        })

    return pd.DataFrame(res)


def enbpi(
    X: NDArray,
    y: NDArray,
    n_test: int,
    alpha: float,
) -> pd.DataFrame:
    """Run EnbPI.

    Args:
        X: features matrix
        y: target vector
        n_test: number of test points to be probed
        alpha: miscoverage level

    Returns:
        dataframe with target, predictions, lower and upper intervals
    """
    # Subset training and test data
    X_train, y_train = X[:-n_test, :], y[:-n_test]
    X_test, y_test = X[-n_test:, :], y[-n_test:]

    # Set model and bootstrap strategy for EnbPI
    model = RandomForestRegressor(random_state=0)
    bb = BlockBootstrap(random_state=0, length=8)

    # Train EnbPI regressor
    mapie_enpbi = MapieTimeSeriesRegressor(
        model,
        method="enbpi",
        cv=bb,
        agg_function="mean",
        n_jobs=1,
    )
    mapie_enpbi = mapie_enpbi.fit(X_train, y_train)

    # Initialize empty arrays to hold results
    y_pred = np.empty(len(y_test))
    y_interval = np.empty((len(y_test), 2, 1))

    # Predict first test point
    y_pred[0], y_interval[0, :, :] = mapie_enpbi.predict(
        X_test[[0]],
        alpha=alpha,
        ensemble=True,
        optimize_beta=True,
    )

    for i in tqdm(range(1, len(X_test))):
        # Update EnbPI with feedback from previous prediction
        _ = mapie_enpbi.partial_fit(
            X_test[[i-1]],
            y_test[i-1],
        )

        # Predict next test point
        y_pred[i], y_interval[i, :, :] = mapie_enpbi.predict(
            X_test[[i]],
            alpha=alpha,
            ensemble=True,
            optimize_beta=True,
        )

    return pd.DataFrame(
        np.vstack([
            y_test,
            y_pred,
            y_interval[:, 0, 0],
            y_interval[:, 1, 0],
        ]).T,
        columns=["target", "pred", "lower", "upper"],
    )


def main(alpha: float, seed: int) -> None:
    # Set total number of points and amount reserved for testing
    N = 10000
    n_test = int(0.3 * N)

    # Generate beta-mixing data from a random walk on the cycle graph
    data = get_synthetic(
        "cycle_random_walk",
        b=0.01,
        s=0.98,
        f=0.01,
        vertices=5,
        N=N,
        lags=10,
        seed=seed,
    )

    # Split data into features and target
    X = data.drop("target", axis=1).to_numpy()
    y = data["target"].to_numpy()

    # Generate prediction intervals via Split CP and EnbPI
    df_scp = split_cp(X, y, n_train=int(0.3 * N), n_test=n_test, alpha=alpha)
    df_enbpi = enbpi(X, y, n_test=n_test, alpha=alpha)

    # Report average interval size
    print(
        "Split CP's average interval size:"
        f"{average_interval_size(df_scp['lower'], df_scp['upper'])}",
    )
    print(
        "EnbPI's average interval size:"
        f"{average_interval_size(df_enbpi['lower'], df_enbpi['upper'])}",
    )

    # Report coverage
    print(
        "Split CP's coverage:"
        f"{empirical_coverage(df_scp['target'], df_scp['lower'], df_scp['upper'])}",
    )
    print(
        "EnbPI's coverage:"
        f"{empirical_coverage(df_enbpi['target'], df_enbpi['lower'], df_enbpi['upper'])}",
    )

    # Calculate coverage over a rolling window
    window = n_test // 6

    def rolling_cov(df: pd.DataFrame, window: int) -> list[float]:
        return [
            empirical_coverage(
                df["target"].iloc[i : i + window],
                df["lower"].iloc[i : i + window],
                df["upper"].iloc[i : i + window],
            ) for i in range(len(df) - window)
        ]

    rolling_cov_scp = rolling_cov(df_scp, window)
    rolling_cov_enbpi = rolling_cov(df_enbpi, window)

    # Plot rolling coverage
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rolling_cov_scp, label="Split CP", color="#0000b8")
    ax.plot(rolling_cov_enbpi, label="EnbPI", color="#b80000")
    plt.ylabel("Rolling coverage")
    plt.xlabel("Starting point")
    plt.axhline(1-alpha, linewidth=1, linestyle="dashed", color="black")
    plt.legend()

    # Save figure
    outpath = get_dir("eval/plots")
    plt.tight_layout()
    plt.savefig(
        f"{outpath}/rolling_coverage-split_cp-enbpi-alpha_{alpha}-seed-{seed}.pdf",
    )
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(**vars(args))
