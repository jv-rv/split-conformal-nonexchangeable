"""Generate table with conditional coverage for distinct events and datasets."""

import os.path
from argparse import ArgumentParser
from itertools import product

import pandas as pd

from src.utils.general import get_dir

parser = ArgumentParser()

# Set common parameters
parser.add_argument("--datasets", nargs="+", default=["eurusd", "bcousd", "spxusd"])
parser.add_argument("--year", type=int, default=2021)
parser.add_argument("--quantile_model", type=str, default="boosting")
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_cal", nargs="+", default=[500, 1000, 5000])
parser.add_argument("--n_test", type=int, default=1)
parser.add_argument("--lags", type=int, default=10)
parser.add_argument("--events", nargs="+", default=["uptrend", "downtrend", "highvol", "lowvol"])

args = parser.parse_args()


def main(
    datasets: list[str],
    year: int,
    quantile_model: str,
    alpha: float,
    n_train: int,
    n_cal: list[int],
    n_test: int,
    lags: int,
    events: list[str],
) -> None:
    inpath = "eval/results/real/conditional_coverage"

    dfs = []

    for n, dataset, event in product(n_cal, datasets, events):
        file = (
            f"{inpath}/{event}-{dataset}-{quantile_model}-alpha_{alpha}-" +
            f"n1_{n_train}-n2_{n}-n3_{n_test}-lags_{lags}-year_{year}.csv"
        )
        if os.path.isfile(file):
            dfs.append(
                pd.read_csv(file, usecols=["dataset", "n_cal", "empirical_coverage", "event"]),
            )

    df = pd.concat(dfs)

    df = df.groupby(["dataset", "n_cal", "event"]).mean("empirical_coverage")

    df["empirical_coverage"] = [f"{c:.2%}" for c in df["empirical_coverage"]]

    df = df.reset_index().pivot(
        index=["dataset", "n_cal"], columns="event", values="empirical_coverage",
    )

    df = df.reindex(datasets, level=0, columns=events)

    df = df.rename(
        {
            "uptrend": "Uptrend",
            "downtrend": "Downtrend",
            "highvol": "High vol.",
            "lowvol": "Low vol.",
        }, axis=1,
    )

    df.index.names = ["Dataset", "Cal. set size"]
    df.columns.name = None

    df = df.reset_index()

    outpath = get_dir("eval/tables")

    df.style.hide(axis="index").format(escape="latex").to_latex(
        f"{outpath}/conditional_coverage-{quantile_model}-alpha_{alpha}-" +
        f"n1_{n_train}-n3_{n_test}-lags_{lags}-year_{year}.tex",
        column_format="c" * df.shape[1],
        hrules=True,
    )


if __name__ == "__main__":
    main(**vars(args))
