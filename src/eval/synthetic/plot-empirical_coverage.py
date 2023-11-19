"""Plot empirical coverage and theoretical bound."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.general import get_dir


def main(
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    lags: int,
    quantile_model: str,
) -> None:
    inpath = "eval/results/synthetic/empirical_coverage"
    outpath = get_dir("eval/plots")

    pq_list = (np.arange(30, 51) / 100).tolist()

    res = []

    for pq in pq_list:
        df = pd.read_csv(
            f"{inpath}/two_state_markov_chain-p_{pq}-q_{pq}-{quantile_model}-" +
            f"alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
        )
        min_coverage = df["empirical_coverage"].min()
        max_coverage = df["empirical_coverage"].max()
        bound_eta = (1 - df["alpha"] - df["eta"]).drop_duplicates().item()
        guarantee = df["guarantee"].mean().item()
        bound_delta = (1 - df["delta_cal"] - df["delta_test"]).drop_duplicates().item()
        res.append({
            "min_coverage": min_coverage,
            "max_coverage": max_coverage,
            "bound_eta": bound_eta,
            "guarantee": guarantee,
            "bound_delta": bound_delta,
            "pq": pq,
            "1mpq": 1 - pq,
        })

    df = pd.DataFrame(res)

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.fill_between(
        df["1mpq"],
        df["min_coverage"],
        df["max_coverage"],
        alpha=0.3,
        label="Empirical coverage",
    )

    ax.plot(
        df["1mpq"],
        df["bound_eta"],
        "-",
        label="Theoretical bound",
    )

    plt.xticks([0.5, 0.55, 0.6, 0.65, 0.7])

    plt.xlabel("Dependence factor")
    plt.ylabel("Coverage")

    plt.legend(loc="center right")

    plt.tight_layout()

    plt.savefig(
        f"{outpath}/empirical_coverage-hmm-pq_{pq}-{quantile_model}-" +
        f"alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.pdf",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_cal", type=int, default=15000)
    parser.add_argument("--n_test", type=int, default=15000)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--quantile_model", type=str, default="boosting")
    args = parser.parse_args()

    main(**vars(args))
