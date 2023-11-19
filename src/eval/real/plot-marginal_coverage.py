"""Plot marginal coverage of real-world experiments over aggregated period."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.general import get_dir


def main(
    datasets: str,
    quantile_model: str,
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    lags: int,
    aggregation: str,
) -> None:
    inpath = "eval/results/real/marginal_coverage"
    outpath = get_dir("eval/plots")

    if len(datasets) == 1:
        plt.rcParams.update({"font.size": 8})
        fig, axs = plt.subplots(ncols=len(datasets), nrows=1, figsize=(6, 2), sharex=True)
        axs = [axs]
    else:
        fig, axs = plt.subplots(ncols=len(datasets), nrows=1, figsize=(10, 2), sharey=True)

    for i, dataset in enumerate(datasets):
        df = pd.read_csv(
            f"{inpath}/{dataset}-{quantile_model}-alpha_{alpha}-" +
            f"n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
            usecols=["prediction_point", "empirical_coverage"],
            index_col="prediction_point",
            parse_dates=["prediction_point"],
        ).squeeze()

        assert df.index.is_monotonic_increasing

        # Subset data to Mon-Thu because
        #   Fri: around 70% -- 80% of usual data points
        #        market closes early, around 16:00 -- 17:00;
        #   Sat: no transactions whatsoever;
        #   Sun: around 20% -- 30% of usual data points
        #        market opens late, around 17:00 -- 20:00.
        # Variation is due to different assets (EUR, BCO, SPX).
        df = df[df.index.dayofweek <= 3]

        if aggregation == "day":
            df.index = df.index.floor("d")
            xlabel = "Days"
        elif aggregation == "week":
            df.index = df.index.isocalendar().week
            xlabel = "Weeks"
        elif aggregation == "hour":
            df.index = df.index.floor("h")
            xlabel = "Hours"

        df.groupby(level=0).mean().plot(ax=axs[i], color="#000075", linewidth=1)

        axs[i].axhline(1-alpha, linewidth=1, linestyle="dashed", color="black")
        axs[i].axhline(df.mean(), linewidth=1, linestyle="dashed", color="#ff7700")

        axs[i].set_title(dataset)

        axs[i].set_xlabel(xlabel)
        axs[i].set_yticks([1 - alpha + i/100 for i in range(-2, 6, 2)])

        axs[i].minorticks_off()
        axs[i].get_xaxis().set_ticks([])

        if i > 0:
            axs[i].axes.get_yaxis().set_visible(False)

    # Export figure
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        f"{outpath}/marginal_coverage-{'_'.join(datasets)}-{quantile_model}-" +
        f"alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.pdf",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["eurusd", "bcousd", "spxusd"])
    parser.add_argument("--quantile_model", type=str, default="boosting")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_cal", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=1)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--aggregation", type=str, default="day")
    args = parser.parse_args()

    main(**vars(args))
