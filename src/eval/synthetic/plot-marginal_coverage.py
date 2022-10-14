"""Plot marginal coverage of CP for test point with varying dependence."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from src.utils.general import get_dir

parser = ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--models", nargs="+", default=["boosting", "neural_network", "random_forest"])
parser.add_argument("--stochastic_process", type=str, default="two_state_markov_chain")
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_cal", type=int, default=500)
parser.add_argument("--n_test", type=int, default=1)
parser.add_argument("--lags", type=int, default=10)
args = parser.parse_args()


def get_ylim(
    alpha: float,
    stochastic_process: str,
) -> tuple[float, float]:
    """Set ylim to be used in plots."""
    match (alpha, stochastic_process):
        case (0.05, "two_state_markov_chain"):
            ylim = (0.92, 0.96)
        case (0.05, "cycle_random_walk"):
            ylim = (0.90, 0.96)
        case (0.05, "ar1"):
            ylim = (0.84, 0.96)
        case (0.05, "renewal"):
            ylim = (0.94, 0.96)

        case (0.1, "two_state_markov_chain"):
            ylim = (0.865, 0.91)
        case (0.1, "cycle_random_walk"):
            ylim = (0.85, 0.91)
        case (0.1, "ar1"):
            ylim = (0.79, 0.91)
        case (0.1, "renewal"):
            ylim = (0.889, 0.908)

        case (0.15, "two_state_markov_chain"):
            ylim = (0.8, 0.9)
        case (0.15, "cycle_random_walk"):
            ylim = (0.79, 0.87)
        case (0.15, "ar1"):
            ylim = (0.75, 0.9)
        case (0.15, "renewal"):
            ylim = (0.83, 0.86)

        case _:
            ylim = ((1-alpha) * 0.86, (1-alpha) * 1.1)

    return ylim


def main(
    alpha: float,
    models: list[str],
    stochastic_process: str,
    n_train: int,
    n_cal: int,
    n_test: int,
    lags: int,
) -> None:

    model_names = {
        "boosting": "Gradient Boosting",
        "knn": "k-Nearest Neighbors",
        "linear_regression": "Linear Regression",
        "neural_network": "Neural Network",
        "random_forest": "Random Forest",
    }

    plt.rcParams.update({"font.size": 8})

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 2))

    inpath = "eval/results/synthetic/marginal_coverage"
    outpath = get_dir("eval/plots")

    ylim = get_ylim(alpha, stochastic_process)

    if stochastic_process == "two_state_markov_chain":
        pq_list = np.concatenate([[0.001], np.arange(1, 51) / 100]).tolist()

        for i, model in enumerate(models):
            res = []
            for pq in pq_list:
                df = pd.read_csv(
                    f"{inpath}/{stochastic_process}-p_{pq}-q_{pq}-" +
                    f"{model}-alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
                )
                res.append({
                    "coverage": df["empirical_coverage"].mean(),
                    "1mpq": 1 - pq,
                })

            df = pd.DataFrame(res)

            df.plot(
                x="1mpq",
                y="coverage",
                lw=1.5,
                c="#000075",
                legend=None,
                title=model_names[model],
                xlabel="Probability of repeating previous state",
                ylabel="Marginal coverage",
                ax=axs[i],
                ylim=ylim,
            )

            axs[i].axhline(
                y=1-alpha,
                linewidth=1.5,
                color="gray",
                linestyle="dashed",
            )

            axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if i not in [0, 3]:
                axs[i].axes.get_yaxis().set_visible(False)

    elif stochastic_process == "cycle_random_walk":
        vertices = 5
        prob_stay_list = (np.arange(1, 101, 2) / 100).tolist()

        for i, model in enumerate(models):
            res = []
            for prob_stay in prob_stay_list:
                prob_b = prob_f = round(0.5 * (1 - prob_stay), 3)
                df = pd.read_csv(
                    f"{inpath}/{stochastic_process}-" +
                    f"b_{prob_b}-s_{prob_stay}-f_{prob_f}-v_{vertices}-" +
                    f"{model}-alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
                )
                res.append({
                    "coverage": df["empirical_coverage"].mean(),
                    "prob_stay": prob_stay,
                })

            df = pd.DataFrame(res)

            df.plot(
                x="prob_stay",
                y="coverage",
                lw=1.5,
                c="#000075",
                legend=None,
                title=model_names[model],
                xlabel="Probability of not moving on the cycle",
                ylabel="Marginal coverage",
                ax=axs[i],
                ylim=ylim,
            )

            axs[i].axhline(
                y=1-alpha,
                linewidth=1.5,
                color="gray",
                linestyle="dashed",
            )

            axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if i not in [0, 3]:
                axs[i].axes.get_yaxis().set_visible(False)

    elif stochastic_process == "ar1":
        coefficient_list = np.concatenate([np.arange(0, 100, 2) / 100, [0.99, 0.999]]).tolist()

        for i, model in enumerate(models):
            res = []
            for coefficient in coefficient_list:
                df = pd.read_csv(
                    f"{inpath}/{stochastic_process}-" +
                    f"phi_{coefficient}-" +
                    f"{model}-alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
                )
                res.append({
                    "coverage": df["empirical_coverage"].mean(),
                    "coefficient": coefficient,
                })

            df = pd.DataFrame(res)

            df.plot(
                x="coefficient",
                y="coverage",
                lw=1.5,
                c="#000075",
                legend=None,
                title=model_names[model],
                xlabel="Autoregressive coefficient",
                ylabel="Marginal coverage",
                ax=axs[i],
                ylim=ylim,
            )

            axs[i].axhline(
                y=1-alpha,
                linewidth=1.5,
                color="gray",
                linestyle="dashed",
            )

            axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if i not in [0, 3]:
                axs[i].axes.get_yaxis().set_visible(False)

    elif stochastic_process == "renewal":
        parameter_list = np.arange(2, 51).tolist()

        for i, model in enumerate(models):
            res = []
            for parameter in parameter_list:
                df = pd.read_csv(
                    f"{inpath}/{stochastic_process}-" +
                    f"n_{parameter}-" +
                    f"{model}-alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.csv",
                )
                res.append({
                    "coverage": df["empirical_coverage"].mean(),
                    "parameter": parameter,
                })

            df = pd.DataFrame(res)

            df.plot(
                x="parameter",
                y="coverage",
                lw=1.5,
                c="#000075",
                legend=None,
                title=model_names[model],
                xlabel="Base distribution parameter",
                ylabel="Marginal coverage",
                ax=axs[i],
                ylim=ylim,
            )

            axs[i].axhline(
                y=1-alpha,
                linewidth=1.5,
                color="gray",
                linestyle="dashed",
            )

            axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

            if i not in [0, 3]:
                axs[i].axes.get_yaxis().set_visible(False)

    # Export figure
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(
        f"{outpath}/marginal_coverage-{stochastic_process}-" +
        f"alpha_{alpha}-n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}.pdf",
    )


if __name__ == "__main__":
    main(**vars(args))
