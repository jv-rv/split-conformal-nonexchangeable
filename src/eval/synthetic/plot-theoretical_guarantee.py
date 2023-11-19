"""Plot theoretical bound sensitivity to calibration set size and dependence factor."""

# Results for two-state markov chain

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.general import get_dir


def main(
    alpha: float,
    delta: float,
    palette: str,
) -> None:
    inpath = "eval/results/synthetic/theoretical_guarantee"

    dfs = []

    for pq in [0.5, 0.45, 0.4, 0.35, 0.3]:
        df = pd.read_csv(
            f"{inpath}/alpha_{alpha}-delta_{delta}-pq_{pq}.csv",
            usecols=["bound", "n_cal", "pq"],
        )
        df["1mpq"] = 1 - df["pq"]
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)

    df["1mpq"] = df["1mpq"].replace({0.5: "0.5 (indep.)"})

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.lineplot(
        data=df,
        x="n_cal",
        y="bound",
        hue="1mpq",
        palette=palette,
    )

    plt.xlabel("Calibration set size")
    plt.ylabel("Coverage guarantee")
    plt.legend(title="Dependence factor")
    plt.axhline(1-alpha, linewidth=1, linestyle="dashed", color="black")

    plt.tight_layout()

    outpath = get_dir("eval/plots")

    plt.savefig(f"{outpath}/theoretical_guarantee-alpha_{alpha}-delta_{delta}.pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--palette", type=str, default="flare_r")
    args = parser.parse_args()

    main(**vars(args))
