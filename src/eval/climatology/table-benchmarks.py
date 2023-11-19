"""Export table with comparison between conformal methods on climatology experiments."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.eval import average_interval_size, empirical_coverage
from src.utils.general import get_dir


def evaluate(
    df: pd.DataFrame,
    method: str,
    hyperparam: int | float | str,
) -> tuple[str, int | float | str, str, float, float, str]:
    """Evaluate results from a method and return coverage and interval metrics."""
    avg_interval = average_interval_size(df["lower"], df["upper"])
    cov_fixed_time = df.groupby(["target_date"]).apply(
        lambda data: empirical_coverage(data["temp"], data["lower"], data["upper"]),
    )
    cov_fixed_space = df.groupby(["lat", "lon"]).apply(
        lambda data: empirical_coverage(data["temp"], data["lower"], data["upper"]),
    )
    assert abs(np.mean(cov_fixed_space) - np.mean(cov_fixed_time)) < 1e-6
    return (
        method,
        hyperparam,
        f"{np.mean(cov_fixed_space):.1%}",
        np.std(cov_fixed_time).round(3),
        np.std(cov_fixed_space).round(3),
        f"{avg_interval:.3f}",
    )


def eval_dtaci(
    I: int,
    alpha: float,
) -> tuple[str, int | float | str, str, float, float, str]:
    """Load DtACI results and call evaluate."""
    return evaluate(
        pd.read_csv(f"eval/results/climatology/dtaci-alpha_{alpha}-I_{I}.csv"),
        method="DtACI",
        hyperparam=I,
    )


def eval_nexcp(
    decay: float,
    alpha: float,
) -> tuple[str, int | float | str, str, float, float, str]:
    """Load Split NexCP results and call evaluate."""
    return evaluate(
        pd.read_csv(
            f"eval/results/climatology/nexcp-alpha_{alpha}-" +
            f"decay_time_{decay}-decay_space_{decay}.csv",
        ),
        method="Split NexCP",
        hyperparam=decay,
    )


def main(alpha: float) -> None:

    # Eval split CP
    for _ in tqdm(range(1), desc="Evaluating split CP"):
        scp = evaluate(
            pd.read_csv(f"eval/results/climatology/splitcp-alpha_{alpha}.csv"),
            method="Split CP",
            hyperparam="None",
        )

    # Eval DtACI
    dtaci = [
        eval_dtaci(I, alpha)
        for I in tqdm(range(1, 10, 2), desc="Evaluating DtACI")
    ]

    # Eval split NexCP
    nexcp = [
        eval_nexcp(decay, alpha)
        for decay in tqdm([0.99, 0.9, 0.7, 0.5, 0.3, 0.1], desc="Evaluating split NexCP")
    ]

    # Export results as LaTeX table and csv file
    table = pd.DataFrame(
        [scp, *nexcp, *dtaci],
        columns=[
            "Method",
            "Hyperparam",
            "Avg Coverage",
            "SD Cov (time)",
            "SD Cov (space)",
            "Avg intervals",
        ],
    )

    outpath = get_dir("eval/tables")

    table.to_latex(
        f"{outpath}/table-benchmarks.tex",
        float_format="%.3f",
        index=False,
    )

    table.to_csv(f"{outpath}/table-benchmarks.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    main(**vars(args))
