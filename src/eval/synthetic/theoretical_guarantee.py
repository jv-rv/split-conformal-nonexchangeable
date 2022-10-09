"""Calculate theoretical bound for distinct settings."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils.dependence import minimize_eta_test_point
from src.utils.general import get_dir

parser = ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--delta", type=float)
parser.add_argument("--pq", type=float)
parser.add_argument("--n_jobs", type=int, default=1)
args = parser.parse_args()


def run(
    n_cal: int,
    alpha: float,
    delta: float,
    pq: float,
) -> dict[str, int | float]:
    """Calculate eta for a single experiment."""
    bound, eta, a, mu, r = minimize_eta_test_point(
        n_cal=n_cal,
        alpha=alpha,
        delta_cal=delta,
        p=pq,
        q=pq,
        bound="bernstein",
        stochastic_process="two_state_markov_chain",
    )

    return {
        "bound": bound,
        "n_cal": n_cal,
        "pq": pq,
        "eta": eta,
        "a": a,
        "mu": mu,
        "r": r,
        "delta": delta,
    }

def main(
    alpha: float,
    delta: float,
    pq: float,
    n_jobs: int,
) -> None:
    n_cal_list = np.arange(50000, 400, -100, dtype=int).tolist()

    res = Parallel(n_jobs=n_jobs)(
        delayed(run)(n_cal, alpha, delta, pq) for n_cal in tqdm(n_cal_list)
    )

    df = pd.DataFrame(res)

    outpath = get_dir("eval/results/synthetic/theoretical_guarantee")

    df.to_csv(f"{outpath}/alpha_{alpha}-delta_{delta}-pq_{pq}.csv")


if __name__ == "__main__":
    main(**vars(args))
