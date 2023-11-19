"""Split CP for spatiotemporal climate data."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from src.utils.general import get_dir, get_month_day


def run(
    month: int,
    day: int,
    alpha: float,
) -> pd.DataFrame:
    """Load data for a fixed month-day and generate prediction intervals."""
    df = pd.read_csv(
        f"data/processed/climatology/{month}-{day}.csv",
        parse_dates=["target_date"],
    )

    df["score"] = (df["temp"] - df["pred"]).abs()

    res = []

    for test_year in range(2014, 2023):
        df_cal = df[df["target_date"].dt.year < test_year]
        df_test = df[df["target_date"].dt.year == test_year].copy()
        d = np.quantile(df_cal["score"], 1 - alpha)
        df_test["lower"] = df_test["pred"] - d
        df_test["upper"] = df_test["pred"] + d
        res.append(df_test)

    return pd.concat(res)


def main(alpha: float, n_jobs: int) -> None:
    month_day = get_month_day()

    df = pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(run)(m, d, alpha) for m, d in tqdm(month_day)),
    ).reset_index(drop=True)

    outpath = get_dir("eval/results/climatology")

    df.to_csv(
        f"{outpath}/splitcp-alpha_{alpha}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    main(**vars(args))
