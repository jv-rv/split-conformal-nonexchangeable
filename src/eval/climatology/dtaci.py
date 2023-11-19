"""Dynamically-tuned Adaptive Conformal Inference for spatiotemporal climate data."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from src.models.dtaci import dtaci
from src.utils.general import get_dir, get_month_day


def run(
    month: int,
    day: int,
    alpha: float,
    I: int,
    gammas: list[float],
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
        df_test["beta"] = df_test["score"].apply(lambda x: np.mean(x <= df_cal["score"]))
        res.append(df_test)

    df_test = pd.concat(res)

    res = []

    for lat, lon in df_test[["lat", "lon"]].value_counts().index:
        df_test_sub = df_test[(df_test["lat"] == lat) & (df_test["lon"] == lon)].copy()

        alphas = dtaci(
            betas=df_test_sub["beta"].to_numpy(),
            gammas=np.array(gammas),
            alpha=alpha,
            I=I,
        )

        for y, alpha_dtaci in zip(range(2014, 2023), alphas):
            df_test_sub_y = df_test_sub[df_test_sub["target_date"].dt.year == y].copy()
            df_cal = df[df["target_date"].dt.year < y]
            d = np.quantile(df_cal["score"], 1 - alpha_dtaci)
            df_test_sub_y["lower"] = df_test_sub_y["pred"] - d
            df_test_sub_y["upper"] = df_test_sub_y["pred"] + d
            res.append(df_test_sub_y)

    return pd.concat(res)


def main(
    alpha: float,
    I: int,
    gammas: list[float],
    n_jobs: int,
) -> None:
    month_day = get_month_day()

    df = pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(run)(m, d, alpha, I, gammas) for m, d in tqdm(month_day)),
    ).reset_index(drop=True)

    outpath = get_dir("eval/results/climatology")

    df.to_csv(
        f"{outpath}/dtaci-alpha_{alpha}-I_{I}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--I", type=int, default=5)
    parser.add_argument(
        "--gammas", nargs="+", default=[
            0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128,
        ],
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    main(**vars(args))
