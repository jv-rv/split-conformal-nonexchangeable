"""Split NexCP for spatiotemporal climate data."""

from argparse import ArgumentParser

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from src.utils.general import get_dir, get_haversine_dist_matrix, get_month_day, weighted_quantile


def run(
    month: int,
    day: int,
    alpha: float,
    decay_time: float,
    decay_space: float,
    dist: pd.DataFrame,
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

        df_test["lower"] = df_test["pred"]
        df_test["upper"] = df_test["pred"]

        for i in df_test.index:
            scores = df_cal["score"].to_numpy()
            lat_lon = tuple(df_test.loc[i, ["lat", "lon"]])

            weights_time = decay_time ** (test_year - df_cal["target_date"].dt.year)
            weights_space = decay_space ** df_cal.set_index(["lat", "lon"]).join(
                dist.loc[lat_lon].rename("dist"),
            ).loc[:, "dist"]
            weights = weights_time.to_numpy() * weights_space.to_numpy()

            d = weighted_quantile(scores, 1 - alpha, weights)
            df_test.loc[i, "lower"] -= d
            df_test.loc[i, "upper"] += d

            res.append(df_test.loc[[i]])

    return pd.concat(res)


def main(
    alpha: float,
    decay_time: float,
    decay_space: float,
    n_jobs: int,
) -> None:
    month_day = get_month_day()
    dist = get_haversine_dist_matrix()

    df = pd.concat(
        Parallel(n_jobs=n_jobs)(
            delayed(run)(m, d, alpha, decay_time, decay_space, dist)
            for m, d in tqdm(month_day)
        ),
    )

    df = df.reset_index(drop=True)

    outpath = get_dir("eval/results/climatology")

    df.to_csv(
        f"{outpath}/nexcp-alpha_{alpha}-" +
        f"decay_time_{decay_time}-decay_space_{decay_space}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--decay_time", type=float, default=0.99)
    parser.add_argument("--decay_space", type=float, default=0.99)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    main(**vars(args))
