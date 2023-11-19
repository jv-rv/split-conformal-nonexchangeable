"""Process raw temperature data and export with climatology model."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from xarray import load_dataset

from src.utils.general import get_dir


def main(start_year: int, end_year: int, window: int) -> None:
    inpath = "data/measurements/temp"
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31")
    # Remove Feb 29 to ease analysis as it is only present on leap years
    dates = dates[~((dates.month == 2) & (dates.day == 29))]

    # Load data

    dfs = []

    for date in tqdm(dates, desc="Reading files"):
        dfs.append(
            load_dataset(f"{inpath}/{date.strftime('%Y%m%d')}.nc").to_dataframe(),
        )

    df = pd.concat(dfs)
    df = df.reset_index()

    # Discard grid points without temperature measurements (mostly oceans)
    df = df.dropna(axis=0, how="any")

    # Split data by year
    d = {}

    for year in tqdm(range(start_year, end_year + 1), desc="Splitting data by year"):
        d[year] = df[df["target_date"].dt.year == year].reset_index(drop=True)

    # From 2014 onwards, there are less observations (lat x lon x month x day)
    # Subset based on 2014, which will happen to be our first test year
    d_base = d[2014].copy()
    d_base["m"] = d_base["target_date"].dt.month
    d_base["d"] = d_base["target_date"].dt.day
    idx = d_base.set_index(["lon", "lat", "m", "d"]).index

    for year in tqdm(range(start_year, end_year + 1), desc="Subsetting data"):
        d[year]["m"] = d[year]["target_date"].dt.month
        d[year]["d"] = d[year]["target_date"].dt.day
        d[year] = d[year].set_index(["lon", "lat", "m", "d"])
        d[year] = d[year].loc[idx].reset_index().drop(["m", "d"], axis=1)

    # Ensure longitude and latitude match across years
    for year in range(start_year, end_year + 1):
        assert d[year][["lon", "lat"]].equals(d[start_year][["lon", "lat"]])

    # Make predictions according to climatology model
    for year in tqdm(range(start_year + window, end_year + 1), desc="Running climatology"):
        d[year]["pred"] = np.mean([d[y]["temp"] for y in range(year-window, year)], axis=0)

    # Join the data in a single dataframe
    df_climatology = pd.concat(
        [d[year] for year in range(start_year + window, end_year + 1)],
        axis=0,
    )

    # Get all 365 days in a year
    month_day = sorted(
        set([(month, day) for month, day in zip(dates.month, dates.day)]),
    )

    # Export as csv with one file for each day of year
    outpath = get_dir("data/processed/climatology")
    for month, day in tqdm(month_day, desc="Exporting results"):
        df_climatology[
            (df_climatology["target_date"].dt.month == month) &
            (df_climatology["target_date"].dt.day == day)
        ].to_csv(f"{outpath}/{month}-{day}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start_year", type=int, default=1979)
    parser.add_argument("--end_year", type=int, default=2022)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    main(**vars(args))
