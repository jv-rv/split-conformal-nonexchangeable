"""Process raw data and export in a convenient format for models."""

from argparse import ArgumentParser

import pandas as pd

from src.utils.general import get_dir


def main(assets: list[str], maxlags: int, year: int) -> None:
    dfs = []

    for i, asset in enumerate(assets):
        df = pd.read_csv(
            f"data/raw/DAT_ASCII_{asset.upper()}_M1_{year}.csv",
            header=None,
            names=[
                "datetime",
                "open_bid_quote",
                "high_bid_quote",
                "low_bid_quote",
                "close_bid_quote",
                "volume",
            ],
            sep=";",
        )
        df = df.drop(["high_bid_quote", "low_bid_quote", "close_bid_quote", "volume"], axis=1)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
        df = df.set_index("datetime")
        # Ensure USD is the quote currency by inverting pairs in which it appears as base currency
        if asset[:3] == "usd":
            df["open_bid_quote"] = 1 / df["open_bid_quote"]
            asset = asset[-3:] + asset[:3]
            assets[i] = asset
        df = df.rename({"open_bid_quote": asset}, axis=1)
        dfs.append(df)
        del df

    # Concatenate all dataframes
    df = pd.DataFrame()
    df = df.join(dfs, how="outer")
    df = df.dropna(axis=0)

    # Sort index and remove duplicated rows
    df = df.sort_index()
    assert df.groupby(df.index).cumcount().max() <= 1  # at most one duplicate
    assert df[df.index.duplicated(keep="first")].equals(df[df.index.duplicated(keep="last")])
    df = df[~df.index.duplicated(keep="first")]

    # Transform prices into returns
    df = df.pct_change().dropna()

    # Create lagged columns
    for col in df.columns:
        for lag in range(1, maxlags+1):
            df[f"{col}_{lag+1}"] = df[col].shift(lag)

    # Drop missing values
    df = df.dropna()

    # Export result as csv file
    outpath = get_dir("data/processed")
    df.to_csv(f"{outpath}/df-{'_'.join(assets)}-{year}-maxlags_{maxlags}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--assets",
        nargs="*",
        default=[],
        help="Assets.",
    )
    parser.add_argument("--maxlags", type=int, default=10)
    parser.add_argument("--year", type=int, default=2021)
    args = parser.parse_args()

    main(**vars(args))
