"""Download and extract datasets from HistData."""

from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from histdata import download_hist_data as dl
from histdata.api import Platform as P
from histdata.api import TimeFrame as TF
from tqdm import tqdm

from src.utils.general import get_dir

parser = ArgumentParser()
parser.add_argument("--delete_zip_files", action="store_true")
args = parser.parse_args()


def main(delete_zip_files: bool) -> None:
    # For simplicity, we will use data until the last full year
    last_full_year = datetime.today().year - 1

    # Load page with information about all available pairs and starting dates
    df = pd.read_html(
        "http://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes",
    )

    # Useful information is on first table
    df = df[0]

    # Merge two columns into single one
    df = df.stack().reset_index(drop=True)

    # Separate pair from starting date
    df = df.str.replace(")", "", regex=False).str.split("(", expand=True)
    df.columns = ["pair", "starting_date"]

    # Format columns and sort data alphabetically according to pairs
    df["starting_date"] = pd.to_datetime(df["starting_date"], format="%Y/%B")
    df["pair"] = df["pair"].str.lower().str.replace("/", "", regex=False).str.strip()
    df = df.sort_values("pair")

    # Starting dates are seldom, if ever, on the first business day of january
    # As we aim to work with full years, select following year from starting date
    df["starting_year"] = df["starting_date"].dt.year + 1
    df = df.drop("starting_date", axis=1)

    # Remove pairs that are no longer updated upstream
    df = df.query(
        "pair not in @abandoned",
        local_dict={"abandoned": ["etxeur", "xauaud", "xauchf", "xaueur", "xaugbp"]},
    )

    # Create list to hold all pairs and years to be downloaded
    pair_year = []
    for pair, starting_year in df.to_records(index=False):
        for pair, year in product(
            [pair],
            range(starting_year, last_full_year + 1),
        ):
            pair_year.append((pair, year))

    # Discard incomplete data files before download
    pair_year.remove(("audusd", 2001))  # two months missing

    # Set path where files will be downloaded to
    outpath = get_dir("data/raw")

    # Download data as zip files
    for pair, year in tqdm(pair_year, desc="Downloading files"):
        dl(
            pair=pair,
            year=year,
            platform=P.GENERIC_ASCII,
            time_frame=TF.ONE_MINUTE,
            output_directory=outpath,
            verbose=False,
        )

    # Extract csv files
    for pair, year in tqdm(pair_year, desc="Extracting files"):
        zip_path = f"{outpath}/DAT_ASCII_{pair.upper()}_M1_{year}.zip"
        with ZipFile(zip_path, mode="r") as archive:
            for member in archive.filelist:
                if member.filename.endswith(".csv"):
                    archive.extract(member, outpath)

    # Delete zip files if desired
    if delete_zip_files:
        for pair, year in tqdm(pair_year, desc="Deleting zip files"):
            zip_path = f"{outpath}/DAT_ASCII_{pair.upper()}_M1_{year}.zip"
            Path(zip_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main(**vars(args))
