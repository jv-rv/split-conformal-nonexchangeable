"""Download temperature data."""

import argparse
import datetime
import pathlib
import sys
import time

import pandas as pd
import requests
import xarray as xr

DATETIME_FORMAT = "%Y%m%d"

IRI_GRID_NAMES = {
    "X": "lon",
    "Y": "lat",
    "S": "issuance_date",
    "M": "model_run",
    "LA": "lead",
    "L": "lead",
    "hdate": "hindcast_date",
    "T": "target_date",
}


def string_to_dt(string: str) -> datetime.datetime:
    """Transform string to datetime."""
    return datetime.datetime.strptime(string, DATETIME_FORMAT)


def dt_to_string(dt: datetime.datetime) -> str:
    """Transform datetime to string."""
    return datetime.datetime.strftime(dt, DATETIME_FORMAT)


def is_valid_measurement_date(
    measurement: str,
    measurement_date: datetime.datetime,
) -> bool:
    """Verify if a measurement date is valid."""
    assert isinstance(measurement_date, datetime.datetime)

    try:
        valid_measurement_dates = [
            string_to_dt("19790101") + datetime.timedelta(days=x)
            for x in range(
                0, int((datetime.datetime.today() - string_to_dt("19790101")).days / 1) + 1,
            )
        ]
        return measurement_date in valid_measurement_dates
    except KeyError:
        return False


def download_url(
    url: str,
    timeout: int = 600,
    retry: int = 3,
    cookies: dict[str, str] = {},
) -> requests.models.Response | None:
    """Download URL, waiting some time between retries."""
    r = None
    for i in range(retry):
        try:
            r = requests.get(url, timeout=timeout, cookies=cookies)
            return r
        except requests.exceptions.Timeout as e:
            # Wait until making another request
            if i == retry - 1:
                raise e
            print(f"Request to url {url} has timed out. Trying again...")
            time.sleep(3)
    print(f"Failed to retrieve file after {retry} attempts. Stopping...")
    return None


def get_dates(date_str: str) -> list[datetime.datetime]:
    """Output the list of dates corresponding to input date string."""
    if "-" in date_str:
        # Input is of the form '20170101-20180130'
        first_date_str, last_date_str = date_str.split("-")
        first_date = string_to_dt(first_date_str)
        last_date = string_to_dt(last_date_str)
        dates = [
            first_date + datetime.timedelta(days=x)
            for x in range(0, (last_date - first_date).days + 1)
        ]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 4:
        # Input '2017' is expanded to 20170101-20171231
        year = int(date_str)
        first_date = datetime.datetime(year=year, month=1, day=1)
        last_date = datetime.datetime(year=year, month=12, day=31)
        dates = [
            first_date + datetime.timedelta(days=x)
            for x in range(0, (last_date - first_date).days + 1)
        ]
        return dates
    elif len(date_str) == 6:
        # Input '201701' is expanded to 20170101-20170131
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime.datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime.datetime(year=year + 1, month=1, day=1)
        else:
            last_date = datetime.datetime(year=year, month=month + 1, day=1)
        dates = [
            first_date + datetime.timedelta(days=x)
            for x in range(0, (last_date - first_date).days)
        ]
        return dates
    elif len(date_str) == 8:
        # Input '20170101' is a date
        dates = [datetime.datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    else:
        raise NotImplementedError(
            "Date string provided cannot be transformed into list of target dates.",
        )


def get_folder(
    folder_path: str | pathlib.Path,
    verbose: bool = True,
) -> pathlib.Path:
    """Create folder, if it doesn't exist, and return folder path.

    Args:
        folder_path: Folder path, either existing or to be created.

    Returns:
        folder path.
    """
    folder_path = pathlib.Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path


def load_s2s_reforecast_dataset(file_path: pathlib.Path) -> xr.Dataset:
    """Open an s2s reforecast nc file.

    It is not possible to use xr.load_dataset() because the encoding
    for the hindcast date (hdate) is given as months since 1960-01-01,
    which is not standard for xarray.
    """

    ds = xr.load_dataset(file_path, decode_times=False)
    ds[IRI_GRID_NAMES["S"]] = pd.to_datetime(
        ds[IRI_GRID_NAMES["S"]].values, unit="D", origin=pd.Timestamp("1960-01-01"),
    )

    model_issuance_day = ds[f"{IRI_GRID_NAMES['S']}.day"].values[0]
    model_issuance_month = ds[f"{IRI_GRID_NAMES['S']}.month"].values[0]
    model_issuance_date_in_1960 = pd.Timestamp(f"1960-{model_issuance_month}-{model_issuance_day}")
    # While hdates refer to years (for which the model issued in ds["S"] is initialized),
    # its values are given as months until the middle of the year, so 6 months are subtracted
    # to yield the beginning of the year.
    ds[IRI_GRID_NAMES["hdate"]] = pd.to_datetime([
        model_issuance_date_in_1960 + pd.DateOffset(months=x - 6)
        for x in ds[IRI_GRID_NAMES["hdate"]].values
    ])

    return ds


def df_contains_nas(
    file_path: pathlib.Path,
    column_name: str,
    how: str = "any",
) -> bool:
    """Check if there are missing values in dataframe."""
    try:
        df = xr.load_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = load_s2s_reforecast_dataset(file_path).to_dataframe().reset_index()
    nas_in_column = df.isna()[column_name]
    if how == "all":
        nas_flag = nas_in_column.all().item()
    elif how == "any":
        nas_flag = nas_in_column.any().item()
    else:
        raise NotImplementedError("Argument 'how' must receive 'any' or 'all'.")
    return bool(nas_flag)


def df_contains_multiple_dates(
    file_path: pathlib.Path,
    time_col: str = IRI_GRID_NAMES["S"],
) -> bool:
    """Check if there are multiple dates in dataframe."""
    try:
        df = xr.load_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = load_s2s_reforecast_dataset(file_path).to_dataframe().reset_index()
    return len(df[time_col].unique()) > 1


def main(
    target_dates: str,
    skip_existing: bool,
    check_file_integrity: bool,
) -> None:
    # For each date, average or sum over this number subsequent of days.
    ACCUMULATE_TIME_PERIOD = 14

    longitudes = ["0", "358.5"]
    latitudes = ["-90.0", "90.0"]
    grid_size = "1.5"
    convert_longitude_to_usual_grid_url = "X/360/shiftGRID/"
    restrict_latitudes_url = f"Y/{latitudes[0]}/{grid_size}/{latitudes[1]}/GRID/"
    restrict_longitudes_url = f"X/{longitudes[0]}/{grid_size}/{longitudes[1]}/GRID/"

    average_max_min_temp_url = "a%3A/.tmax/%3Aa%3A/.tmin/%3Aa/add/2/div/"
    accumulate_time_url = f"T/{ACCUMULATE_TIME_PERIOD}/0.0/runningAverage/"
    recenter_time_url = f"T/-{int(ACCUMULATE_TIME_PERIOD/2)}/shiftGRID/"

    rename_grids_url = (
        f"X/({IRI_GRID_NAMES['X']})/renameGRID/"
        f"Y/({IRI_GRID_NAMES['Y']})/renameGRID/"
        f"T/({IRI_GRID_NAMES['T']})/renameGRID/"
    )

    output_folder = get_folder("data/measurements/temp")
    print(f"Output folder: {output_folder}")

    for target_date in get_dates(target_dates):

        file_path = output_folder / f"{dt_to_string(target_date)}.nc"

        day, month, year = datetime.datetime.strftime(target_date, "%d,%b,%Y").split(",")

        if target_date < string_to_dt("19790101"):
            print(
                f"WARNING: Skipping: {day} {month} {year}" +
                " (date is prior to earliest measurement available).",
            )
            continue
        elif not is_valid_measurement_date("temp", target_date):
            print(f"WARNING: Skipping: {day} {month} {year} (not a valid temp date).")
            continue

        if file_path.exists() and skip_existing:
            print(f"INFO: Skipping: {day} {month} {year} (file already exists).\n")
            continue

        date_period_end = datetime.datetime.strftime(
            target_date + datetime.timedelta(days=ACCUMULATE_TIME_PERIOD - 1), "%d,%b,%Y",
        )
        day_period_end, month_period_end, year_period_end = date_period_end.split(",")

        restrict_date_url = (
            f"T/({day}%20{month}%20{year}%20-%20{day_period_end}%20" +
            f"{month_period_end}%20{year_period_end})/VALUES/"
        )
        t = time.time()
        URL = (
            f"http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.temperature/.daily/"
            f"{average_max_min_temp_url}"
            f"{accumulate_time_url}"
            f"{restrict_date_url}"
            f"{recenter_time_url}"
            f"{convert_longitude_to_usual_grid_url}"
            f"{restrict_longitudes_url}"
            f"{restrict_latitudes_url}"
            f"{rename_grids_url}"
            f"data.nc"
        )

        r = download_url(URL)

        if r is None:
            print("ERROR: Could not retrieve files.")

        elif r.status_code == 200 and r.headers["Content-Type"] == "application/x-netcdf":
            print(f"Downloading: {day} {month} {year}.")
            with open(file_path, "wb") as f:
                f.write(r.content)

            ds = xr.load_dataset(file_path)
            ds = ds.rename({"asum": "temp"})
            ds = ds.assign_coords(
                target_date=pd.to_datetime(
                    ds.target_date.values, unit="D",
                    origin="julian",
                ),
            )
            ds.to_netcdf(file_path)

            print(
                f"-done (downloaded {sys.getsizeof(r.content)/1024:.2f} KB in" +
                f" {time.time() - t:.2f}s).\n",
            )

            if check_file_integrity:
                if df_contains_nas(file_path, "temp", how="any"):
                    print(f"WARNING: {day} {month} {year} contains nas in weather variable.")
                if df_contains_multiple_dates(file_path, time_col=IRI_GRID_NAMES["T"]):
                    print(f"WARNING: {day} {month} {year} file contains multiple forecast dates.")

        elif r.status_code == 404:
            print(f"WARNING: Data for {day} {month} {year} is not available for temperature.\n")

        else:
            print(
                "ERROR: Unknown error occurred when trying to download data for" +
                f" {day} {month} {year} for temperature.\n",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dates",
        "-d",
        default="20200101",
        help=(
            "Dates to download data over; format can be"
            " '20200101-20200304', '2020', '202001', '20200104'"
        ),
    )
    parser.add_argument(
        "--check_file_integrity",
        "-cfi",
        action="store_true",
        help="Perform basic checks on downloaded file",
    )
    parser.add_argument(
        "--skip_existing",
        "-se",
        action="store_true",
        help="If true, skips downloading data if resulting file already exists",
    )
    args = parser.parse_args()

    main(**vars(args))
