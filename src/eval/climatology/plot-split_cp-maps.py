"""Plot heatmaps with actual temperature on a given day and split cp's lower and upper bounds."""

from argparse import ArgumentParser

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.general import get_dir


def main(
    alpha: float,
    date: str,
) -> None:
    matplotlib.rcParams.update({"font.size": 10})

    df = pd.read_csv(
        f"eval/results/climatology/splitcp-alpha_{alpha}.csv",
        parse_dates=["target_date"],
        index_col=["lat", "lon"],
    )

    df = df[df["target_date"] == date]

    map_projection = ccrs.PlateCarree()
    maps = ["lower", "temp", "upper"]

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        subplot_kw={"projection": map_projection},
        figsize=(14, 2.5),
    )

    for map, ax in zip(maps, axs):
        df[map].to_xarray().plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=-50,
            vmax=40,
            cmap="coolwarm",
            add_colorbar=False,
        )
        ax.coastlines()

    axs[0].set_title("Lower prediction")
    axs[1].set_title("Target")
    axs[2].set_title("Upper prediction")

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()

    outpath = get_dir("eval/plots")
    plt.savefig(
        f"{outpath}/splitcp-climatology-pred_intervals-{date.replace('-','')}.png",
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--date", type=str, default="2022-12-31")
    args = parser.parse_args()

    main(**vars(args))
