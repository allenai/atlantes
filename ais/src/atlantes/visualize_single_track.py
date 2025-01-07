""" Script for visualizing a single track, tool to use when looking at validation data, if want to see the trajectory on a map
python3 visualize_single_track.py --path-to-track
# TODO add ability to plot with labels
"""

import datetime

import click
import contextily
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atlantes.log_utils import get_logger
from matplotlib.widgets import SpanSelector

logger = get_logger(__name__)


def plot_ais_track(
    df: pd.DataFrame,
    zoom_padding: float = 4,
) -> None:
    """
    Plot a geospatial and time-series visualization from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns 'unixDateTime', 'lon', 'lat', 'sog', and 'send'.
    zoom_padding : float, optional
        Padding to add for zooming in the geospatial plot. Default is 4.

    Returns:
    --------
    None
    """
    plt.ion()

    df["send"] = pd.to_datetime(df["send"])
    df["unixDateTime"] = df.apply(
        lambda x: int(datetime.datetime.timestamp(x["send"])), axis=1
    )
    df["approxLocalTime"] = df.apply(
        lambda x: x["send"] + datetime.timedelta(0, x["lon"] * 4 * 60), axis=1
    )
    df["localHour"] = df.apply(lambda x: x["approxLocalTime"].hour, axis=1)
    df["noonDiff"] = df.apply(lambda x: abs(x["localHour"] - 12), axis=1)

    plotDF = df.sort_values(by="unixDateTime", ascending=True)

    x = plotDF["lon"].to_numpy()
    y = plotDF["lat"].to_numpy()
    dttm = plotDF["unixDateTime"].to_numpy()
    sog = plotDF["sog"].to_numpy()
    local = plotDF["send"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 3]}
    )

    ax1.plot(local, sog, color="black", linewidth=2, alpha=0.1)
    ax1.scatter(local, sog, c=sog, cmap="twilight_shifted", s=2)

    ax1.set_xlabel("unix timestamp")
    ax1.set_ylabel("speed")
    ax1.set_title(
        "Press left mouse button and drag to select a region in the top graph"
    )

    ax2.set_xlim(x.min() - zoom_padding, x.max() + zoom_padding)
    ax2.set_ylim(y.min() - zoom_padding, y.max() + zoom_padding)

    ax2.scatter(x, y, c=sog, cmap="turbo", s=10)

    try:
        contextily.add_basemap(
            ax2, crs=4326, source=contextily.providers.Esri.OceanBasemap
        )
    except Exception as e:
        raise e

    def onselect(xmin: int, xmax: int) -> None:
        indmin, indmax = np.searchsorted(dttm, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        region_x = x[indmin:indmax]
        region_y = y[indmin:indmax]

        if len(region_x) >= 2:
            line2.set_data(region_x, region_y)
            fig.canvas.draw_idle()

    (line2,) = ax2.plot([], [], color="lime", alpha=0.5, linewidth=10)

    SpanSelector(
        ax1,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:blue"),
        interactive=True,
        drag_from_anywhere=True,
    )

    plt.show(block=True)


@click.command()
@click.option("-p", "--path-to-track", type=str, help="Path to track csv file")
@click.option(
    "--zoom-padding",
    type=float,
    default=4,
    help="Padding to add for zooming in the geospatial plot",
)
def plot_ais_track_cli(
    path_to_track: str,
    zoom_padding: float,
) -> None:
    """CLI wrapper for plot_ais_track function."""
    columns = ["lat", "lon", "sog", "send"]
    if path_to_track.endswith(".parquet"):
        df = pd.read_parquet(path_to_track, columns=columns)
    elif path_to_track.endswith(".csv"):
        df = pd.read_csv(path_to_track, usecols=columns)
    else:
        raise ValueError("File type not supported")

    plot_ais_track(df, zoom_padding)


if __name__ == "__main__":
    plot_ais_track_cli()
