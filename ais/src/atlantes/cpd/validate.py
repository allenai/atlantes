"""
This script is used for visualizing specific aspects of Automatic Identification System (AIS) tracks.

Given a directory of AIS track files in CSV format and a CSV file containing 'changepoints', this script will produce
visualizations of both the AIS tracks and the change points within those tracks. The change points are represented as
red dots on the graphs produced by the script. The script also includes the option to limit the number of tracks that
are visualized.

The script includes three main functions:

1. `save_track_changepoint_visuals` generates and saves a scatter plot of a track, with the track's changepoints marked
in red. It also generates and saves a timeseries graph of the changepoints over time.


2. `main` serves as the entry point for the script. It handles the script's command-line options, reads in the input
CSVs, and calls the other functions to generate the visualizations.

The script can be executed from the command line with options to specify the parent directory of the AIS track CSV files,
an output folder for the visualizations, and the number of tracks to visualize.

For now if you wnat to visualize some tracks use gsutil to copy it locally

Command-line options:

    --ais-track-parent-dir    Parent directory containing csvs of individual ais tracks
    --output_folder           Output folder (defaults to current directory)
    --num_tracks              Number of tracks to visualize (defaults to 100)

Example usage:
    python3 validate.py --ais-track-parent-dir /data/ais/ais_tracks/ --changepoint-csv-path /data/ais/changepoints.csv --output_folder /data/ais/visualizations/ --num_tracks 200
# TODO: Also plot changepoints with respect to time
"""

import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


def get_changepoint_binary(track_df: pd.Series) -> np.ndarray:
    """gets binary array of changepoints from a track"""
    # Check that there is a subpath 0
    if track_df["subpath_num"].iloc[0] != 0:
        raise ValueError(f"First subpath is not 0 in trajectory {track_df}")
    cpoint_binary = track_df["subpath_num"].diff(-1).fillna(-1.0).reset_index(drop=True)
    if len(cpoint_binary) == 0:
        raise ValueError(f"No changepoints in trajectory {track_df}")
    return np.array(1 * [cpoint_binary <= -1])


def save_track_changepoint_visuals(track_df: pd.DataFrame, output_folder: str) -> None:
    """
    Generates and saves a visualization of a single AIS track and its changepoints.

    Parameters
    ----------
    track_df : pd.DataFrame
        Dataframe of a single AIS track
    output_folder : str
        Path to output folder
    """
    trackId = track_df.loc[0, "trackId"]
    changepoints = get_changepoint_binary(track_df)

    # Graph the track with changepoints in red
    plt.scatter(track_df["lon"], track_df["lat"], c=track_df["sog"])
    plt.colorbar()
    plt.plot(track_df["lon"], track_df["lat"])
    plt.plot(
        track_df["lon"][changepoints == 1], track_df["lat"][changepoints == 1], "ro"
    )
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.savefig(os.path.join(output_folder, f"changepoints_{trackId}.jpeg"))
    plt.close()

    # Graph the Changepoints as a timeseries fo sog
    plt.plot(track_df["sog"])
    plt.plot(np.where(changepoints == 1)[0], track_df["sog"][changepoints == 1], "ro")
    plt.savefig(os.path.join(output_folder, f"cp_sog_{trackId}.jpeg"))
    plt.close()
    return


@click.command()
@click.option(
    "--ais-track-parent-dir",
    type=str,
    required=True,
    help="Parent directory containing csvs of individual ais tracks",
)
@click.option("--output_folder", default="./", help="Output folder")
@click.option(
    "--num_tracks", type=int, default=100, help="Number of tracks to visualize"
)
def main(
    ais_track_parent_dir: str,
    output_folder: str,
    num_tracks: int,
) -> None:
    """
    Main function for validating AIS tracks.

    Parameters
    ----------
        ais_track_parent_dir (str): Path to the parent directory containing AIS track CSV files.
        output_folder (str): Path to the folder where the output will be saved.
        num_tracks (int): Number of tracks to validate.

    Returns
    -------
        None
    """
    data_paths = np.array([x for x in Path(ais_track_parent_dir).rglob("*.csv")])
    val_data_paths = np.random.choice(data_paths, num_tracks)
    logger.info("Reading CSVs")
    val_tracks = [pd.read_csv(x) for x in val_data_paths]
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    for track in val_tracks:
        logger.info(f"Visualizing tracks: track {track.loc[0, 'trackId']}")
        save_track_changepoint_visuals(track, output_folder)

    logger.info("Change points Visualized!")


if __name__ == "__main__":
    main()
    logger.info("Done")
