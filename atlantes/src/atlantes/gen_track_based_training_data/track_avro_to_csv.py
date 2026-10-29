"""This script reads track avro files then writes the csv to disk
Command to copy the track incremental data from GCS to local disk:

gcloud alpha storage cp -r gs://skylight-data-sky-int-a-wxbc/track/track-incremental/\
track-incremental.2022-*.avro .
Example:
    python3 track_avro_to_csv.py --avro_data_path \
     /data-mount/track_incr_csvs --track_csv_dir \
    /data-mount/track_incr_csvs

# TODO: Add synchronous unit tests
"""

import csv
from functools import partial
from pathlib import Path

import click
import dask
from atlantes.log_utils import get_logger
from dask.distributed import Client
from fastavro import reader
from tqdm import tqdm
import os

logger = get_logger(__name__)


def avro_to_csv(filename: Path, track_csv_dir: str) -> None:
    """reads a track-incremental avro file, joins with vessel data and writes to csv


    Parameters
    ----------
    filename : Path
        track-incremental avro file
    track_csv_dir : str
        path to write csv to

    """
    head = True
    track_csv = f"{track_csv_dir}/{filename.stem}.csv"
    os.makedirs(track_csv_dir, exist_ok=True)
    track_incremental_schema_example = {
        "trackId": "B:413941005:1509494485:2865115:1194818",
        "rec": "2017-11-01T16:43:36Z",
        "send": "2017-11-01T16:43:25Z",
        "type": 19,
        "mmsi": 413941005,
        "nav": None,
        "lon": 106.12316166666666,
        "lat": 29.158505,
        "sog": 0.0,
        "cog": 0.0,
        "hd": 511,
        "rot": 0.0,
        "depth": 184,
        "dist2coast": -835,
        "dist2port": 21583,
    }

    logger.info(f"Reading {filename} and writing to {track_csv}")
    with open(filename, "rb") as fo, open(track_csv, "w+") as outfile:
        csv_writer = csv.writer(outfile)
        avro_reader = reader(fo)
        for emp in tqdm(avro_reader):
            if head:
                csv_writer.writerow(list(track_incremental_schema_example.keys()))
                head = False
            values_list = list(emp.values())
            trackid = values_list[0]
            position = values_list[1]
            geo = values_list[2]
            csv_writer.writerow(
                [
                    trackid,
                    position["rec"],
                    position["send"],
                    position["type"],
                    position["mmsi"],
                    position["nav"],
                    position["lon"],
                    position["lat"],
                    position["sog"],
                    position["cog"],
                    position["hd"],
                    position["rot"],
                    geo["depth"],
                    geo["dist2coast"],
                    geo["dist2port"],
                ]
            )
        logger.info(f"Finished writing {filename} to {track_csv}")


@click.command()
@click.option(
    "--avro_data_path",
    default="track-incremental",
    help="path to track incremental data",
)
@click.option(
    "--track_csv_dir",
    default="track-csv",
    help="path to track incremental data",
)
@click.option(
    "--year",
    default=2023,
    help="year of track incremental data",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="run computation synchronously for debugging",
)
def main(
    avro_data_path: str,
    track_csv_dir: str,
    year: int,
    debug: bool,
) -> None:
    """main function to read avro incremental data and write to csv

    Parameters
    ----------
    avro_data_path : str
        path to track incremental data
    track_csv_dir : str
        output path for csv files
    debug : bool
        flag to run computation synchronously for debugging

    Returns
    -------
    None"""
    all_files = []
    for month in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]:
        logger.info(f"Creating list of files processing month {month}")
        all_files.extend(
            list(Path(avro_data_path).glob(
                f"track-incremental.{year}-{month}-*.avro"))
        )
    avro_to_csv_partial = partial(avro_to_csv, track_csv_dir=track_csv_dir)
    if debug:
        for file in all_files:
            avro_to_csv_partial(file)
    else:
        with Client() as client:
            logger.info(client.dashboard_link)
            results = []
            for file in all_files:
                lazy_csv = dask.delayed(avro_to_csv_partial)(file)
                results.append(lazy_csv)
            dask.compute(*results)
            client.close()
    logger.info("Finished writing all avro files to csv")


if __name__ == "__main__":
    main()
