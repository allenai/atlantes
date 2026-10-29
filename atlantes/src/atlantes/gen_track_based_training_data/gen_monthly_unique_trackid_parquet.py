""""""

import os
from pathlib import Path

import click
import dask
import dask.dataframe as dd
from atlantes.datautils import MONTHS_TO_LOAD
from atlantes.log_utils import get_logger
from dask.diagnostics import ProgressBar
from tqdm import tqdm

logger = get_logger(__name__)


@click.command()
@click.option(
    "--path-to-input-csvs",
    multiple=False,
    type=str,
    required=False,
    help="Parent directory containing csvs that will be read into a geodataframe for building the training data.",
)
@click.option(
    "--output-training-dir-root",
    is_flag=False,
    type=str,
    default=os.getcwd(),
    help="Root directory to output training data. Default is None, which will output to the current working directory.",
)
@click.option(
    "--start-month",
    type=str,
    required=False,
    default="0",
    help="Month to start reading data from, zero index",
)
@click.option(
    "--end-month",
    type=str,
    required=False,
    default="12",
    help="Month to stop reading data from zero index",
)
@click.option("--year", type=int, default=2022, help="year for dataset")
@click.option(
    "--temp-dir",
    type=str,
    required=False,
    default="/data-mount",
    help="Temporary directory to use for dask storage leakage.",
)
def cli(
    path_to_input_csvs: str,
    output_training_dir_root: str,
    start_month: str,
    end_month: str,
    year: int,
    temp_dir: str,
) -> None:
    """Creates a parquet file for each month of unique trackIds from the track_incremental csvs.

    Parameters
    ----------
    path_to_input_csvs : str
        Parent directory containing csvs that will be read into a geodataframe for building the training data.
    output_training_dir_root : str
        Root directory to output training data. Default is None, which will output to the current working directory.
    start_month : str
        Month to start reading data from, zero index
    end_month : str
        Month to stop reading data from zero index
    temp_dir : str
        Temporary directory to use for dask storage leakage.
    year: int
        year for building dataset
    -------
    None
        Writes training data to output_training_dir_root."""

    with dask.config.set({"temporary_directory": temp_dir}), dask.config.set(
        scheduler="processes"
    ):
        months_requested = MONTHS_TO_LOAD[int(start_month) : int(end_month)]
        for month in tqdm(
            months_requested, desc="Months ais data written to track csv files"
        ):
            logger.info(f"reading: track-incremental.{year}-{month}-*.csv")
            with ProgressBar():
                csv_wildcard_path = (
                    Path(path_to_input_csvs) / f"track-incremental.{year}-{month}-*.csv"
                )
                gdf = dd.read_csv(
                    csv_wildcard_path, usecols=["trackId"], assume_missing=True
                )

                logger.info(f"grouping by trajectory for month {month}")
                # aggregate unique
                gdf.drop_duplicates().to_parquet(
                    Path(output_training_dir_root) / f"trackIds_{year}_{month}.parquet",
                    engine="pyarrow",
                    compression="snappy",
                )
                logger.info(f"finished writing for {month=}")

        logger.info("Computation complete! Stopping workers...")

if __name__ == "__main__":
    cli()
