"""Create track files from NOAA AIS coastal data that contain data from a single trajectory.
This script reads daily AIS data files and combines them into track files, where each file
contains all messages from a single vessel (MMSI) across the entire time period.
"""

from pathlib import Path

import click
import dask.dataframe as dd
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.log_utils import get_logger
from dask.diagnostics import ProgressBar
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)


def process_entity_group(
    group: DataFrame[TrackfileDataModelTrain], output_dir: Path, mmsi: str
) -> None:
    """Process and save data for a single entity.
    Parameters
    ----------
    group : DataFrame
        DataFrame containing all messages for a single MMSI
    output_dir : Path
        Directory where the track files will be saved
    """
    output_path = output_dir / f"track_mmsi_{mmsi}.csv"
    logger.info(f"Writing track file for MMSI {mmsi} to {output_path}")
    group.to_csv(output_path, index=False)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """Create track files from daily AIS data files.
    Parameters
    ----------
    input_dir : str
        Directory containing the daily converted AIS data files
    output_dir : str
        Directory where the track files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get list of all CSV files
    csv_files = list(input_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} input files")

    # Read all CSV files using Dask
    logger.info("Reading files with Dask...")
    with ProgressBar():
        df = dd.read_csv(csv_files).compute()

    # Group by MMSI and process each group
    logger.info("Creating track files...")
    # Convert to pandas for the final groupby operation since we need to write
    # individual files
    for mmsi, group in tqdm(df.groupby("mmsi"), desc="Creating track files"):
        process_entity_group(group, output_path, mmsi)


if __name__ == "__main__":
    main()