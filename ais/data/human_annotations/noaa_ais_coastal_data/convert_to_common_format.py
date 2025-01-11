"""Convert Daily NOAA AIS coastal data to common format aligning with the rest of the codebase"""

import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd
import pandera as pa
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.datautils import NAV_NAN, VES_NAN
from atlantes.log_utils import get_logger
from atlantes.utils import get_flag_code_from_mmsi
from pandera.typing import DataFrame

logger = get_logger(__name__)

NOAA_COLUMN_NAMES = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "VesselName",
    "IMO",
    "CallSign",
    "VesselType",
    "Status",
    "Length",
    "Width",
    "Draft",
    "Cargo",
    "TransceiverClass",
]
COLUMN_MAPPING = {
    "MMSI": "mmsi",
    "BaseDateTime": "send",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog",
    "COG": "cog",
    "VesselName": "name",
    "VesselType": "category",
    "Status": "nav",
}


def process_file(csv_file: Path, output_dir: Path) -> None:
    logger.info(f"Converting {csv_file}")
    df = convert_to_common_format(str(csv_file))
    output_path = output_dir / f"convert_{csv_file.name}"
    logger.info(f"Writing to {output_path}")
    df.to_csv(output_path, index=False)


@pa.check_types
def convert_to_common_format(file_path: str) -> DataFrame[TrackfileDataModelTrain]:
    """Convert a single daily NOAA AIS coastal data file to common format"""
    df = pd.read_csv(file_path)
    # print(df.head())
    # print(df.columns)
    df = df.rename(columns=COLUMN_MAPPING)
    df.category = df.category.fillna(VES_NAN)
    df.category = df.category.astype(int)
    df.nav = df.nav.fillna(NAV_NAN)
    df.nav = df.nav.astype(int)
    df.loc[:, "dist2coast"] = 3000
    df.loc[:, "flag_code"] = df.mmsi.apply(get_flag_code_from_mmsi)
    # Assuming all data from same mmsi is actually the same vessel and part of a coherent track
    df.loc[:, "trackId"] = "B:" + df.mmsi.astype(str)
    df.loc[:, "name"] = df.name.fillna("Unknown")  # Not sure what we do elsewhere
    return df


if __name__ == "__main__":

    root_dir = "/Users/henryh/Desktop/NOAA_AIS_DATA"
    output_dir = Path(root_dir) / "converted_daily_ais"
    output_dir.mkdir(exist_ok=True)

    # Get list of all CSV files
    csv_files = list(Path(root_dir).glob("*.csv"))

    # Create pool of workers
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)

    # Process files in parallel
    process_func = partial(process_file, output_dir=output_dir)
    pool.map(process_func, csv_files)

    pool.close()
    pool.join()