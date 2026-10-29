"""Converts track-incremental csvs to daily parquet files for each unique trackId.


This script reads track-incremental csvs, filters to unique trackIds, and writes daily parquet files for each unique trackId.

Example
-------
To run the script, pass the path to the parent directory containing the track-incremental csvs, the path to the vessel metadata file, and the root directory to output the daily parquet files:

    $ python track_incremental_to_daily_trackid_parquet.py --path-to-input-csvs /path/to/track-incremental-csvs --path-to-vessel-metadata /path/to/vessel-metadata.csv --output-training-dir-root /path/to/output-dir

"""

import os
from datetime import datetime
from pathlib import Path

import click
import dask
import dask.dataframe as dd
import pandas as pd
import spatialpandas as sp
from atlantes.datautils import DFCOLS, DFCOLS_NO_META, MONTHS_TO_LOAD
from atlantes.log_utils import get_logger
from dask.distributed import Client
from holoviews.util.transform import lon_lat_to_easting_northing as ll2en
from tqdm import tqdm

logger = get_logger(__name__)

DT_STRING = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

DF_META_COLS = list(set(DFCOLS) - set(DFCOLS_NO_META))
config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
DF_META_COLS = list(set(DFCOLS) - set(DFCOLS_NO_META))

NAV_NAN = 9999
VES_NAN = 9999


def read_vessel_dict(vessel_path: str) -> pd.DataFrame:
    """reads vessel data dictionary into memory

    Parameters
    ----------
    vessel_path : str, optional
        all vessel data (static AIS messages) associated with incremental data,
        by default VESSEL_DICT_PATH

    Returns
    -------
    pd.DataFrame
        vessel data dictionary dataframe
    """
    vessel_df = pd.read_csv(vessel_path).set_index("unique_id", drop=True)
    vessel_df = vessel_df[~vessel_df.index.duplicated(keep="first")]

    return vessel_df


def get_vessel_metadata(metadata: pd.DataFrame, track_id: str) -> list:
    """gets metadata associated with a track id, if any

    Parameters
    ----------
    metadata : pd.DataFrame
        dataframe containing metadata
    track_id : str
        track id to get metadata for

    Returns
    -------
    list
        metadata associated with a track id"""
    num_columns = len(metadata.columns)
    if track_id in metadata.index:
        metadata = metadata.loc[track_id].values.tolist()
    else:
        metadata = [None] * num_columns
    return metadata


def add_metadata_columns_to_df(
    df: pd.DataFrame, metadata: list, metadata_cols: list
) -> pd.DataFrame:
    """adds metadata columns to a dataframe

    Parameters
    ----------
    metadata : list
        list of metadata associated with a track id
    df : pd.DataFrame
        dataframe to add metadata columns to"""

    # Ensure order of metadata is the same as the order of the columns in DF_META_COLS
    metadata_dict = dict(zip(metadata_cols, metadata))
    for col_name in DF_META_COLS:
        metadata = metadata_dict[col_name]
        df.loc[:, col_name] = metadata
    return df


def convert_to_geo_df(df: pd.DataFrame) -> sp.GeoDataFrame:
    """converts a partition of a Dask DataFrame to a spatialpandas DataFrame"""
    east, north = ll2en(df.lon.astype("float32"), df.lat.astype("float32"))
    try:
        df = sp.GeoDataFrame(
            {
                "geometry": sp.geometry.PointArray((east, north)),
                "mmsi": df.mmsi.fillna(0).astype("int32"),
                "category": df["ais_type"].fillna(VES_NAN).astype("int32"),
                "nav": df.nav.fillna(NAV_NAN).astype("int32"),
                "trackId": df.trackId.fillna("none").astype("str"),
                "name": df["name"].fillna("none").astype("str"),
                "send": df.send.fillna("none").astype("str"),
                "lon": df.lon.fillna(0).astype("float32"),
                "lat": df.lat.fillna(0).astype("float32"),
                "sog": df.sog.fillna(0).astype("float32"),
                "cog": df.cog.fillna(0).astype("float32"),
                "flag_code": df.flag_code.fillna("none").astype("str"),
                "vessel_class": df.vessel_class.fillna("none").astype("str"),
                "dist2coast": df.dist2coast.fillna(0).astype("float32"),
            }
        )
    except Exception as e:
        logger.error(f"Error converting to spatialpandas dataframe: {e}")
        raise ValueError(f"Error converting to spatialpandas dataframe  {df.head()}")
    return df


def filter_to_track_id(df: pd.DataFrame, track_id: str) -> pd.DataFrame:
    """filters a dataframe to a specific track id"""
    return df[df["trackId"] == track_id]


def add_metadata_convert_and_write_to_parquet(
    df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: os.PathLike
) -> None:
    """adds metadata columns to a dataframe, converts to a spatialpandas dataframe, and writes to a csv"""
    trackid = df["trackId"].values[0]
    metadata_cols = metadata_df.columns
    logger.info(f"metadata cols: {metadata_cols}  ")
    metadata = get_vessel_metadata(metadata_df, trackid)
    logger.info(f"metadata: {metadata}")
    df = add_metadata_columns_to_df(df, metadata, metadata_cols)
    gdf = convert_to_geo_df(df)
    write_to_parquet(gdf, output_dir)


def write_to_parquet(df: dd, output_dir: os.PathLike) -> None:
    """writes a dataframe to a csv"""
    track_id = df["trackId"].values[0]
    month = df["send"].values[0].split("-")[1]
    day = df["send"].values[0].split("-")[2].split("T")[0]
    ves_code = df["category"].iloc[0]
    # consider doing mode here or just accepting bad flag codes or recomputing from mmsi directly
    flag_code = df["flag_code"].iloc[0]
    track_id_month = f"{track_id}_{month}"
    output_dir = (
        Path(output_dir)
        / str(ves_code)
        / str(flag_code)
        / str((len(list(Path(output_dir).glob("*"))) // 1000))
        / Path(track_id)
        / Path(track_id_month)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"{track_id}_{month}_{day}.parquet"
    output_path = os.path.join(output_dir, out_name)
    df.sort_values(by="send", ascending=True, inplace=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")


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
    "--path-to-vessel-metadata",
    type=str,
    required=False,
    help="Path to vessel metadata file",
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
    "--parquet-dir",
    type=str,
    required=False,
    default="/data-mount",
    help="Directory to read parquet files from containg unique monthly trackIds", # this flag needs to be clearer
)
@click.option(
    "--temp-dir",
    type=str,
    required=False,
    default="/data-mount",
    help="Temporary directory to use for dask storage leakage.",
)
def create_daily_trackid_parquets(
    path_to_input_csvs: str,
    path_to_vessel_metadata: str,
    output_training_dir_root: str,
    start_month: str,
    end_month: str,
    year: int,
    parquet_dir: str,
    temp_dir: str,
) -> None:
    """Creates a daily parquet file for each unique_trackId that appears on a given day in the track-incremental csvs.

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

    Returns
    -------
    None
        Writes training data to output_training_dir_root."""
    vessel_metadata_df = read_vessel_dict(path_to_vessel_metadata)

    logger.info(
        f"vessel metadata size: {vessel_metadata_df.memory_usage().sum() // 1e6} MB"
    )
    logger.info(f"vessel metadata columns: {vessel_metadata_df.columns}")
    logger.info(f"vessel metadata index: {vessel_metadata_df.index}")
    dask.config.set({"temporary_directory": temp_dir})

    # Worker saturation is set to 1.0 to reduce memory leakage and ensure smooth operation
    with Client() as client, dask.config.set(
        {"distributed.scheduler.worker-saturation": 1.0}
    ):
        logger.info(client.dashboard_link)
        months_requested = MONTHS_TO_LOAD[int(start_month) : int(end_month)]
        for month in tqdm(
            months_requested, desc="Months ais data written to track csv files"
        ):  # this is month
            logger.info(f"Reading trackIds for year {year} and month {start_month}")
            track_ids = dd.read_parquet(
                Path(parquet_dir) / f"trackIds_{year}_{month}.parquet"
            ).compute()
            logger.info(f"track_ids: {track_ids[:10]}")
            logger.info(f"Number of trackIds in month {month}: {len(track_ids)}")
            vessel_metadata_monthly = vessel_metadata_df[
                vessel_metadata_df.index.isin(track_ids.values.flatten())
            ]
            logger.info(f" vessel meta monthly: {vessel_metadata_monthly.head()}")
            logger.info(
                f"memory usage of vessel metadata for month {month}: {vessel_metadata_monthly.memory_usage().sum() // 1e6} MB"
            )
            logger.info(
                f"Number of trackIds filtered in month {month}: {len(vessel_metadata_monthly)}"
            )
            logger.info(f"reading: track-incremental.{year}-{month}-*.csv")
            files = list(
                Path(path_to_input_csvs).glob(f"track-incremental.{year}-{month}-*.csv")
            )
            tasks = []
            for file in tqdm(files, desc="Files processed"):
                df = dask.delayed(pd.read_csv)(file, usecols=DFCOLS_NO_META)
                tasks.append(
                    dask.delayed(
                        lambda df: df.groupby("trackId").apply(
                            add_metadata_convert_and_write_to_parquet,
                            metadata_df=vessel_metadata_monthly,
                            output_dir=output_training_dir_root,
                        )
                    )(df)
                )
            dask.compute(*tasks)
            logger.info(f"finished writing for month {month}")
            client.restart()

    logger.info("Computation complete! Stopping workers...")


if __name__ == "__main__":
    create_daily_trackid_parquets()
