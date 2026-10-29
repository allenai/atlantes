"""Sample from Fishing regions

Given monthly geojsons with aois for fishing areas,
samples from the ais data stratified across AOIs and months
amongst vessels with ship type fishing or unknown
and writes the paths to the track files to a .txt file in project sized chunks
in the current directory

Use big machine with ample memory and cores (for gcp N1 with 96 CPUs and 624 GB RAM)
Geojsons must be projected into EPSG::3395 or equivalent web mercator

# TODO: Create precomputed spatiotemporal geospatial indexes for each month and year this will make it a lot faster
# TODO: Move some of these functions into a general utils file
# I need to kill dask warning
# TODO: Core performance bottleneck is reading in the parquets and maintaining the file names as this is not supported with dd.read_parquet
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Union

import click
import dask
import dask.dataframe as dd
import pandas as pd
import spatialpandas as sp
from atlantes.create_metadata_index_parquet import (
    filter_index_by_ais_categories,
    filter_index_by_month,
    load_metadata_index,
)
from atlantes.human_annotation.constants import MonthsToLoad
from atlantes.human_annotation.human_annotation_utils import (
    filter_already_sampled_tracks_from_metadata_index,
    write_file_names_to_project_txts,
)
from atlantes.human_annotation.schemas import TrackMetadataIndex
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import (
    FISHING_VESSEL_CATEGORIES,
    UNKNOWN_VESSEL_CATEGORIES,
)
from dask import delayed
from dask.diagnostics import ProgressBar
from pandera.typing import DataFrame
from shapely.geometry import Polygon
from spatialpandas.io import read_parquet

logger = get_logger(__name__)

# Supress numba info logging so that it doesn't fill up the logs
logging.getLogger("numba").setLevel(logging.WARNING)

MONTHLY_JSON_AOIS_FOLDER = (
    Path(__file__).parents[3] / "data" / "human_annotations" / "fishing_areas"
)


def load_aois_old(path_to_aois_json: Path) -> list[Polygon]:
    """Load in the geojson containing the aois

    Parameters
    ----------
    path_to_aois_json : str
        Path to the geojson containing the aois

    Returns
    -------
    list[Polygon]

        list of polygons of the aois
    """
    logger.warning(
        "Using old geojson file structure this needs to be updated will be removed in the future"
    )
    with open(path_to_aois_json) as f:
        aois = json.load(f)
    # Deals with old aoi structure
    return [
        Polygon(polygon["geometry"]["coordinates"][0][0])
        for polygon in aois["features"]
    ]


def load_aois(path_to_aois_json: Path) -> list[Polygon]:
    """Load in the geojson containing the aois

    Parameters
    ----------
    path_to_aois_json : str
        Path to the geojson containing the aois

    Returns
    -------
    list[Polygon]
        list of polygons of the aois
    """
    with open(path_to_aois_json) as f:
        aois = json.load(f)
    return [
        Polygon(polygon["geometry"]["coordinates"][0]) for polygon in aois["features"]
    ]


# Other option is to read in and then search for the paths in the metadata index via trackId month year, means more columns to be read in
def read_parquets_with_file_names(files: list[str], **kwargs: Any) -> dd.DataFrame:
    """Reads in the parquet files and adds the file name as a column


    Parameters
    ----------
    files : list[str]
        List of paths to the track files

    Returns
    -------

    dd.DataFrame
        Dask dataframe containing the trackId, lat, lon, and file_name columns
    """
    lazy_dataframes = []
    for path in files:
        df = delayed(read_parquet)(path, **kwargs)
        df = delayed(sp.GeoDataFrame.assign)(df, Path=path)
        lazy_dataframes.append(df)

    ddf = dd.from_delayed(lazy_dataframes)
    return ddf


def build_geospatial_index(
    files: list[str], partition_size: Union[int, str] = "64MB"
) -> dd.DataFrame:
    """Builds a geospatial index from the given files

    Parameters
    ----------
    files : list[str]
        List of paths to the track files

    Returns
    -------
    dd.DataFrame
        Dask dataframe containing the trackId, lat, and lon columns
    """
    ddf = read_parquets_with_file_names(files, columns=["geometry"])
    ddf = ddf.repartition(partition_size=partition_size)
    return ddf


def filter_messages_in_aois(
    ais_data: dd.DataFrame, aois: list[Polygon]
) -> list[pd.DataFrame]:
    """Filter messages in the bounding box containing the aoi

    Parameters
    ----------
    ais_data : dd.DataFrame
        Fishing unknown track ais data
    aois : GeoDataFrame
        GeoDataFrame containing the aois as a polygon array in a columnn called 'aois'

    Returns
    -------
    list[pd.DataFrame]
        list of dataframes containing the messages in the aoi

    """
    with ProgressBar():
        messages_in_aois = dask.compute(
            *[
                ais_data[ais_data.geometry.intersects_bounds(aoi_polygon.bounds)]
                for aoi_polygon in aois
            ]
        )
    return messages_in_aois


def get_paths_in_aoi(messages_in_aoi: pd.DataFrame) -> pd.Series:
    """Get trackIds meetin in aoi criteria

    Parameters
    ----------
    messages_in_aoi : pd.DataFrame
        Messages in the aoi

    Returns
    -------
    pd.Series

        Series of Paths that are in the aoi
    """
    return messages_in_aoi.Path.drop_duplicates()


def stratified_sample_across_aois(
    Path_dfs: list[pd.Series], n_samples: int
) -> pd.Series:
    """Stratified sample across the aois

    Parameters
    ----------
    Path_dfs : pd.DataFrame
        TrackIds in the aoi
    n_samples : int
        Number of samples to take

    Returns
    -------
    pd.Series
        Series of Paths sampled from the aois in a stratified manner
    """
    num_aois = len(Path_dfs)
    logger.info(f"Sampling {n_samples} Paths from {num_aois} aoi")
    samples_per_aoi = n_samples // num_aois + 1
    logger.info(f"Sampling {samples_per_aoi} Paths from each of {num_aois} aoi")

    def sample_from_aoi(
        Path_df: pd.Series,
    ) -> pd.Series:
        """Sample from a single aoi without replacement"""
        num_in_aoi = len(Path_df)
        logger.info(f"Sampling {samples_per_aoi} Paths from {num_in_aoi} in aoi")
        sample_per_aoi = min(samples_per_aoi, num_in_aoi)
        return Path_df.sample(n=sample_per_aoi)

    return pd.concat([sample_from_aoi(Path_df) for Path_df in Path_dfs])


def format_geojson_file_name(
    year: int, month: str, ais_category_names: list[str]
) -> str:
    """Format the geojson file name for the given year, month, and ais categories."""
    return f"{year}_{month}_{'_and_'.join(ais_category_names)}_areas.geojson"


def filter_index_by_aois(
    metadata: DataFrame[TrackMetadataIndex],
    aois: list[Polygon],
) -> list[pd.Series]:
    """Filters the metadata index by aois.
    This is doen by reading in the files in the metadata index and filtering to the aoi

    """
    with ProgressBar():
        geospatial_index_ddf = build_geospatial_index(
            metadata["Path"].to_list()
        ).persist()
    logger.info("Filter across all the aois in the polygon array fishing")
    ais_type_messages_in_aois = filter_messages_in_aois(geospatial_index_ddf, aois)
    paths_in_aois = [
        get_paths_in_aoi(messages) for messages in ais_type_messages_in_aois
    ]
    return paths_in_aois


def sample_tracks_month(
    fishing_metadata_df: DataFrame[TrackMetadataIndex],
    unknown_metadata_df: DataFrame[TrackMetadataIndex],
    percent_out_of_aoi: float,
    n_samples_fishing: int,
    n_samples_unknown: int,
    month: str,
    year: int,
) -> list[str]:
    """Stratified smaple across fishing regions for a month


    Parameters
    ----------
    fishing_metadata_df : DataFrame[TrackMetadataIndex]
        Metadata for fishing vessels
    unknown_metadata_df : DataFrame[TrackMetadataIndex]
        Metadata for unknown vessels
    percent_out_of_aoi : float
        Percentage of tracks outside AOI.
    n_samples : int
        Number of samples to generate.
    month: str
        Month to sample from in numerical format (e.g. '01')
    year : int
        Year to sample from.

    Returns
    -------
    list[str]
        list of paths to track files

    """
    percent_in_aoi = 1 - percent_out_of_aoi
    # Load in the geojson
    logger.info(f"Loading in the geojson at {MONTHLY_JSON_AOIS_FOLDER}")
    if year == 2022:
        logger.warning(
            "For 2022, the geojson file structure has changed and this function needs to be updated to handle this"
        )
        aois_json_path = MONTHLY_JSON_AOIS_FOLDER / format_geojson_file_name(
            year, month, ["fishing", "unknown"]
        )
        fishing_aois = load_aois_old(aois_json_path)
        unknown_aois = fishing_aois
        logger.warning("Using the same aois for fishing and unknown")
    else:
        # Load fishing ais_type aois
        aois_json_path = MONTHLY_JSON_AOIS_FOLDER / format_geojson_file_name(
            year, month, ["fishing"]
        )
        fishing_aois = load_aois(aois_json_path)

        # Load unknown ais type aois
        aois_json_path = MONTHLY_JSON_AOIS_FOLDER / format_geojson_file_name(
            year, month, ["unknown"]
        )
        unknown_aois = load_aois(aois_json_path)

    # filter by month
    logger.info(f"Filtering fishing to month {month}")
    fishing_metadata_df = filter_index_by_month(fishing_metadata_df, month)
    logger.info(f" fishing  month {month} shape: {fishing_metadata_df.shape}")
    unknown_metadata_df = filter_index_by_month(unknown_metadata_df, month)
    logger.info(f" unknown  month {month} shape: {unknown_metadata_df.shape}")

    paths_in_aois_fishing = filter_index_by_aois(fishing_metadata_df, fishing_aois)
    logger.info("Sampling fishing")
    sampled_Paths_in_aois_fishing = stratified_sample_across_aois(
        paths_in_aois_fishing, n_samples_fishing
    )
    paths_in_all_aois_fishing = pd.concat(paths_in_aois_fishing).tolist()
    logger.info(
        f"Samples found for in aois fishing: {len(sampled_Paths_in_aois_fishing)}"
    )
    paths_in_aois_unknown = filter_index_by_aois(unknown_metadata_df, unknown_aois)
    logger.info("Sampling unknown")
    sampled_Paths_in_aois_unknown = stratified_sample_across_aois(
        paths_in_aois_unknown, n_samples_unknown
    )

    paths_in_all_aois_unknown = pd.concat(paths_in_aois_unknown).tolist()
    logger.info(
        f"Samples found for in aois unknown: {len(sampled_Paths_in_aois_unknown)}"
    )
    paths_in_all_aois = paths_in_all_aois_fishing + paths_in_all_aois_unknown
    logger.info("Get paths outside of aois")

    logger.info(
        f"percent in aoi {percent_in_aoi} percent out of aoi {percent_out_of_aoi}"
    )

    logger.info("Sampling paths not in aois")
    fishing_paths_not_in_aois = fishing_metadata_df[
        ~fishing_metadata_df["Path"].isin(paths_in_all_aois)
    ]
    num_out_of_aoi_samples = int(
        (n_samples_fishing + n_samples_unknown) * percent_out_of_aoi
    )
    logger.info(f"Number of out of aoi samples: {num_out_of_aoi_samples}")
    sampled_Paths_out_of_aois = fishing_paths_not_in_aois.sample(
        n=num_out_of_aoi_samples
    ).Path
    logger.info(f"Samples found: {len(sampled_Paths_out_of_aois)}")
    logger.info(f"Sampled paths out of aois: {sampled_Paths_out_of_aois}")

    paths_to_track_files = pd.concat(
        [
            sampled_Paths_in_aois_fishing,
            sampled_Paths_out_of_aois,
            sampled_Paths_in_aois_unknown,
        ]
    ).tolist()
    return paths_to_track_files


@click.command()
@click.option(
    "--metadata_path",
    default="gs://ais-track-data/2023/metadata_index_ais_2023_subpath.parquet",
    help="Path to metadata CSV file.",
)
@click.option(
    "--percent_out_of_aoi",
    default=0.1,
    type=float,
    help="Percentage of tracks outside AOI.",
)
@click.option(
    "--n_fishing_samples", default=1100, type=int, help="Number of fishing samples."
)
@click.option(
    "--n_unknown_samples", default=300, type=int, help="Number of unknown samples."
)
@click.option("--start_month_idx", default=0, type=int, help="Start month index.")
@click.option("--end_month_idx", default=12, type=int, help="End month index.")
@click.option("--year", default=2023, type=int, help="Year to sample from.")
@click.option(
    "--project_size",
    default=50,
    type=int,
    help="Number of files to put in each project",
)
@click.option(
    "--output_directory",
    default="./sampled_tracks",
    type=str,
    help="Path to the directory where the track files will be written.",
)
def sample_tracks(
    metadata_path: str,
    percent_out_of_aoi: float,
    n_fishing_samples: int,
    n_unknown_samples: int,
    start_month_idx: int,
    end_month_idx: int,
    year: int,
    project_size: int,
    output_directory: str,
) -> None:
    """Sample tracks from fishing regions for a given year and write to project files

    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file.
    percent_out_of_aoi : float
        The percentage of samples to be taken outside the AOIs.
    n_fishing_samples : int
        The number of fishing samples to be taken.
    n_unknown_samples : int
        The number of unknown samples to be taken.
    start_month_idx : int
        The index of the starting month (inclusive) for sampling.
    end_month_idx : int
        The index of the ending month (exclusive) for sampling.
    year : int
        The year for which the tracks will be sampled.
    project_size : int
        The number of track files to be written in each project-sized chunk.
    output_directory : str
        Path to the directory where the track files will be written.

    Returns
    -------
    None
        Writes sampled track files to a .txt file in project-sized chunks
        in the specified output directory.
    """
    # There is a spatial pandas Deprecation Warning that I am filtering for logging clarity
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pandas._libs.spatial"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger.info(f"Loading in the metadata at {metadata_path}")
    metadata = load_metadata_index(metadata_path)
    logger.info(f"Metadata head:\n{metadata.head()}")

    logger.info("Filtering already sampled tracks")
    metadata = filter_already_sampled_tracks_from_metadata_index(metadata)

    logger.info("Filtering Fishing")
    logger.warning("Filtering already Sampled requires elastic search")
    fishing_metadata_df = filter_index_by_ais_categories(
        metadata, FISHING_VESSEL_CATEGORIES
    )
    logger.info(
        f"Fishing metadata head:\n{fishing_metadata_df.head()} \
                shape: {fishing_metadata_df.shape}"
    )

    unknown_metadata_df = filter_index_by_ais_categories(
        metadata, UNKNOWN_VESSEL_CATEGORIES
    )
    logger.info(
        f"Unknown metadata head:\n{unknown_metadata_df.head()} \
                shape: {unknown_metadata_df.shape}"
    )

    month_count = 0
    paths_to_track_files = []
    try:
        for month in MonthsToLoad:
            if month_count < start_month_idx or month_count >= end_month_idx:
                logger.info(f"Skipping month {month.name}")
                month_count += 1
                continue
            month_count += 1
            total_months = end_month_idx - start_month_idx
            n_samples_per_month_fishing = n_fishing_samples // total_months + 1
            n_samples_per_month_unknown = n_unknown_samples // total_months + 1
            month_str = month.value
            logger.info(f"Sampling for month {month.name}")
            paths_to_track_files.extend(
                sample_tracks_month(
                    fishing_metadata_df,
                    unknown_metadata_df,
                    percent_out_of_aoi,
                    n_samples_per_month_fishing,
                    n_samples_per_month_unknown,
                    month_str,
                    year,
                )
            )
            logger.info(f"Finished sampling for month {month.name}")
        logger.info(paths_to_track_files)
    except Exception as e:
        logger.error(f"Error sampling for month {month.name}: {e}")
        raise e
    finally:
        write_file_names_to_project_txts(
            paths_to_track_files, project_size, output_directory
        )
        logger.info("Finished sampling for all requested months")


if __name__ == "__main__":
    sample_tracks()
