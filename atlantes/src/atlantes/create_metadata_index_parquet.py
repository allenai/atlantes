"""Creates an index for the gcp bucket storing the training data for the ATLAS activity model.
THis index allows random access and stratified sampling across any of the information stored in the parquet file names.
# TODO: Add some sort of geometry to this index so that we can do spatial queries on the data.
"""

from pathlib import Path
from typing import Any

import click
import pandas as pd
import pandera as pa
from atlantes.datautils import GCP_TRACK_DATA_BUCKET
from atlantes.human_annotation.schemas import TrackMetadataIndex
from atlantes.log_utils import get_logger
from google.cloud import storage
from pandera.typing import DataFrame

logger = get_logger(__name__)

# Chosen because it is enough to validate quickly without taking too long
NUM_ROWS_TO_VALIDATE = 100


# Dtype ocercion is applied before validation
@pa.check_types(head=NUM_ROWS_TO_VALIDATE)
def load_metadata_index(
    path_to_metadata: str, **kwargs: Any
) -> DataFrame[TrackMetadataIndex]:
    """Loads the metadata and returns a DataFrame of trackIds

    Note: This does not include trackIds that do not transmit any metadata at all

    Parameters
    ----------
    path_to_metadata : str
        Path to the metadata CSV file
        This file should contain (at minimum the following columns):
            - unique_id: trackId (e.g B:123456789:123456789:123456789)
            - ais_type: vessel type (e.g 30 for fishing) see src/config/AIS_categories.csv
            - vessel_category: vessel category (e.g fishing, cargo, tanker, etc.)
            - flag_code: flag code (e.g USA, CHN, etc.)

    """
    logger.info(f"Loading in the metadata at {path_to_metadata}")
    metadata = pd.read_parquet(path_to_metadata, **kwargs)
    return metadata


def filter_index_by_ais_categories(
    metadata: DataFrame[TrackMetadataIndex], ais_categories_subset: list[int]
) -> DataFrame[TrackMetadataIndex]:
    """Filters the metadata index to only include tracks that are in the specified categories

    Parameters
    ----------
    metadata : DataFrame[TrackMetadataIndex]
        The metadata index
    ais_categories_subset: list[int]
        The AIS categories to filter the metadata index to based on the reported ais_type e.g 30 for fishing

    Returns
    -------
    DataFrame[TrackMetadataIndex]
        The filtered metadata index
    """
    logger.info(
        f"Filtering the metadata index to only include {ais_categories_subset=}"
    )
    filtered_metadata = metadata[metadata["ais_type"].isin(ais_categories_subset)]
    return filtered_metadata


def filter_index_by_month(
    metadata: DataFrame[TrackMetadataIndex], month: str
) -> DataFrame[TrackMetadataIndex]:
    """Filters the metadata index to only include tracks from a specific month

    Parameters
    ----------
    metadata : DataFrame[TrackMetadataIndex]
        The metadata index
    month : int
        The month to filter the metadata index to

    Returns
    -------
    DataFrame[TrackMetadataIndex]
        The filtered metadata index
    """
    logger.info(f"Filtering the metadata index to only include tracks from {month=}")
    filtered_metadata = metadata[metadata["month"] == month]
    return filtered_metadata


@click.command()
@click.option(
    "--bucket-name", default=GCP_TRACK_DATA_BUCKET, help="Name of the GCP bucket"
)
@click.option("--blob-name", required=True, help="Name of the blob")
@click.option("--file-type", default="parquet", help="Type of the file")
def create_metadata_index_parquet(
    bucket_name: str,
    blob_name: str,
    file_type: str = "parquet",
) -> None:
    """Create a metadata index for the ATLAS activity model training data."""
    # Get a list of all the parquet files in the training data bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # Create a pandas dataframe to store the metadata index
    metadata_index = pd.DataFrame(
        columns=[
            "file_name",
            "year",
            "flag_code",
            "month",
            "ais_type",
            "trackId",
            "Path",
        ]
    )
    # TODO: SPECify DATA TYPES
    # Iterate over the parquet files and extract the metadata
    # Get year outside of the loop
    blob_path = Path(blob_name)
    year = blob_path.parts[0]
    dataset_name = blob_path.parts[1]
    flag_codes = []
    ais_types = []
    trackIds = []
    months = []
    file_names = []
    full_paths = []
    for file in bucket.list_blobs(prefix=blob_name):
        file_name = Path(file.name)
        file_name_parts = file_name.parts
        if not file.name.endswith(f".{file_type}"):
            continue
        # Extract the metadata from the file name
        flag_code = file_name_parts[3]
        ais_type = file_name_parts[2]
        trackId = file_name_parts[5]
        month = str(file_name.stem).split("_")[1]
        full_path = f"gs://{bucket_name}/{file.name}"
        # Append the metadata to the lists
        flag_codes.append(flag_code)
        ais_types.append(ais_type)
        trackIds.append(trackId)
        months.append(month)
        file_names.append(file.name)
        full_paths.append(full_path)

    # Add the metadata to the dataframe
    metadata_index["file_name"] = file_names
    metadata_index["year"] = year
    metadata_index["flag_code"] = flag_codes
    metadata_index["month"] = months
    metadata_index["ais_type"] = ais_types
    metadata_index["trackId"] = trackIds
    metadata_index["Path"] = full_paths
    schema = TrackMetadataIndex.to_schema()
    validated_metadata_index = schema.validate(metadata_index, sample=100)
    logger.info(validated_metadata_index.head())
    logger.info(validated_metadata_index.shape)
    output_path = f"gs://{bucket_name}/{year}/metadata_index_{dataset_name}.parquet"
    validated_metadata_index.to_parquet(
        output_path, engine="pyarrow", compression="snappy", index=False
    )


if __name__ == "__main__":
    create_metadata_index_parquet()
