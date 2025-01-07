from io import BytesIO, StringIO

import pandas as pd
import pyarrow.parquet as pq
import pytest
from google.cloud import storage


@pytest.fixture(scope="module")
def read_parquet_from_gcs() -> pd.DataFrame:
    """Read a Parquet file from GCS and return a DataFrame."""
    bucket_name = "ais-track-data"
    parquet_file = "2022/all_2022_tracks_w_subpaths/1/ARE/0/B:470832000:1513926758:2344566:1145508/B:470832000:1513926758:2344566:1145508_04.parquet"

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Download the Parquet file into memory
    parquet_blob = bucket.blob(parquet_file)
    parquet_data = parquet_blob.download_as_bytes()

    # Use pyarrow to read the Parquet file from the in-memory bytes
    parquet_buffer = BytesIO(parquet_data)
    table = pq.read_table(parquet_buffer)

    # Convert the table to a pandas DataFrame
    df = table.to_pandas()

    return df


@pytest.fixture(scope="module")
def read_csv_from_gcs() -> pd.DataFrame:
    """Read a CSV file from GCS and return a DataFrame."""
    bucket_name = "ais-track-data"
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    csv_file = "2022/all_2022_tracks_w_subpaths/1/ARE/0/B:470832000:1513926758:2344566:1145508/B:470832000:1513926758:2344566:1145508_04.csv"
    # Download the CSV file into memory
    csv_blob = bucket.blob(csv_file)
    csv_data = csv_blob.download_as_text()

    # Use pandas to read the CSV data from the in-memory string
    df = pd.read_csv(StringIO(csv_data))

    return df


def test_compare_files(
    read_parquet_from_gcs: pd.DataFrame, read_csv_from_gcs: pd.DataFrame
) -> None:
    """Compare a Parquet file and a CSV file row by row."""
    df_parquet = read_parquet_from_gcs
    df_csv = read_csv_from_gcs

    # Check if the DataFrames have the same shape
    assert df_parquet.shape == df_csv.shape

    # Compare the DataFrames row by row
    comparison = df_parquet.equals(df_csv)

    assert comparison
