from io import BytesIO, StringIO
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from atlantes.log_utils import get_logger
from google.cloud import storage
from tqdm import tqdm

logger = get_logger(__name__)


def list_files(bucket_name: str, prefix: str = "") -> list[str]:
    """List up to max_files files in the GCS bucket."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Create an iterator for the blobs with pagination
    blobs = bucket.list_blobs(prefix=prefix)

    # Collect file names
    files = []
    for blob in tqdm(blobs):
        files.append(blob.name)

    return files


def convert_csv_to_parquet(bucket_name: str, file: str) -> None:
    """Convert a file in GCS to Parquet format and upload it back, assuming it's a CSV."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Download CSV file
        csv_blob = bucket.blob(file)
        csv_data = csv_blob.download_as_text()
        # Convert file to Parquet using Pandas and PyArrow
        df = pd.read_csv(StringIO(csv_data))

        # Convert DataFrame to Parquet format in memory
        parquet_buffer = BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_buffer)

        # Reset buffer position to the beginning
        parquet_buffer.seek(0)

        # Determine the Parquet file path
        parquet_file = file.replace(".csv", ".parquet")

        # Upload Parquet file to GCS from memory
        parquet_blob = bucket.blob(parquet_file)
        parquet_blob.upload_from_file(
            parquet_buffer, content_type="application/octet-stream"
        )

        logger.info(f"Converted {file} to {parquet_file}")

    except Exception as e:
        logger.exception(f"Failed to process {file}: {e}", exc_info=True)


def main(bucket_name: str, prefix: str = "", num_files: int = 5) -> None:
    csv_files = list_files(bucket_name, prefix)

    # Limit to a handful of files
    csv_files = csv_files[:num_files]

    # Use multiprocessing Pool to parallelize the conversion
    with Pool() as pool:
        pool.starmap(
            convert_csv_to_parquet, [(bucket_name, csv_file) for csv_file in csv_files]
        )


if __name__ == "__main__":
    bucket_name = "ais-track-data"
    prefix = (
        "2022/all_2022_tracks_w_subpaths/"  # Optional, set to '' if no prefix is needed
    )
    # num_files = 1000  # Number of files to process
    main(bucket_name, prefix)
