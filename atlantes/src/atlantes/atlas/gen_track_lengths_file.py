"""Module to get the preprocessed lengths of all the files in the dataset and save it locally

Example:
For speed up, copy data locally and generate a local metadata index file
python3 get_traj_lengths_file.py --root-dir 2022/trajectory_with_subpath_monthly/ --output-path ./trajectory_lengths.csv
"""

import click
import dask
import pandas as pd
import pandera as pa
from atlantes.atlas.atlas_utils import preprocess_trackfile
from atlantes.atlas.schemas import TrajectoryLengthsDataModel
from atlantes.log_utils import get_logger
from atlantes.utils import batch, write_file_to_bucket
from dask.diagnostics import ProgressBar
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)


def get_len_of_preprocessed(df: pd.DataFrame) -> int:
    """Delayed function to get the length of a dataframe"""
    return preprocess_trackfile(df).index.size


@pa.check_types
def format_into_track_lengths_file(
    track_csv_files: list, lengths: list
) -> DataFrame[TrajectoryLengthsDataModel]:
    df_lengths = pd.DataFrame(zip(track_csv_files, lengths), columns=["Path", "Length"])
    return df_lengths


@pa.check_types
def get_track_lengths_distributed_single_chunk(
    sublist: list, use_parquet: bool = False
) -> list[int]:
    """Write the trajectory lengths to a file

    Parameters
    ----------
    sublist : list
        List of paths to the csv files
    """
    if use_parquet:
        read_func = pd.read_parquet
    else:
        read_func = pd.read_csv
    results = []
    with ProgressBar():
        # logger.info(client.dashboard_link)
        for trajectory in sublist:
            lazy_df = dask.delayed(read_func)(trajectory)
            lazy_len = dask.delayed(get_len_of_preprocessed)(lazy_df)
            results.append(lazy_len)
        results = dask.compute(*results)
    return results


@click.command()
@click.option(
    "--bucket-name", type=str, default="ais-track-data", help="name of gcloud bucket"
)
@click.option(
    "--output-path",
    default="",
    type=str,
    help="path to the output data",
)
@click.option(
    "--num-files-per-loop",
    type=int,
    required=False,
    default=20000,
    help="Minimum number of files to process per loop",
)
@click.option(
    "--metadata-index",
    type=str,
    default=None,
    help="Path to the metadata index file",
)
@click.option(
    "--use-parquet",
    is_flag=True,
    help="Flag to indicate whether to read parquet files instead of CSV",
)
def write_traj_lengths_distributed(
    bucket_name: str,
    output_path: str,
    num_files_per_loop: int,
    metadata_index: str,  # Added metadata_index option
    use_parquet: bool,  # Added parquet option
) -> None:
    """CLI for creating the trajectory lengths file that will be used to filter out short trajectories during training

    We precompute it at scale so we don't have to do it every time we train a model
    Must be recomputed if preproccess function is changed

    Parameters
    ----------
    bucket_name : str
        the name of the Google Cloud Storage bucket
    output_path : str
        the path to the output file
    num_files_per_loop : int
        the number of files to process per loop
    metadata_index : str
        the path to the metadata index file
    use_parquet : bool
        flag to indicate whether to read parquet files instead of CSV
    """
    logger.info("Reading data")
    metadata_index_df = pd.read_parquet(metadata_index)
    track_files = metadata_index_df["Path"].tolist()

    logger.info(track_files[:10])
    num_paths = len(track_files)
    logger.info(f"Found {num_paths} paths")
    if len(track_files) == 0:
        raise ValueError("No trajectories found")
    # start building the trajectory lengths file
    lengths = []
    total_chunks = num_paths // num_files_per_loop
    for sublist in tqdm(batch(track_files, num_files_per_loop), total=total_chunks):
        results = get_track_lengths_distributed_single_chunk(
            sublist, use_parquet=use_parquet
        )
        lengths.extend(results)
    logger.info("Finished processing all files")
    logger.info("Formatting into DataFrame")
    df_lengths = format_into_track_lengths_file(track_files, lengths)
    logger.info(f"Writing Trajectory lengths file to {output_path} in {bucket_name}")
    write_file_to_bucket(output_path, bucket_name, df_lengths, index=False)
    logger.info(f"Trajectory lengths written to {output_path} in {bucket_name}")


if __name__ == "__main__":
    write_traj_lengths_distributed()
