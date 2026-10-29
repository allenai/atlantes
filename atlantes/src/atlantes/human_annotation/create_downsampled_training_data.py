"""

This script will download trajectories from GCP, resample them using both change point
detection (subpath_num in the downloaded trajectories) and rdp (computed on the fly)
at EPSILON=0.0005. The downsampling will result in approximately 90% message reduction
with ideally no loss of information. The downsampling is done to maintain high
performance of the annotation platform.

These trajectories are intended to be annotated by humans and the labels should be
joined against the original (non-downsampled) trajectories.
# TODO : Ensure at least one point from each subpath is included in the downsampled data

"""

import os
import random
import subprocess  # nosec
from concurrent.futures import ThreadPoolExecutor

import click
from atlantes.human_annotation.constants import (
    MIN_MESSAGES_TO_USE_FOR_HUMAN_ANNOTATION,
    RDP_THRESHOLD,
)
from atlantes.log_utils import get_logger
from atlantes.utils import is_directory_empty, read_df_file_type_handler
from rdp import rdp

logger = get_logger(__name__)


def download_file(
    path: str, local_directory: str, use_local_files: bool, use_csv: bool
) -> None:
    """Download files from GCP paths to a local directory


    Parameters
    ----------
    path : str
        GCP path to download
    local_directory : str
        Local directory to download to if the path is stored on a local disk (use_local_files=True)
    use_local_files : bool
        Whether to use local files instead of downloading from GCP
    use_csv : bool
        Whether to use csv format for the downloaded files

    Returns
    -------
    None
        downloads files to local_directory and downsamples them using change rdp algorithm
    """

    # List files in the GCP directory
    try:
        logger.info(f"Processing {path}")
        if not use_local_files:
            ls_output = subprocess.check_output(["gsutil", "ls", path]).decode(  # nosec
                "utf-8"
            )  # nosec
            all_files = ls_output.splitlines()
        else:
            all_files = [path]  # nosec
        if not all_files:
            return  # no files in this directory, skip to next
        chosen_file = random.choice(all_files)  # nosec

        # Extract the last directory and file name from the chosen GCP path
        file_name = os.path.basename(chosen_file)
        destination_path = os.path.join(local_directory, file_name)

        # Create the destination directory if it doesn't exist
        os.makedirs(local_directory, exist_ok=True)

        # Download the chosen file
        if use_local_files:
            df = read_df_file_type_handler(chosen_file)
        else:
            subprocess.run(  # nosec
                ["gsutil", "cp", chosen_file, destination_path], check=True
            )  # nosec
            # Check the length of the CSV
            df = read_df_file_type_handler(destination_path)
        n_original_messages = len(df)
        if n_original_messages > MIN_MESSAGES_TO_USE_FOR_HUMAN_ANNOTATION:
            df["change_detected"] = (df["subpath_num"].diff() != 0).astype(int)

            geo_coords = df[["lon", "lat"]].to_numpy()
            df["mask"] = rdp(
                geo_coords, algo="iter", epsilon=RDP_THRESHOLD, return_mask=True
            )

            df_cpd_and_rdp = df[df["change_detected"] | df["mask"]]
            n_final_messages = len(df_cpd_and_rdp)
            # If use parquet change file end to csv
            if not use_csv:
                destination_path = destination_path.replace(".parquet", ".csv")

            df_cpd_and_rdp.to_csv(destination_path, index=False)
            logger.info(
                f"Wrote {destination_path} with {n_original_messages=} \
                    and {n_final_messages=}."
            )

        else:
            logger.info(
                f"{destination_path} had fewer than \
                    {MIN_MESSAGES_TO_USE_FOR_HUMAN_ANNOTATION}\
                          original messages."
            )
            if os.path.exists(destination_path) and not use_local_files:
                os.remove(destination_path)

        # Check if the directory is empty and remove it if it is
        if is_directory_empty(local_directory):
            os.rmdir(local_directory)
            logger.info("Removed empty directory")

    except Exception:
        logger.exception(f"Failed to process the directory {path}")


def download_files(
    gcp_paths: list[str],
    local_directory: str,
    use_local_files: bool,
    use_csv: bool,
) -> None:
    """Download files from GCP paths to a local directory in parallel"""
    with ThreadPoolExecutor() as executor:
        executor.map(
            lambda path: download_file(path, local_directory, use_local_files, use_csv),
            gcp_paths,
        )


def read_paths_file_to_get_paths_to_tracks(
    paths_file: str, use_local_files: bool
) -> list[str]:
    """Open a file containing GCP paths to download."""
    if not paths_file.endswith(".txt"):
        raise ValueError("Please provide a .txt file.")
    with open(paths_file, "r") as file:
        if use_local_files:
            paths = [line.strip() for line in file]
        else:
            paths = [line.strip() + "*" for line in file]
    return paths


@click.command()
@click.option(
    "--paths_txt_folder",
    type=str,
    required=False,
    default=None,
    help="Path to a folder containing GCP paths to download.",
)
@click.option(
    "--paths_file",
    type=str,
    required=False,
    default=None,
    help="Path to a file containing GCP paths to download.",
)
@click.option(
    "--use_local_files",
    type=bool,
    is_flag=True,
    default=False,
    help="Use local files instead of downloading from GCP.",
)
@click.option(
    "--use_csv",
    type=bool,
    is_flag=True,
    default=False,
    help="Use csv format for the downloaded files.",
)
def main(
    paths_txt_folder: str, paths_file: str, use_local_files: bool, use_csv: bool
) -> None:
    """Download files from GCP paths."""
    logger.info(f"use local files:{use_local_files}")
    if paths_file is None and paths_txt_folder is None:
        raise ValueError("Please provide either a paths_file or paths_txt_folder.")
    if paths_file is not None:
        paths_lst = [paths_file]
    else:  # paths_txt_folder is not None
        files = os.listdir(paths_txt_folder)

        # Get absolute paths
        paths_lst = [
            os.path.abspath(os.path.join(paths_txt_folder, file)) for file in files
        ]

    for paths_file in paths_lst:
        logger.info(f"Processing {paths_file}")
        project_name = paths_file.split(".txt")[0]

        paths = read_paths_file_to_get_paths_to_tracks(paths_file, use_local_files)
        logger.debug(f"Downloading files from {paths[:10]}")
        download_files(paths, project_name, use_local_files, use_csv)


if __name__ == "__main__":
    main()
