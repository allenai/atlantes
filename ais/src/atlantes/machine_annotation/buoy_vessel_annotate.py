""" Module for generating a csv of buoy vessel machine annotations

This module generates a csv file containing trackIds and the labels for the buoy/vessel classification task.
Dask is use to parallelize the annotation process. Reccomended to run this on a
machine with at least 8 cores and 32GB of RAM to effectively paralellize.

NOTE: number of files per loop is a hyperparameter that should be tuned to your machines hardware

1. Set up GCLOUD credentials on your machine using gcloud default application credentials
2. Ensure the bucket name is correct and the track data is in the specified directory


"""

import math
import re
from pathlib import Path

import dask
import pandas as pd
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import (
    ENGLISH_SPEAKING_MMSIS_CODES, NAME_PATTERNS_FOR_BUOYS)
from atlantes.utils import batch, write_file_to_bucket
from dask.diagnostics import ProgressBar
from tqdm import tqdm

logger = get_logger(__name__)


def is_buoy_based_on_name(mmsi: str, entity_name: str) -> bool:
    """Checks if a track is a buoy based on the name of the entity

    Parameters
    ----------
    mmsi : str
        the mmsi of the entity
    entity_name : str
        the name of the entity

    Returns
    -------
    bool
        True if the track is a buoy, False otherwise
    """
    is_buoy_in_name = "buoy" in entity_name.lower()
    # Ensure not a pun boat name
    is_from_english_speaking_country = any(
        [mmsi.startswith(code) for code in ENGLISH_SPEAKING_MMSIS_CODES]
    )
    patterns_regex = re.compile("|".join(NAME_PATTERNS_FOR_BUOYS), re.IGNORECASE)
    does_name_match_known_buoy_patterns = bool(patterns_regex.search(entity_name))
    is_buoy = bool(
        is_buoy_in_name & ~is_from_english_speaking_country
        | does_name_match_known_buoy_patterns
    )
    return is_buoy


def label_buoy_or_vessel(df: pd.DataFrame) -> int:
    """Labels a csv as buoy, vessel or unknown

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the track to label

    Returns
    -------
    int
        the label of the track, 0 for buoy, 1 for vessel, -1 for unknown
    """
    mmsi = df["mmsi"].astype(str)
    is_buoy = is_buoy_based_on_name(df.mmsi.values[0], df.name.values[0])
    is_unreliable_mmsi = (
        mmsi.str.startswith("9").any()
        | mmsi.str.startswith("8").any()
        | mmsi.str.startswith("0").any()
    )
    is_vessel = ~is_buoy & ~is_unreliable_mmsi
    return 1 if is_buoy else 0 if is_vessel else -1


def generate_buoy_vessel_labels(
    metadata_index_path: str,
    bucket_name: str,
    output_dir: str,
    num_files_per_loop: int,
    use_parquet: bool = True,
) -> None:
    """Generates a csv file containing the paths to the incremental data and the labels for the buoy vessel task

    Process can be viewed by going to the dask dashboard link and tunneling
    gcloud compute ssh [vm-name] -- -NL [localport]:localhost:8787
    and then going to https://localhost:2222/status
    Parameters
    ----------
    bucket_name : str
        name of gcloud bucket
    output_dir : str
        path to the output data
    num_files_per_loop : int
        number of files to process per loop, should be tuned to your machines hardware
    """
    metadata_index_df = pd.read_parquet(metadata_index_path)
    input_files = metadata_index_df["Path"].tolist()
    if use_parquet:
        read_func = pd.read_parquet
        columns = ["mmsi", "name"]
        from functools import partial

        read_func = partial(read_func, columns=columns)
    else:
        logger.warning(
            "CSV mode will be deprecated in the future please use parquet files when possible"
        )
        read_func = pd.read_csv
        # Read from the metadata index instead
    num_files = len(input_files)
    total = math.ceil(num_files / num_files_per_loop)
    labels_list = []
    for sublist in tqdm(batch(input_files, num_files_per_loop), total=total):
        with ProgressBar():
            results = []
            for path in sublist:
                lazy_df = dask.delayed(read_func)(path)
                lazy_label = dask.delayed(label_buoy_or_vessel)(lazy_df)
                results.append(lazy_label)
            labels = dask.compute(*results)
        labels_list.extend(labels)
    df = pd.DataFrame(labels_list, columns=["entity_class_label"], index=input_files)
    df.index.name = "Path"
    output_path = str(Path(output_dir) / "buoy_vessel_labels.csv")
    logger.info(f"Writing buoy vessel labels to {output_path} in {bucket_name}")
    write_file_to_bucket(output_path, bucket_name, df)
    logger.info("Finished generating buoy vessel labels")
