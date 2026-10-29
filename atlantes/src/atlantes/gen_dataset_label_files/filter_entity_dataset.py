"""Given an Entity Labels file create a smaller dataset that is better for learning.

1. Remove all entities that are not buoys or vessels
2. Limit too short tracks
3. Only sample fishing or unknown entities because the rest are known not buoy?

"""

import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import pandera as pa
from atlantes.atlas.atlas_utils import (AtlasEntityLabelsTrainingWithUnknown,
                                        read_entity_label_csv,
                                        read_trajectory_lengths_file)
from atlantes.atlas.schemas import (EntityClassLabelDataModel,
                                    TrajectoryLengthsDataModel)
from atlantes.log_utils import get_logger
from atlantes.utils import write_file_to_bucket
from pandera.typing import DataFrame

logger = get_logger(__name__)
DT_STRING = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


@pa.check_types
def filter_trajectories_by_length(
    df: DataFrame[TrajectoryLengthsDataModel], min_length: int
) -> np.ndarray:
    """Filters out trajectories that are shorter than the minimum length."""
    df = df[df["Length"] >= min_length]
    return df["Path"].values


def sample_buoy_and_vessel_entities(
    df: DataFrame[EntityClassLabelDataModel], num_samples: int
) -> DataFrame[EntityClassLabelDataModel]:
    """Samples an equal number of buoy and vessel data entities."""
    buoy_samples = df[
        df["entity_class_label"] == AtlasEntityLabelsTrainingWithUnknown.BUOY.value
    ].sample(n=num_samples)
    vessel_samples = df[
        df["entity_class_label"] == AtlasEntityLabelsTrainingWithUnknown.VESSEL.value
    ].sample(n=num_samples)
    return pd.concat([buoy_samples, vessel_samples]).sample(frac=1)


@click.command()
@click.option(
    "--path_to_entity_labels",
    default="gs://ais-track-data/2022/labels/entity_class_labels-2024-02-15-18-15-34/buoy_vessel_labels.csv",
    help="Path to the entity labels csv file.",
)
@click.option(
    "--path_to_trajectory_lengths",
    default="gs://ais-track-data/2022/labels/trajectory_lengths_all_2022.csv",
    help="Path to the trajectory lengths csv file.",
)
@click.option(
    "--activity_labels_path",
    default="gs://ais-track-data/2022/labels/subpath_labels/subpath_machine_annotated_labels-2024-02-01-23-44-24.csv",
    help="Path to the activity labels csv file.",
)
@click.option(
    "--min_length",
    default=20,
    help="Minimum length of trajectories to include in the dataset.",
)
@click.option(
    "--num_samples",
    default=200000,
    help="Number of samples to take from each class (buoys and vessels).",
)
@click.option(
    "--gcp_folder",
    default=f"2022/labels/entity_class_labels-{DT_STRING}",
    help="GCP folder to upload the filtered dataset to.",
)
@click.option(
    "--filter_to_unknown_or_fishing",
    default=True,
    help="Whether to filter to only unknown or fishing.",
)
def filter_entity_dataset(
    path_to_entity_labels: str,
    path_to_trajectory_lengths: str,
    min_length: int,
    num_samples: int,
    gcp_folder: str,
    filter_to_unknown_or_fishing: bool,
) -> None:
    """Filters the entity labels dataset to a smaller dataset.

    Parameters
    ----------
    path_to_entity_labels : str
        Path to the entity labels csv file.
    path_to_trajectory_lengths : str
        Path to the trajectory lengths csv file.
    min_length : int
        Minimum length of trajectories to include in the dataset.
    num_samples : int
        Number of samples to take from each class (buoys and vessels).
    gcp_folder : str
        GCP folder to upload the filtered dataset to.
    filter_to_unknown_or_fishing : bool
        Whether to filter to only unknown or fishing.

    Returns
    -------
    None


    """
    entity_labels_df = read_entity_label_csv(path_to_entity_labels)
    logger.info(entity_labels_df.head())
    trajectory_lengths_df = read_trajectory_lengths_file(path_to_trajectory_lengths)
    logger.info(trajectory_lengths_df.head())
    logger.info("loaded activity labels")
    if filter_to_unknown_or_fishing:
        logger.info("Filtering to only unknown or fishing")
        # use metadataindex to do this filtering
    logger.info(f"Found {len(entity_labels_df)} entities")
    logger.info(f"Filtering to trajectories longer than {min_length}")
    long_enough_tracks = filter_trajectories_by_length(
        trajectory_lengths_df, min_length
    )
    logger.info(long_enough_tracks[:5])
    logger.info(f"Found {len(long_enough_tracks)} tracks longer than {min_length}")
    entity_labels_df = entity_labels_df[entity_labels_df.index.isin(long_enough_tracks)]
    logger.info(f"Found {len(entity_labels_df)} entities after filtering")

    logger.info(f"Sampling {num_samples} from each class")
    logger.info(entity_labels_df.head())
    all_samples = sample_buoy_and_vessel_entities(entity_labels_df, num_samples)
    logger.info(f"Writing {len(all_samples)} samples to GCP")
    write_file_to_bucket(
        os.path.join(gcp_folder, f"filtered_entity_labels_{DT_STRING}.csv"),
        "ais-track-data",
        all_samples,
    )


if __name__ == "__main__":
    filter_entity_dataset()
