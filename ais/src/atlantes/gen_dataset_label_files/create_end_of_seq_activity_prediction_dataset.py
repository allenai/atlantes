"""Module for creating the dataset for the end of sequence activity prediction task.


TASK Description: Given a sequence of Ais messages, predict the activity occurring at the end of the sequence.
INput: A sequence of Ais messages ending with the activity labeled message, the dataset should be irrespective of the lookback length


Each sample should include the following data:

- A class label (should I actually translate the class label in the dataset so it is more modular?)
- A timestamp for the message corresponding to that class label
- the track_Id of the vessel
- The source of the message (human annotated or inferred)
- A list of temporally ordered paths to the relevant data files needed to create the sample



Additional Capabilities:
- Enable Inference of labels for points between Human Annotated points
- Incorporate MA data
- Pull Feedback
"""

import ast
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from atlantes.atlas.schemas import ActivityEndOfSequenceLabelDataModel
from atlantes.datautils import (ALL_META_DATA_INDEX_PATHS,
                                GCP_BUCKET_NAME_AIS_TRACKS)
from atlantes.gen_dataset_label_files.subsample_end_of_seq_activity_dataset import \
    SubsampleEndOfSeqActivityLabels
from atlantes.human_annotation.constants import (COMPLETED_ANNOTATIONS_FOLDER,
                                                 DT_STRING,
                                                 GCP_BUCKET_DOWNSAMPLED)
from atlantes.human_annotation.human_annotation_utils import \
    get_all_rejected_or_submitted_trajectory_id_project_id_pairs
from atlantes.human_annotation.prepare_annotations_for_training import \
    get_end_of_sequence_activity_labels_from_downsampled_track_files
from atlantes.human_annotation.schemas import TrackMetadataIndex
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import \
    list_csv_files_from_bucket
from atlantes.machine_annotation.end_of_sequence_activity_task_data_mining import \
    AISActivityDataMiner
from atlantes.utils import (export_dataset_to_gcp, load_all_metadata_indexes,
                            plot_and_save_ais_dataset)
from google.cloud import storage
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)
EXCLUDED_FILE_STRINGS = ["playground", "Example", "osr", "prelabel"]
DEFAULT_OUTPUT_DIR = f"labels/end_of_sequence_activity_dataset_{DT_STRING}/"


def upload_config_to_gcp(config: dict, bucket_name: str, file_name: str) -> None:
    """Upload the config as a JSON file to GCP bucket.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    bucket_name : str
        The name of the GCP bucket.
    file_name : str
        The name of the JSON file.

    Returns
    -------
    None
    """
    # Convert the config to JSON string
    config_json = json.dumps(config)

    # Create a client for GCP storage
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Create a blob with the file name
    blob = bucket.blob(file_name)

    # Upload the JSON string as the content of the blob
    blob.upload_from_string(config_json)


def load_dataset_creation_configuration() -> dict:
    """Load the dataset creation configuration."""
    directory = Path(__file__).parent
    file = str(directory / "dataset_creation_end_of_seq_config.yaml")
    with open(file, "r") as yaml_file:
        dataset_creation_configuration = yaml.safe_load(yaml_file)
    return dataset_creation_configuration


def exclude_file(
    file_name: str, exclude_strs: list[str], excluded_str_pairs: list[tuple[str, str]]
) -> bool:
    """Exclude files from the downsampled bucket if they contain any of the exclude strings.

    Parameters
    ----------
    file_name : str
        The file to check
    exclude_strs : list[str]
        The list of strings to check for in the file name
    excluded_str_pairs : list[tuple[str,str]]
        The list o fpairs strings to check for in the file name

    Returns
    -------
    bool
        Whether the file should be excluded from the dataset
    """
    exclude_based_on_single_strings = any(
        exclude_str.lower() in file_name.lower() for exclude_str in exclude_strs
    )

    exclude_based_on_pairs = any(
        (
            exclude_str1.lower() in file_name.lower()
            and exclude_str2.lower() in file_name.lower()
        )
        for exclude_str1, exclude_str2 in excluded_str_pairs
    )
    return exclude_based_on_single_strings or exclude_based_on_pairs


# Function for annotator consensus calculation if a track has been labeled by one or more people
def get_activity_label_consensus(activity_label_df: pd.DataFrame) -> pd.DataFrame:
    """Get the activity label consensus for points annotated by multiple annotators"""
    # Group annotations for each tracId and send
    activity_label_df.groupby(["trackId", "send"]).activity_type_name.apply(
        lambda x: x.values.tolist()
    )
    # we cna pick most common and drop if there is a tie many options
    raise NotImplementedError("This function is not yet implemented")


def drop_rows_with_specific_labels(
    df: pd.DataFrame, labels_to_drop: list[str]
) -> pd.DataFrame:
    """Drop rows with specific labels from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    labels_to_drop : list[int]
        List of labels to drop.

    Returns
    -------
    pd.DataFrame
        Dataframe with rows with specific labels dropped.
    """
    return df[~df["activity_type_name"].isin(labels_to_drop)]


def build_ais_activity_dataset_labels_from_human_annotations(
    files: list[str],
    metadata_df: DataFrame[TrackMetadataIndex],
) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
    """Build the ais activity dataset label file.

    This function takes a list of paths to downsampled files and builds a dataframe
    with the paths to the raw data  and activity labels of the files. It performs
    the following steps:
    1. Retrieves and upsamples activity labels from the downsampled track files.
    2. Creates a dataframe with the paths and subpath activity labels.

    Parameters
    ----------
    files : list[str]
        A list of paths to the downsampled track files.
    paths_to_metadate_indexes : list[str]
        A list of paths to the parquet files containing the metadata indexes for the dataset
    Returns
    -------
    DataFrame[ActivityEndOfSequenceLabelDataModel
        A dataframe with the paths and activity labels of the files
    """
    activity_label_df = (
        get_end_of_sequence_activity_labels_from_downsampled_track_files(
            files, metadata_df
        )
    )
    activity_label_df = activity_label_df.dropna()
    # We want to pull this from an annotation ENUM or something
    # we want to drop any track with a row labeled as other
    activity_label_df = drop_rows_with_specific_labels(activity_label_df, ["other"])

    logger.info(f"End of Sequence Activity label df shape: {activity_label_df.shape}")

    #  TODO: deduplication by label consensus Or something that keeps this in mind as well as other
    activity_label_df = activity_label_df.drop_duplicates(
        subset=[
            "trackId",
            "send",
        ],
        keep="first",
    )
    logger.info(
        f"End of Sequence Activity label df shape post deduplication: {activity_label_df.shape}"
    )
    assert (
        activity_label_df.shape[0] > 0
    ), "No End of Sequence Activity labels found post deduplication"
    return activity_label_df


def get_annotation_data_file_paths(
    gcp_bucket_downsampled: str,
    completed_annotations_folder: str,
    exclude_files_in_project: list[str],
) -> list[str]:
    """Get the paths to the annotation data files.

    Parameters
    ----------
    gcp_bucket_downsampled : str
        The name of the GCP bucket containing the downsampled files.
    completed_annotations_folder : str
        The folder in the GCP bucket where the completed annotation files are stored.
    exclude_files_in_project : list[str]
        The list of files to exclude from the dataset.

    Returns
    -------
    list[str]
        The paths to the annotation data files.
    """
    files = list_csv_files_from_bucket(
        gcp_bucket_downsampled, completed_annotations_folder
    )
    logger.info(f"Found {len(files)} files")
    logger.info(files[:10])

    logger.info(exclude_files_in_project)
    ## Remove all Reviewer Rejected or Annotator Submitted Files as well as excluded files
    reviewer_rejected_or_submitted_tajectory_id_project_pairs = (
        get_all_rejected_or_submitted_trajectory_id_project_id_pairs()
    )
    logger.info(
        f"Found {len(reviewer_rejected_or_submitted_tajectory_id_project_pairs)} rejected or submitted trajectory id project id pairs"
    )
    exclude_file_partial = partial(
        exclude_file,
        exclude_strs=exclude_files_in_project,
        excluded_str_pairs=reviewer_rejected_or_submitted_tajectory_id_project_pairs,
    )
    with Pool() as p:
        file_exclusion_mask = list(
            tqdm(
                p.imap(
                    exclude_file_partial,
                    files,
                ),
                total=len(files),
            )
        )
    logger.info(f"Found {len(files)} files post removal of iaa files")
    files = [file for i, file in enumerate(files) if not file_exclusion_mask[i]]
    logger.info(f"Found {len(files)} files post removal of excluded files")
    logger.info(files[:10])
    return files


def plot_geographic_distribution_by_activity(
    activity_label_df: pd.DataFrame, output_dir: str, gcs_bucket: str
) -> None:
    """Plot the distribution of the dataset by activity and save to GCS."""
    for activity in activity_label_df.activity_type_name.unique():
        activity_label_df_activity = activity_label_df[
            activity_label_df.activity_type_name == activity
        ]
        output_dir_activity = Path(f"{output_dir}/{activity}")
        plot_and_save_ais_dataset(
            activity_label_df_activity, output_dir_activity, gcs_bucket
        )
    output_dir_all = Path(f"{output_dir}/all")
    plot_and_save_ais_dataset(activity_label_df, output_dir_all, gcs_bucket)


def create_ais_activity_end_of_seq_dataset() -> None:
    """Create the AIS activity dataset.

    Optionally you can sample non fishing vessel tracks from the raw trackfile data

    -------
    None
        writes the subpath activity label csv to the GCP bucket
    """
    dataset_creation_config = load_dataset_creation_configuration()
    # check the kwargs for the input appropriately
    build_config = dataset_creation_config["build"]
    logger.info("Creating AIS activity dataset")

    ## Load Metadata Indexes

    metadata_df = load_all_metadata_indexes(
        ALL_META_DATA_INDEX_PATHS,
    )
    metadata_df.loc[:, "month"].astype(int)

    path_to_initial_labels = build_config["path_to_initial_labels"]
    if path_to_initial_labels:
        # Load Initial Labels
        logger.info(f"Loading initial labels from {path_to_initial_labels}")
        activity_label_df = pd.read_csv(
            path_to_initial_labels,
            converters={"raw_paths": ast.literal_eval},
        )
        logger.info(
            f"Loaded initial labels from {path_to_initial_labels} as {activity_label_df.head()}"
        )
        label_file_name = path_to_initial_labels.split("/")[-1].split(".")[0]
    elif not build_config["use_human_annotated_data"]:
        logger.info("Building AIS Activity Dataset from Machine Annotations")
        mine_config = build_config["mine"]
        activity_label_df = AISActivityDataMiner.mine_data(metadata_df, **mine_config)
        label_file_name = ("_").join(
            [
                "machine_annotated_activity_labels",
                mine_config["searcher_strategy"],
                mine_config["filter_strategy"],
            ]
        )
    else:
        logger.info("Building AIS Activity Dataset from Human Annotations ")
        files = get_annotation_data_file_paths(
            GCP_BUCKET_DOWNSAMPLED,
            COMPLETED_ANNOTATIONS_FOLDER,
            EXCLUDED_FILE_STRINGS,
        )
        activity_label_df = build_ais_activity_dataset_labels_from_human_annotations(
            files, metadata_df
        )
        assert (
            activity_label_df.shape[0] > 0
        ), f"No subpath activity labels found {activity_label_df.shape}"
        logger.info(f"Subpath activity label df head: {activity_label_df.head()}")

    activity_label_df = ActivityEndOfSequenceLabelDataModel.to_schema().validate(
        activity_label_df
    )
    logger.info(f"Pre subsample columns: {activity_label_df.columns}")
    logger.info(
        f"Activity label value counts pre-subsample: {activity_label_df.activity_type_name.value_counts()}"
    )
    # Subsample
    subsample_config = dataset_creation_config["subsample"]
    subsample_strategy = subsample_config["strategy"]
    if subsample_strategy is not None:
        if subsample_config["use_params"]:
            subsample_kwargs = subsample_config["kwargs"]
        else:
            subsample_kwargs = {}
        activity_label_df = SubsampleEndOfSeqActivityLabels.subsample(
            activity_label_df,
            subsample_strategy,
            **subsample_kwargs,
        )
        label_file_name = label_file_name + f"_subsample_{subsample_strategy}"
    logger.info(
        f"Activity label value counts post-subsample: {activity_label_df.activity_type_name.value_counts()}"
    )
    export_config = dataset_creation_config["export"]
    # overide Default output dir or label_file if specified in config
    output_dir = export_config["output_dir"] or DEFAULT_OUTPUT_DIR
    label_file_name = export_config["label_file_name"] or label_file_name
    export_dataset_to_gcp(
        output_dir,
        activity_label_df,
        label_file_name,
        GCP_BUCKET_NAME_AIS_TRACKS,
    )
    try:
        if len(activity_label_df) < 1000:
            raise ValueError(
                f"Dataset too small plot not useful, only {len(activity_label_df)} samples found"
            )
        plot_geographic_distribution_by_activity(
            activity_label_df, output_dir, GCP_BUCKET_NAME_AIS_TRACKS
        )
    except Exception as e:
        logger.error(
            f"Failed to plot and save geographic distribution {e}", exc_info=True
        )
    upload_config_to_gcp(
        dataset_creation_config,
        GCP_BUCKET_NAME_AIS_TRACKS,
        f"{output_dir}/dataset_creation_config.yaml",
    )


if __name__ == "__main__":
    create_ais_activity_end_of_seq_dataset()
