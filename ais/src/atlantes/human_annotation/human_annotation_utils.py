"""Utility functions for processing human annotation data.

"""

import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from random import shuffle
from typing import Any, Optional, Union

import dask
import dask.config
import pandas as pd
import pandera as pa
from atlantes.elastic_search.elastic_search_utils import (
    TRACK_ANNOTATION_INDEX, TRACK_ANNOTATION_SUMMARY_INDEX, get_es_client)
from atlantes.human_annotation.constants import (
    DT_STRING, EQUIPMENT_ACTIVITY_TYPE_LABELS, PROJECTS_TO_EXCLUDE)
from atlantes.human_annotation.schemas import (
    AnnotatedDownsampledAISTrackDataFrameModel, AnnotationIdentifierDataModel,
    TrackfileDataModel, TrackIdentifierDataModel)
from atlantes.log_utils import get_logger
from atlantes.utils import read_df_file_type_handler
from dask.diagnostics import ProgressBar
from google.cloud import storage
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)


@dataclass
class AnnotationTrajectoryInfo:
    """Class to store information that defines a unique annotation trajectory.

    This is what we compare to see if a trajectory is already in the annotation tool,
    if it hs been annotated by multiple annotators, etc.
    """

    trackId: str
    month: int
    year: int


def sample_from_ais_type(
    metadata_index_df: DataFrame, ais_type: int, num_samples: int
) -> list[str]:
    """Sample from the AIS type.

    Parameters
    ----------
    metadata_index_df : pd.DataFrame
        The metadata index dataframe.
    ais_type : int
        The AIS type, which is the vessel category i.e 70 = cargo, 30 = fishing, etc.

    num_samples : int
        The number of samples to take, if there are less samples \
            than the number of samples requested, it will take all the samples.

    Returns
    -------
    pd.DataFrame
        The sampled dataframe.
    """
    # Sample from the AIS type
    filtered_df = metadata_index_df[metadata_index_df["ais_type"] == str(ais_type)]
    num_samples = min(num_samples, len(filtered_df))
    sampled_df = filtered_df.sample(n=num_samples)
    sampled_paths = sampled_df["Path"].tolist()
    return sampled_paths


def filter_already_sampled_tracks_from_metadata_index(
    metadata_index_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filters already sampled tracks from the metadata index by looking at Elasticsearch.

    Parameters
    ----------
    metadata_index_df : pd.DataFrame
        The metadata index dataframe.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.
    """
    sampled_track_infos = get_all_trajectory_infos_in_annotation_tool()
    logger.debug(sampled_track_infos[:5])
    already_sampled_track_infos = [
        asdict(track_info) for track_info in sampled_track_infos
    ]
    already_sampled_df = pd.DataFrame(already_sampled_track_infos)
    already_sampled_df.loc[:, "month"] = already_sampled_df["month"].astype(int)
    already_sampled_df.loc[:, "year"] = already_sampled_df["year"].astype(int)
    already_sampled_df.loc[:, "trackId"] = already_sampled_df["trackId"].astype(str)
    # Filter already sampled data from the metadata index
    filtered_df = metadata_index_df[
        ~metadata_index_df[["trackId", "month", "year"]]
        .isin(already_sampled_df[["trackId", "month", "year"]])
        .all(axis=1)
    ]
    return filtered_df


def get_all_trajectory_infos_in_annotation_tool(
    return_unique_trajectory_ids: bool = False,
) -> list[AnnotationTrajectoryInfo]:
    """Get all trajectory infos in the annotation tool."""
    # First query all the trajectoy_ids for each unique trackIds
    es_client = get_es_client("production")
    query = {
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {
                    "field": "trajectory_id",
                    "size": 1000000000,  # adjust the size based on your data
                },
            }
        },
    }

    # Execute the query
    response = es_client.search(index=TRACK_ANNOTATION_INDEX, body=query)

    # Trajectory Ids are the names of the files that contain the trajectories We also want to likely keep the track_id otherwise we should just query them directly
    trajectory_ids = [
        trajectory_id_bucket["key"]
        for trajectory_id_bucket in response["aggregations"]["unique_trajectory_ids"][
            "buckets"
        ]
    ]
    logger.info(f"Found {len(trajectory_ids)} unique trajectory ids")
    all_trajectories = []
    for trajectory_id in tqdm(trajectory_ids, desc="getting trajectory info"):
        get_track_id_send_timestamp_query = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trajectory_id": trajectory_id}},
                    ],
                }
            },
        }
        response = es_client.search(
            index=TRACK_ANNOTATION_INDEX, body=get_track_id_send_timestamp_query
        )
        doc = response["hits"]["hits"][0]
        track_id = doc["_source"]["track_id"]
        send_timestamp = pd.to_datetime(doc["_source"]["send_timestamp"])
        month = send_timestamp.month
        year = send_timestamp.year
        trajectory_info = AnnotationTrajectoryInfo(
            trackId=track_id, month=month, year=year
        )
        if not return_unique_trajectory_ids:
            all_trajectories.append(trajectory_info)
        else:
            if trajectory_info not in all_trajectories:
                all_trajectories.append(trajectory_info)

    # Should this be a dataframe?
    return all_trajectories


def get_trajectory_info(
    path: str,
) -> dict:
    df = pd.read_csv(path, nrows=1)
    # Turn send to pd.Timestamp
    df.loc[:, "send"] = pd.to_datetime(df["send"])
    return get_unique_identifiers_from_df(df)


@pa.check_types
def get_all_trajectory_infos_for_paths(
    paths: list[str],
) -> DataFrame[TrackIdentifierDataModel]:
    """Get all trajectory infos for a list of paths."""
    all_trajectories = [dask.delayed(get_trajectory_info)(path) for path in paths]
    with ProgressBar():
        all_trajectories = dask.compute(*all_trajectories)

    logger.info(f"Found {pd.DataFrame(all_trajectories)} unique trajectory ids")
    return pd.DataFrame(all_trajectories)


@pa.check_types
def is_track_equipment(
    downsampled_df: DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
) -> bool:
    """Check if the track is labeled as equipment."""
    object_type_name = downsampled_df["object_type_name"].unique()
    is_object_type_name_equipment = object_type_name[0] == "equipment"
    does_it_have_equipment_activity_type = any(
        activity in downsampled_df["activity_type_name"].unique()
        for activity in EQUIPMENT_ACTIVITY_TYPE_LABELS
    )
    return is_object_type_name_equipment or does_it_have_equipment_activity_type


def read_downsampled_track_handler(
    downsampled_track_path: str, **kwargs: Any
) -> Optional[AnnotatedDownsampledAISTrackDataFrameModel]:
    """Reads a downsampled track from gcs and handles the case where the track is not annotated."""
    try:
        return load_downsampled_track_df(downsampled_track_path, **kwargs)
    except SchemaError as e:
        df = pd.read_csv(downsampled_track_path, nrows=1)
        if df["annotator_uid"].isna().any():
            logger.error(f"Track {downsampled_track_path} has no annotator_uid")
            return None
        else:
            logger.error(
                f"Error reading track: {downsampled_track_path}: {e}", exc_info=True
            )
            raise e
    except Exception as e:
        logger.error(
            f"Error reading track: {downsampled_track_path}: {e}", exc_info=True
        )
        raise e


@pa.check_types
def load_downsampled_track_df(
    path: str,
    dtypes: dict = {"activity_type_confidence": float, "object_type_confidence": float},
    parse_dates: list[str] = ["send"],
    **kwargs: Any,
) -> DataFrame[AnnotatedDownsampledAISTrackDataFrameModel]:
    """Load the downsampled track file."""
    logger.debug(f"Loading downsampled track file from {path}")
    kwargs["dtype"] = dtypes
    kwargs["parse_dates"] = parse_dates
    downsampled_df = pd.read_csv(path, **kwargs)
    return downsampled_df


@pa.check_types
def load_non_downsampled_track_df(
    path: str, **kwargs: Any
) -> DataFrame[TrackfileDataModel]:
    """Get the non-downsampled track dataframe."""
    try:
        return read_df_file_type_handler(path, **kwargs)
    except Exception as e:
        logger.error(
            f"Error loading non-downsampled track file: {path}: {e}", exc_info=True
        )
        raise e


def find_first_prev_num_not_in_list(num: int, num_list: list[int]) -> int:
    """Find the first previous number not in a list."""
    while num in num_list:
        num -= 1
    return num


def get_track_id_from_df(
    df: pd.DataFrame,
) -> str:
    """Get the track id from a dataframe."""
    return df.iloc[0]["trackId"]


def get_month_from_df(df: pd.DataFrame) -> int:
    """Get the month from a dataframe."""
    return df.iloc[0]["send"].month


def format_blob_name_into_gcp_path(bucket_name: str, blob_name: str) -> str:
    """Format a blob name into a GCP path."""
    return f"gs://{bucket_name}/{blob_name}"


def get_unique_identifiers_from_df(
    annotated_df: DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
) -> dict:
    """Get unique identifiers from a dataframe."""
    return dict(
        trackId=annotated_df.iloc[0]["trackId"],
        month=annotated_df.iloc[0]["send"].month,
        year=annotated_df.iloc[0]["send"].year,
        annotator_username=annotated_df.iloc[0]["annotator_username"],
    )


def get_unique_trajectory_in_project(
    bucket_name: str,
    client: storage.Client,
    prefix_dir: str,
    project: str,
) -> DataFrame[AnnotationIdentifierDataModel]:
    """Get all gold standard trajectories unique identifiers.

    Annotations are uniquely Identified by year, month, trackId and annotator_username
    """
    gcp_bucket = client.bucket(bucket_name)
    blobs = gcp_bucket.list_blobs(
        prefix=f"{prefix_dir}/{project}",
    )
    unique_identifiers: list[dict] = []
    for blob in blobs:
        # check if blob is a file
        logger.info(f"{blob.name=}")
        path = format_blob_name_into_gcp_path(bucket_name, blob.name)
        if not path.endswith((".csv", ".parquet")):
            continue
        df = read_downsampled_track_handler(path)
        if df is None:
            continue
        unique_ids_dict = get_unique_identifiers_from_df(df)
        unique_ids_dict.update(path=path)
        unique_identifiers.append(
            unique_ids_dict,
        )
    return pd.DataFrame.from_records(unique_identifiers)


@pa.check_types
def get_all_unique_trajectories_identities_in_projects(
    projects: list[str],
    client: storage.Client,
    bucket_name: str,
    prefix_dir: str,
) -> DataFrame[AnnotationIdentifierDataModel]:
    """Get all gold standard trajectories unique identifiers.

    Annotations are uniquely Identified by year, month, trackId and annotator_username
    """
    logger.info(f"{bucket_name=} {prefix_dir=} {projects=}")
    return pd.concat(
        [
            get_unique_trajectory_in_project(bucket_name, client, prefix_dir, project)
            for project in projects
        ]
    )


def check_projects_to_exclude(project: str) -> bool:
    """Check if any of the strings in projects to exclude are in a given string."""
    project_lower = project.lower()
    for exclude_project_phrase in PROJECTS_TO_EXCLUDE:
        if exclude_project_phrase.lower() in project_lower:
            return True
    return False


def find_first_next_num_not_in_list(num: int, num_list: list[int]) -> int:
    """Find the first next number not in a list."""
    while num in num_list:
        num += 1
    return num


def is_path_in_gold_standard_projects(
    path: str, gold_standard_projects: list[str]
) -> bool:
    """Check if a path is in the gold standard projects."""
    return any(project in path for project in gold_standard_projects)


def build_set_of_trackid_month_pairs_from_downsampled_track_paths(
    track_paths: Union[set[str], list[str]]
) -> set[tuple[str, int]]:
    """Build a set of trackID and month pairs.

    Parameters
    ----------
    track_paths : Union[set[str], list[str]]
        A set or list of paths to the track files.

    Returns
    -------
    set[tuple[str, int]]
        A set of tuples with the trackID and month.

    Raises
    ------
    TypeError
        If track_paths is not a set or list.

    """
    track_id_months = set()
    for path in track_paths:
        downsampled_track_df = load_downsampled_track_df(path)
        track_id = get_track_id_from_df(downsampled_track_df)
        month = get_month_from_df(downsampled_track_df)
        track_id_months.add((track_id, month))

    return track_id_months


# THis function is likely repetitive
def build_list_of_trackid_month_pairs_from_downsampled_track_paths(
    track_paths: Union[set[str], list[str]]
) -> list[tuple[str, int]]:
    """Build a list of trackID and month pairs.


    Parameters
    ----------
    track_paths : Union[set[str], list[str]]
        a list of the paths to the track files

    Returns
    -------
    list[tuple[str, int]]
        a list of tuples with the trackID and month

    """
    track_id_months = []
    for path in track_paths:
        downsampled_track_df = load_downsampled_track_df(path)
        track_id = get_track_id_from_df(downsampled_track_df)
        month = get_month_from_df(downsampled_track_df)
        track_id_months.append((track_id, month))

    return track_id_months


def is_track_in_list(
    candidate_track_path: str, track_id_months: list[tuple[str, int]]
) -> bool:
    """Check if a track is a duplicate of a gold standard track.

    Tracks are defined by a trackID and a month. So if the same trackID is found in the same month, it is a duplicate.
    Parameters
    ----------
    candidate_track_path : str
        the path to the candidate track file
    track_id_months : list[tuple[str, int]]
        a list of tuples with the trackID and month

    Returns
    -------
    bool
        a boolean indicating whether the track is a duplicate of a gold standard track
    """
    track_df = load_non_downsampled_track_df(candidate_track_path)
    track_id, month = get_track_id_from_df(track_df), get_month_from_df(track_df)
    return (track_id, month) in track_id_months


def write_file_names_to_project_txts(
    paths_to_track_files: list[str], project_size: int, output_directory: str
) -> None:
    """Write file names into project txts

    Parameters
    ----------
    paths_to_track_files : list[str]
        List of paths to track files

    project_size : int
        Number of files to put in each project

    output_directory : str
        Directory to write the project txt files

    Returns
    -------
    None
        writes each project to a .txt file in the output directory
    """

    def write_to_project_file(
        project_num: int, paths_to_track_files: list[str], output_directory: str
    ) -> None:
        """Write to a single project file

        Parameters
        ----------
        project_num : int
            Project number
        paths_to_track_files : list[str]
            List of paths to track files
        output_directory : str
            Directory to write the project txt files
        """
        os.makedirs(output_directory, exist_ok=True)
        out_file = f"{output_directory}/sampled_trajectories_human_annotate{DT_STRING}_{project_num}.txt"
        with open(out_file, "a") as f:
            for sample in paths_to_track_files:
                f.write(f"{sample}\n")

    shuffle(paths_to_track_files)

    # divide files into project size file chunks
    for i in range(0, len(paths_to_track_files), project_size):
        project_num = i // project_size
        project_chunk = paths_to_track_files[i : i + project_size]
        write_to_project_file(project_num, project_chunk, output_directory)


def get_trajectory_id_project_id_pairs() -> list[tuple[str, str]]:
    """Get all trajectory id and project id pairs."""
    es_client = get_es_client("production")
    query = {
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {"field": "trajectory_id", "size": 100000},
                "aggs": {
                    "project_ids": {"terms": {"field": "project_id", "size": 100000}}
                },
            }
        },
    }
    res = es_client.search(index=TRACK_ANNOTATION_INDEX, body=query)
    project_id_trajectory_id_pairs = []
    for record in res["aggregations"]["unique_trajectory_ids"]["buckets"]:
        trajectory_id = record["key"]
        project_id_trajectory_id_pairs.extend(
            [
                (trajectory_id, project["key"])
                for project in record["project_ids"]["buckets"]
                if project["key"] is not None
            ]
        )
    return project_id_trajectory_id_pairs


def check_if_project_has_given_feedback_status(id: str, feedback_status: str) -> bool:
    """Check if a track in A project has accepted feedback"""
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"id": id}},
                    # {"term": {"status": accepted_status}},
                ],
            }
        }
    }
    es_client = get_es_client("production")
    res = es_client.search(index=TRACK_ANNOTATION_SUMMARY_INDEX, body=query)
    status = res["hits"]["hits"][0]["_source"].get("status", None)
    logger.info(f"Status of project {id} is {status}")
    return status == feedback_status


def get_summary_id_from_project_id_trajectory_id_pair(
    trajectory_id: str, project_id: str
) -> str:
    """Get the summary id from a project_id and trajectory_id"""
    es_client = get_es_client("production")
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"project_id": project_id}},
                    {"term": {"trajectory_id": trajectory_id}},
                ],
            }
        }
    }
    res = es_client.search(index=TRACK_ANNOTATION_SUMMARY_INDEX, body=query)
    return res["hits"]["hits"][0]["_id"]


def is_trajectory_rejected_or_submitted_for_approval(pair: tuple[str, str]) -> bool:
    """Check if a trajectory is rejected or submitted for approval."""
    summary_id = get_summary_id_from_project_id_trajectory_id_pair(*pair)
    return check_if_project_has_given_feedback_status(
        summary_id, "REVIEWER-REJECTED"
    ) or check_if_project_has_given_feedback_status(summary_id, "ANNOTATOR-SUBMITTED")


def get_all_rejected_or_submitted_trajectory_id_project_id_pairs() -> (
    list[tuple[str, str]]
):
    """Get all rejected trajectory infos in the annotation tool from ES

    Note trajectory id project id paris are 995 likely to be unique but could not be if for e.g.
    the same track Id in the same month and different years is in the same project
    Trajectory Ids should be unique because of this
    """
    # First query all the trajectoy_ids for each unique trackIds
    all_project_trajectory_id_pairs = get_trajectory_id_project_id_pairs()
    logger.info(
        f"Found {len(all_project_trajectory_id_pairs)} unique trajectory id and project id pairs"
    )

    with Pool() as p:
        rejected_track_mask = list(
            tqdm(
                p.imap(
                    is_trajectory_rejected_or_submitted_for_approval,
                    all_project_trajectory_id_pairs,
                ),
                total=len(all_project_trajectory_id_pairs),
            )
        )
    total_number_rejected = sum(rejected_track_mask)
    logger.info(
        f"Found {total_number_rejected} rejected or submitted for approval trajectory id and project id pairs"
    )
    rejected_track_infos = [
        all_project_trajectory_id_pairs[i]
        for i, rejected in enumerate(rejected_track_mask)
        if rejected
    ]
    return rejected_track_infos
