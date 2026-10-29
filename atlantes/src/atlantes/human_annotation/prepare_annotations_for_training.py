"""Module for transforming Labeled Downsampled Trackfiles into samples for training.

# TODO: add pandera as needed
# TODO: Give Clearer task specifc file name"""

import dask
import dask.config
import pandas as pd
from atlantes.human_annotation.human_annotation_utils import (
    get_all_trajectory_infos_for_paths, is_track_equipment,
    load_downsampled_track_df)
from atlantes.human_annotation.schemas import (TrackIdentifierDataModel,
                                               TrackMetadataIndex)
from atlantes.log_utils import get_logger
from dask.diagnostics import ProgressBar
from pandera.typing import DataFrame

logger = get_logger(__name__)


def get_activity_labels_from_downsampled_track_df(
    raw_paths: list[str],
    downsampled_path: str,
) -> pd.DataFrame:
    """Get the subpath activity labels from the downsampled track df.

    Parameters
    ----------
    path : str
        Path to the downsampled track df.
    metadata_index : DataFrame[TrackMetadataIndex]
        Metadata index.

    Returns
    -------
    Optional[Tuple[str, list[int]]]
        Tuple of the path to the raw track df and the list of subpath activity labels.
    """
    downsampled_df = load_downsampled_track_df(downsampled_path)
    if is_track_equipment(downsampled_df):
        return pd.DataFrame()
    label_information_df = downsampled_df[
        ["trackId", "send", "activity_type_name", "lat", "lon"]
    ].copy()
    # addd the downsampled_path and the raw_paths as columns
    label_information_df = label_information_df.assign(
        downsampled_path=downsampled_path
    )
    label_information_df.loc[:, "raw_paths"] = [
        raw_paths for _ in range(len(label_information_df))
    ]
    return label_information_df


def pull_previous_months_context(
    annotated_df: DataFrame[TrackIdentifierDataModel],
    metadata_index: DataFrame[TrackMetadataIndex],
    prior_months_of_context: int,
) -> list[pd.Series]:
    """
    Pulls the previous months' context for the given annotated DataFrame.

    Args:
        annotated_df (pd.DataFrame): The annotated DataFrame.
        - month, year, trackId is needed
        metadata_index (DataFrame[TrackMetadataIndex]): The metadata index DataFrame.
        prior_months_of_context (int): The number of prior months' context to pull.

    Returns:
        list[pd.Series]: A list of Series containing the previous months' context.

    """
    prior_months_context_df_list = []
    for i in range(prior_months_of_context):
        annotated_df_prior_month = annotated_df.copy()
        annotated_df_prior_month.loc[:, "month"] = (
            annotated_df_prior_month["month"] - i - 1
        )
        # if month is less than 1, then we need to subtract a year and add 12 months
        annotated_df_prior_month.loc[annotated_df_prior_month.month < 1, "year"] = (
            annotated_df_prior_month[annotated_df_prior_month.month < 1]["year"] - 1
        )
        annotated_df_prior_month.loc[annotated_df_prior_month.month < 1, "month"] = (
            annotated_df_prior_month[annotated_df_prior_month.month < 1]["month"] + 12
        )
        # Pull the file_path from the metadata_index matching the info of the prior month
        # ensure both month and year are integers and trackId is a string for both dataframes
        annotated_df_prior_month = annotated_df_prior_month.astype(
            {"month": int, "year": int, "trackId": str}
        ).copy()
        metadata_index = metadata_index.astype(
            {"month": int, "year": int, "trackId": str})
        raw_and_downsampled_df_prior_month = annotated_df_prior_month.merge(
            metadata_index, on=["trackId", "month", "year"], how="left"
        )
        prior_months_context_df_list.append(raw_and_downsampled_df_prior_month["Path"])
    return prior_months_context_df_list


def build_raw_paths_context_column(
    prior_months_context: list[pd.Series], current_month_context_df: pd.DataFrame
) -> pd.Series:
    """Create a column of raw paths context from a list of series of prior months context"""
    context_paths_df = current_month_context_df[["Path"]].copy()
    raw_path_col_names = ["Path"]
    for i, context_paths in enumerate(prior_months_context):
        new_col_name = f"Path_{i}"
        context_paths_df.loc[:, new_col_name] = context_paths
        raw_path_col_names.append(new_col_name)
    # Combine the context paths into a single column with Path
    # Reverse the path col_names so that the context is in chronological order
    raw_path_col_names = raw_path_col_names[::-1]
    return context_paths_df[raw_path_col_names].apply(
        turn_row_into_list_with_no_nan, axis=1
    )


def turn_row_into_list_with_no_nan(row: pd.Series) -> list[str]:
    """Turn a row into a list with no NaN values."""
    return row.dropna().tolist()


def find_raw_paths_from_downsampled_paths(
    files: list[str],
    metadata_index: DataFrame[TrackMetadataIndex],
    prior_months_of_context: int = 1,
) -> list[tuple[str, str]]:
    """Find the raw path from the downsampled path.
    TODO: find multiple paths of context for each downsampled path


    Assumptions and Set up that are needed for this function:
    1. Data files are uniquely defined by trackId, month, and year

    Parameters
    ----------
    files : list[str]
        List of paths to the downsampled files.
    metadata_index : DataFrame[TrackMetadataIndex]
        Metadata index.
    prior_months_of_context : int
        Number of months of context to include before the downsampled path.

    Returns
    -------
    list[tuple[str, str]]
        List of tuples of the raw path and the downsampled path.
    """

    annotated_df = get_all_trajectory_infos_for_paths(files)
    annotated_df.loc[:, "downsampled_path"] = files
    logger.info(f"{annotated_df.head()}")
    raw_and_downsampled_df = annotated_df.merge(
        metadata_index, on=["trackId", "month", "year"]
    )
    track_identication_info_df = raw_and_downsampled_df[
        ["trackId", "month", "year"]
    ].copy()
    prior_months_context_df_list = pull_previous_months_context(
        track_identication_info_df, metadata_index, prior_months_of_context
    )

    context_paths_df = raw_and_downsampled_df[["Path", "downsampled_path"]]

    context_paths_df.loc[:, "raw_paths"] = build_raw_paths_context_column(
        prior_months_context_df_list, context_paths_df
    )

    return context_paths_df[["raw_paths", "downsampled_path"]].values.tolist()


# Orchestrator function we would want to do pandera typing guarentees on that
def get_end_of_sequence_activity_labels_from_downsampled_track_files(
    files: list[str], metadata_index: DataFrame[TrackMetadataIndex]
) -> pd.DataFrame:
    """Get the activity labels from downsampled track files.
    Parameters
    ----------
    files : list[Path]
        list of paths to the downsampled files
    Returns
    -------
    Tuple[str, list[int]]
        a tuple of the path and the activity labels
    """
    path_tuples_lst = find_raw_paths_from_downsampled_paths(files, metadata_index)

    with ProgressBar(), dask.config.set(scheduler="synchronous"):
        tasks = [
            dask.delayed(get_activity_labels_from_downsampled_track_df)(*path_tuple)
            for path_tuple in path_tuples_lst
        ]
        label_information_df_lst = dask.compute(*tasks)
    activity_label_df = pd.concat(label_information_df_lst, ignore_index=True)
    return activity_label_df
