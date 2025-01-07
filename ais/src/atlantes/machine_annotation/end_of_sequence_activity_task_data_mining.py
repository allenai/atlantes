"""Module to Mine for and Machine Annotate Messages based on Metadata and SME for Training"""

from abc import ABC, abstractmethod

import dask
import pandas as pd
import pandera as pa
from atlantes.atlas.atlas_utils import (AtlasActivityLabelsTraining,
                                        preprocess_trackfile)
from atlantes.atlas.schemas import (
    ActivityEndOfSequenceLabelDataModel,
    ActivityEndOfSequenceLabelNoContextDataModel)
# Should this be in annotation utils outside of human annotation utils
from atlantes.human_annotation.prepare_annotations_for_training import (
    build_raw_paths_context_column, pull_previous_months_context)
from atlantes.human_annotation.schemas import TrackMetadataIndex
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import (
    ANCHORED_MAX_SOG_METERS_PER_SECOND, ANCHORED_NAV_STATUS,
    KNOWN_NON_FISHING_VESSEL_CATEGORIES, MOORED_MAX_SOG_METERS_PER_SECOND,
    MOORED_NAV_STATUS, TRANSITING_MAX_MIDSPEED_SOG_METERS_PER_SECOND,
    TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND,
    TRANSITING_MIN_MIDSPEED_SOG_METERS_PER_SECOND)
from atlantes.utils import batch, read_df_file_type_handler
from dask.delayed import delayed
from dask.diagnostics import ProgressBar
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)


class BaseMetadataFilterStrategy(ABC):
    @staticmethod
    @abstractmethod
    def filter(
        metadata_df: DataFrame[TrackMetadataIndex],
    ) -> list[str]:
        """Abstract method to filter files in a metadataindex into a subset of paths"""
        pass


class NonFishingVesselFilter(BaseMetadataFilterStrategy):
    @staticmethod
    def filter(
        metadata_df: pd.DataFrame,
    ) -> list[str]:
        """Filter files to only include non-fishing vessels."""
        """Filter files to only include non-fishing vessels."""
        logger.info("Filtering files to only include non-fishing vessels.")
        metadata_df.loc[:, "ais_type"] = metadata_df["ais_type"].astype(int)
        file_paths_list = metadata_df[
            metadata_df["ais_type"].isin(KNOWN_NON_FISHING_VESSEL_CATEGORIES)
        ].Path.tolist()
        logger.info(f"Number of files after filtering: {len(file_paths_list)}")
        return file_paths_list


class NullFilter(BaseMetadataFilterStrategy):
    @staticmethod
    def filter(
        metadata_df: pd.DataFrame,
    ) -> list[str]:
        return metadata_df.Path.tolist()


class MetadataFilter:
    """Class to execute a filter strategy on metadata."""

    strategy_map = {
        "non_fishing_vessels": NonFishingVesselFilter,
        "null_filter": NullFilter,
        # Add other strategies here
    }

    @classmethod
    def execute_filter(cls, strategy_key: str, metadata_df: pd.DataFrame) -> list[str]:
        if strategy_key not in cls.strategy_map:
            raise ValueError(f"Unknown strategy key: {strategy_key}")
        filter_strategy = cls.strategy_map[strategy_key]
        return filter_strategy.filter(metadata_df)


# TODO: Searchers should go in a separate file so should filters
class BaseSearchStrategy(ABC):
    """Base class for search strategies."""

    ACTIVITY_LABEL_NAME_TO_ASSIGN: str = "NAN"

    @classmethod
    @abstractmethod
    def search_strategy(
        cls,
        preprocessed_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Abstract method to search a file for messages to machine annotate."""
        pass

    @classmethod
    def search(
        cls,
        file_path: str,
        min_trajectory_length: int,
        max_num_messages_to_label: int,
    ) -> pd.DataFrame:
        """Abstract method to search a file for messages to machine annotate."""
        preprocessed_df = cls.read_and_preprocess(file_path)
        if len(preprocessed_df) < min_trajectory_length:
            return pd.DataFrame()
        messages_to_label_df = cls.search_strategy(preprocessed_df)
        if len(messages_to_label_df) == 0:
            logger.info("No midspeed transiting messages found, returning empty df")
            # All output dfs will be concatenated so we need to return a df
            return pd.DataFrame()

        subsampled_mid_speed_transiting_df = cls.sample_up_to_max_messages(
            messages_to_label_df, max_num_messages_to_label
        )
        label_information_df = cls.assign_labels(
            subsampled_mid_speed_transiting_df,
            cls.ACTIVITY_LABEL_NAME_TO_ASSIGN,
            file_path,
        )
        logger.info(
            f"Found {len(label_information_df)} messages to machine annotate with mid-speed transiting"
        )
        return label_information_df

    @classmethod
    def read_and_preprocess(cls, file_path: str) -> pd.DataFrame:
        """Read and preprocess a file. This is a common operation for all search strategies."""
        df = read_df_file_type_handler(file_path)
        df.loc[:, "send"] = pd.to_datetime(df["send"])
        preprocessed_df = preprocess_trackfile(df)
        return preprocessed_df

    @classmethod
    def sample_up_to_max_messages(
        cls,
        messages_df: pd.DataFrame,
        max_num_messages_to_label: int,
    ) -> pd.DataFrame:
        """Sample messages to machine annotate."""
        max_num_messages_to_label = min(max_num_messages_to_label, len(messages_df))
        subsampled_df = messages_df.sample(max_num_messages_to_label)
        return subsampled_df

    @classmethod
    def assign_labels(
        cls, messages_df: pd.DataFrame, activity_type_name: str, file_path: str
    ) -> pd.DataFrame:
        """Assign activity type name and path to the labels."""
        label_information_df = messages_df[["send", "trackId", "lat", "lon"]].copy()
        label_information_df.loc[:, "activity_type_name"] = activity_type_name.lower()
        label_information_df.loc[:, "Path"] = file_path
        return label_information_df


class HighSpeedTransitingSearch(BaseSearchStrategy):
    """Search a file to find messages where the vessel may be transiting at high speed.

    Only confident for non fishing vessels.
    """

    ACTIVITY_LABEL_NAME_TO_ASSIGN: str = AtlasActivityLabelsTraining.TRANSITING.name

    @classmethod
    def search_strategy(
        cls,
        preprocessed_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Search a file to find messages where the vessel may be transiting at high speed.

        Criteria:
        - speed over ground is greater than or equal to TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND
        """
        high_speed_transiting_messages_df = preprocessed_df[
            preprocessed_df["sog"]
            >= TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND
        ]
        return high_speed_transiting_messages_df


class MidSpeedTransitingSearch(BaseSearchStrategy):
    """Search a file to find messages where the vessel may be transiting at mid-speed.

    Only confident for non fishing vessels.
    """

    ACTIVITY_LABEL_NAME_TO_ASSIGN: str = AtlasActivityLabelsTraining.TRANSITING.name

    @classmethod
    def search_strategy(
        cls,
        preprocessed_df: pd.DataFrame,
    ) -> pd.DataFrame:
        mid_speed_transiting_messages_df = preprocessed_df[
            preprocessed_df["sog"] >= TRANSITING_MIN_MIDSPEED_SOG_METERS_PER_SECOND
        ]
        mid_speed_transiting_messages_df = mid_speed_transiting_messages_df[
            preprocessed_df["sog"] <= TRANSITING_MAX_MIDSPEED_SOG_METERS_PER_SECOND
        ]
        return mid_speed_transiting_messages_df


class AnchoredSearch(BaseSearchStrategy):
    """Search a file to find messages where the vessel is anchored."""

    ACTIVITY_LABEL_NAME_TO_ASSIGN: str = AtlasActivityLabelsTraining.ANCHORED.name

    @classmethod
    def search_strategy(
        cls,
        preprocessed_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Search a file to find messages where the vessel is anchored.

        Criteria:
        - nav status is anchored
        - speed over ground is less than or equal to ANCHORED_MAX_SOG_METERS_PER_SECOND
        """
        anchored_nav_filt = preprocessed_df["nav"] == ANCHORED_NAV_STATUS
        anchored_sog_filt = preprocessed_df["sog"] <= ANCHORED_MAX_SOG_METERS_PER_SECOND
        anchored_messages_df = preprocessed_df[anchored_nav_filt & anchored_sog_filt]
        return anchored_messages_df


class MooredSearch(BaseSearchStrategy):
    """Search a file to find messages where the vessel is moored."""

    ACTIVITY_LABEL_NAME_TO_ASSIGN: str = AtlasActivityLabelsTraining.MOORED.name

    @classmethod
    def search_strategy(
        cls,
        preprocessed_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Search a file to find messages where the vessel is moored.

        Criteria:
        - nav status is moored
        - speed over ground is less than or equal to MOORED_MAX_SOG_METERS_PER_SECOND
        """
        moored_nav_filt = preprocessed_df["nav"] == MOORED_NAV_STATUS
        moored_sog_filt = preprocessed_df["sog"] <= MOORED_MAX_SOG_METERS_PER_SECOND
        moored_messages_df = preprocessed_df[moored_nav_filt & moored_sog_filt]
        return moored_messages_df


class Searcher:
    """Class to execute a search strategy on a file."""

    strategy_map = {
        "high_speed_transiting": HighSpeedTransitingSearch,
        "mid_speed_transiting": MidSpeedTransitingSearch,
        "anchored": AnchoredSearch,
        "moored": MooredSearch,
        # Add other strategies here
    }

    @classmethod
    def execute_search(
        cls,
        strategy_key: str,
        file_path: str,
        min_trajectory_length: int,
        max_num_messages_to_label: int = 100,
    ) -> pd.DataFrame:
        """Execute a search strategy on a file."""
        if strategy_key not in cls.strategy_map:
            raise ValueError(f"Unknown strategy key: {strategy_key}")
        search_strategy = cls.strategy_map[strategy_key]
        return search_strategy.search(
            file_path, min_trajectory_length, max_num_messages_to_label
        )


class AISActivityDataMiner:
    @classmethod
    def distributed_acquisition(
        cls,
        file_paths: list[str],
        num_samples_per_file: int,
        min_trajectory_length: int,
        searcher_strategy_key: str,
        chunk_size: int = 10000,
    ) -> list[DataFrame[ActivityEndOfSequenceLabelNoContextDataModel]]:
        """Acquire information from files using the searcher

        This function uses dask to parallelize the search across multiple files.

        Parameters
        ----------
        file_paths : list[str]
            List of file paths to search
        num_samples_per_file : int
            Number of samples to take from each file
        min_trajectory_length : int
            Minimum length of a trajectory to consider
        searcher_strategy_key : str
            Key for the searcher strategy to use
        chunk_size : int
            Number of file paths to process in each chunk

        Returns
        -------
        list[DataFrame[ActivityEndOfSequenceLabelNoContextDataModel]]
            List of dataframes containing the information from the search
        """

        logger.info("Acquiring information from files Distributed")
        final_results = []
        total_num_chunks = len(file_paths) // chunk_size
        for file_path_chunk in tqdm(
            batch(file_paths, chunk_size),
            desc="File Path Chunks",
            total=total_num_chunks,
        ):
            with ProgressBar():
                tasks = [
                    delayed(Searcher.execute_search)(
                        searcher_strategy_key,
                        file_path,
                        num_samples_per_file,
                        min_trajectory_length,
                    )
                    for file_path in file_path_chunk
                ]
                results = dask.compute(*tasks)
                final_results.extend(results)
        return final_results

    @classmethod
    def synchronous_acquisition(
        cls,
        file_paths: list[str],
        num_samples_per_file: int,
        min_trajectory_length: int,
        searcher_strategy_key: str,
    ) -> list[DataFrame[ActivityEndOfSequenceLabelNoContextDataModel]]:
        """Acquire information from files using the searcher"""
        logger.info("Acquiring information from files Synchronously")
        results = []
        for file_path in file_paths:
            results.append(
                Searcher.execute_search(
                    searcher_strategy_key,
                    file_path,
                    num_samples_per_file,
                    min_trajectory_length,
                )
            )
        return results

    @classmethod
    @pa.check_types(head=5)
    def collate_information(
        cls,
        machine_annotated_messages_df: list[
            DataFrame[ActivityEndOfSequenceLabelNoContextDataModel]
        ],
        metadata_df: DataFrame[TrackMetadataIndex],
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """Collate information from the search results.

        This function also finds the files the previous months files
          to provide context leading up to the activity."""
        activity_label_df = pd.concat(machine_annotated_messages_df).reset_index(
            drop=True
        )
        logger.info(f"Number of messages machine annotated: {len(activity_label_df)}")
        # Make a month and year column
        activity_label_df.loc[:, "month"] = activity_label_df["send"].apply(
            lambda x: x.month
        )
        activity_label_df.loc[:, "year"] = activity_label_df["send"].apply(
            lambda x: x.year
        )
        track_identication_info_df = activity_label_df[
            ["trackId", "month", "year"]
        ].copy()
        context_paths_series_lst = pull_previous_months_context(
            track_identication_info_df, metadata_df, 1
        )
        context_paths_df = activity_label_df[["Path"]].copy()
        activity_label_df.loc[:, "raw_paths"] = build_raw_paths_context_column(
            context_paths_series_lst, context_paths_df
        )
        activity_label_df = activity_label_df.drop(columns=["Path"])
        return activity_label_df

    @classmethod
    @pa.check_types(head=5)
    def mine_data(
        cls,
        metadata_df: DataFrame[TrackMetadataIndex],
        min_trajectory_length: int,
        num_samples_per_file: int,
        sample_pool_size: int,
        filter_strategy: str,
        searcher_strategy: str,
        debug_mode: bool,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """Mine data using the specified searcher and filter object."""
        sample_pool_size = min(sample_pool_size, len(metadata_df))
        metadata_df = metadata_df.sample(sample_pool_size)  # Randomly sample
        file_paths = MetadataFilter.execute_filter(filter_strategy, metadata_df)
        machine_annotated_messages = (
            cls.distributed_acquisition(
                file_paths,
                num_samples_per_file,
                min_trajectory_length,
                searcher_strategy,
            )
            if not debug_mode
            else cls.synchronous_acquisition(
                file_paths,
                num_samples_per_file,
                min_trajectory_length,
                searcher_strategy,
            )
        )
        activity_label_df = cls.collate_information(
            machine_annotated_messages, metadata_df
        )
        return activity_label_df
