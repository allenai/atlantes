"""Module for subsampling the end of sequence activity dataset."""

from typing import Any, Callable, Generator

import numpy as np
import pandas as pd
from atlantes.atlas.schemas import ActivityEndOfSequenceLabelDataModel
from atlantes.log_utils import get_logger
from atlantes.utils import filter_kwargs
from pandera.typing import DataFrame
from tqdm import tqdm

logger = get_logger(__name__)


class SubsampleEndOfSeqActivityLabels:
    """
    Class for subsampling the end of sequence activity dataset.
    """

    @classmethod
    def compile_subsampled_dataframes(
        cls, dataframe_list: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Concatenates a list of dataframes into a single dataframe and cleans up the index.

        Parameters
        ----------
        dataframe_list : list[pd.DataFrame]
            A list of dataframes to concatenate.

        Returns
        -------
        pd.DataFrame
            The concatenated dataframe.
        """
        labels_df = pd.concat(dataframe_list).reset_index(drop=True)
        if "Unnamed: 0" in labels_df.columns:
            labels_df.drop("Unnamed: 0", axis=1, inplace=True)
        return labels_df

    @classmethod
    def track_id_iterator(
        cls, dataset: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> Generator[str, None, None]:
        """
        Iterates over the unique trackIds in the dataset.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to iterate over.

        Returns
        -------
        Generator
            A generator that yields the unique trackIds.
        """
        unique_track_ids = dataset.trackId.unique()
        for id in tqdm(unique_track_ids, desc="Subsampling"):
            yield id

    @classmethod
    def filter_to_sorted_single_track_id(
        cls, dataset: DataFrame[ActivityEndOfSequenceLabelDataModel], track_id: str
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Filters the dataset to a single trackId and sorts it by the send column.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to filter.

        Returns
        -------
        DataFrame
            The filtered dataset.
        """
        is_track_id = dataset.trackId == track_id
        return dataset[is_track_id].sort_values("send").reset_index(drop=True)

    @classmethod
    def get_activity_change_indices(
        cls, single_track_labels_df: pd.DataFrame
    ) -> np.ndarray:
        """Return the indices of the rows where the activity changes.

        Given a sorted DataFrame of a ais track with a single trackId,

        Parameters
        ----------
        single_track_labels_df : pd.DataFrame
            The DataFrame of a single trackId.

        Returns
        -------
        np.ndarray
            The indices of the rows where the activity changes."""
        if single_track_labels_df.trackId.nunique() != 1:
            raise ValueError("trackId column must be unique")

        single_track_labels_df.sort_values("send", inplace=True)
        activity_change_filter = (
            single_track_labels_df.activity_type_name
            != single_track_labels_df.activity_type_name.shift(-1)
        )
        activity_change_indices = single_track_labels_df[
            activity_change_filter
        ].index.values
        return activity_change_indices

    @classmethod
    def subsample(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        strategy: str,
        **kwargs: Any,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the end of sequence activity dataset using a specified strategy.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        strategy : str
            The subsampling strategy to use. Can be "random", "stratified", or "validation_focused".
        **kwargs
            Additional keyword arguments to pass to the subsampling strategy.

        Returns
        -------
        DataFrame
            The subsampled dataset.

        Raises
        ------
        ValueError
            If the specified strategy is not supported.
        """
        strategies = {
            "random": cls._random_fraction,
            "during_activity": cls._during_activity,
            "activity_boundaries": cls._activity_boundaries,
            "trackId_stratified": cls._trackId_stratified,
            "random_trackId_fraction": cls._random_trackId_fraction,
            "every_nth_percentile_of_same_activity": cls._every_nth_percentile_of_same_activity,
            "every_nth_plus_all_anchored_and_moored": cls._every_nth_plus_all_anchored_and_moored,
            # remove other dataset overlapped trackIds
            # Every Nth message within a activity sequence
        }

        if strategy in strategies:
            subsample_function: Callable[..., pd.DataFrame] = strategies[strategy]  # type: ignore
            subsampled_df = filter_kwargs(subsample_function)(dataset, **kwargs)
        else:
            raise ValueError(f"Subsampling strategy {strategy} is not supported.")

        return subsampled_df

    @classmethod
    def _random_fraction(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        fraction: float,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset by randomly selecting a fraction of the data.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        fraction : float
            The fraction of data to select. Should be between 0 and 1.

        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        num_samples = int(len(dataset) * fraction)
        subsampled_df = dataset.sample(n=num_samples)
        return subsampled_df

    @classmethod
    def _trackId_stratified(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        fraction: float,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset by randomly selecting a fraction of the data for each trackId.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        fraction : float
            The fraction of data to select for each trackId. Should be between 0 and 1.

        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        subsampled_dfs = []
        for id in cls.track_id_iterator(dataset):
            track_labels_df = cls.filter_to_sorted_single_track_id(dataset, id)
            num_samples = int(len(track_labels_df) * fraction)
            subsampled_df = track_labels_df.sample(n=num_samples)
            subsampled_dfs.append(subsampled_df)

        return cls.compile_subsampled_dataframes(subsampled_dfs)

    @classmethod
    def _random_trackId_fraction(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        fraction: float,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset by randomly selecting all the data for a fraction of the trackIds.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        fraction : float
            The fraction of trackIds to select. Should be between 0 and 1.

        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        unique_track_ids = dataset.trackId.unique()
        num_samples = int(len(unique_track_ids) * fraction)
        selected_track_ids = np.random.choice(unique_track_ids, num_samples)
        subsampled_df = dataset[dataset.trackId.isin(selected_track_ids)]
        return subsampled_df

    @classmethod
    def _during_activity(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset with a focus on messages in the middle of activity

        For each continuous segment of Activity:
        1/4 of the way through the segment
         - Pick the point in the middle of that segment
        - Pick the point 3/4 of that segment

        Reason: We want to be able to see if we can do well when activity is clear

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        subsampled_dfs = []
        for id in cls.track_id_iterator(dataset):
            track_labels_df = cls.filter_to_sorted_single_track_id(dataset, id)
            activity_change_indices = cls.get_activity_change_indices(track_labels_df)

            # get the median time between activity changes for each consecutive activity
            shift_indices = np.roll(activity_change_indices, 1)
            shift_indices[0] = 0
            joint_array = np.stack([shift_indices, activity_change_indices])
            median_index_between_activity_change = np.median(
                joint_array, axis=0
            ).astype(int)
            q25, q75 = np.percentile(joint_array, [25, 75], axis=0)
            q25 = q25.astype(int)
            q75 = q75.astype(int)
            all_indices = np.concatenate(
                [q25, median_index_between_activity_change, q75]
            ).astype(int)
            sorted_indices = np.unique(np.sort(all_indices))
            subsampled_df = track_labels_df[track_labels_df.index.isin(sorted_indices)]
            subsampled_dfs.append(subsampled_df)
        return cls.compile_subsampled_dataframes(subsampled_dfs)

    @classmethod
    def _activity_boundaries(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset at the beginning and end of activity.

        For each continuous segment of Activity:
         - Pick the point at the start of that segment
         - Pick the point at the end of that segment

        Reason: We want to be able to see if we can do well when activity is transitioning

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        subsampled_dfs = []
        for id in cls.track_id_iterator(dataset):
            track_labels_df = cls.filter_to_sorted_single_track_id(dataset, id)
            activity_change_indices = cls.get_activity_change_indices(track_labels_df)
            all_indices = activity_change_indices.astype(int)
            sorted_indices = np.unique(np.sort(all_indices))
            subsampled_df = track_labels_df[track_labels_df.index.isin(sorted_indices)]
            subsampled_dfs.append(subsampled_df)
        return cls.compile_subsampled_dataframes(subsampled_dfs)

    @classmethod
    def _every_nth_percentile_of_same_activity(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        nth_percentile: int,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset by selecting every Nth message within a sequence where activity class is the same

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        nth_percentile : int
            select the message at every Nth percentile

        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        subsampled_dfs = []
        for id in cls.track_id_iterator(dataset):
            track_labels_df = cls.filter_to_sorted_single_track_id(dataset, id)
            activity_change_indices = cls.get_activity_change_indices(track_labels_df)

            # get the median time between activity changes for each consecutive activity
            shift_indices = np.roll(activity_change_indices, 1)
            shift_indices[0] = 0
            joint_array = np.stack([shift_indices, activity_change_indices])
            percentiles_list = [i for i in range(0, 100, nth_percentile)]
            percentile_joint_array_lst = np.percentile(
                joint_array, percentiles_list, axis=0
            )

            all_indices = np.concatenate(percentile_joint_array_lst).astype(int)
            sorted_indices = np.unique(np.sort(all_indices))
            subsampled_df = track_labels_df[track_labels_df.index.isin(sorted_indices)]
            subsampled_dfs.append(subsampled_df)
        return cls.compile_subsampled_dataframes(subsampled_dfs)

    @classmethod
    def _every_nth_plus_all_anchored_and_moored(
        cls,
        dataset: DataFrame[ActivityEndOfSequenceLabelDataModel],
        nth_percentile: int,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """
        Subsample the dataset by selecting every Nth message within a sequence where activity class is the same and includes all anchored and moored messages

        Parameters
        ----------
        dataset : DataFrame
            The dataset to subsample.
        nth_percentile : int
            select the message at every Nth percentile

        Returns
        -------
        DataFrame
            The subsampled dataset.
        """
        anchored_moored_filt = dataset.activity_type_name.isin(["anchored", "moored"])
        anchored_and_moored_df = dataset[anchored_moored_filt]
        not_anchored_and_moored_df = dataset[~anchored_moored_filt]
        every_nth_df = cls._every_nth_percentile_of_same_activity(
            not_anchored_and_moored_df, nth_percentile
        )
        return pd.concat([every_nth_df, anchored_and_moored_df])
