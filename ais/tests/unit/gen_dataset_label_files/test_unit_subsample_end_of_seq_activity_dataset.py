"""Unit tests for the subsample_end_of_seq_activity_dataset module.

This module contains unit tests for the subsample_end_of_seq_activity_dataset module.
"""

import pandas as pd
import pytest
from atlantes.atlas.schemas import ActivityEndOfSequenceLabelDataModel
from atlantes.gen_dataset_label_files.subsample_end_of_seq_activity_dataset import \
    SubsampleEndOfSeqActivityLabels
from pandas.testing import assert_frame_equal
from pandera.typing import DataFrame


@pytest.fixture(scope="class")
def activity_labels_df() -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
    """Fixture for a DataFrame of activity labels"""
    return pd.DataFrame(
        {
            "trackId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "activity_type_name": ["a", "a", "b", "a", "a", "b", "a", "a", "b"],
            "send": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )


@pytest.fixture(scope="class")
def activity_labels_df_with_many_different_activities() -> (
    DataFrame[ActivityEndOfSequenceLabelDataModel]
):
    """Fixture for a DataFrame of activity labels with many different activities"""
    return pd.DataFrame(
        {
            "trackId": ["1", "1", "1", "1", "1", "1", "1", "1", "3"],
            "activity_type_name": ["a", "a", "a", "a", "a", "a", "a", "a", "c"],
            "send": ["1", "2", "3", "1", "2", "3", "1", "2", "3"],
        }
    )


class TestSubsampleEndOfSeqActivityLabels:
    def test_subsample_e2e_of_activity_labels(
        self, activity_labels_df: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> None:
        """Test the subsample method"""
        strategy = "random"
        fraction = 0.5
        kwargs = {"fraction": fraction, "unused": "unused"}

        # Test that we can filter args
        subsampled_df = SubsampleEndOfSeqActivityLabels.subsample(
            activity_labels_df, strategy, **kwargs
        )
        assert len(subsampled_df) <= len(activity_labels_df) * fraction

    def test_random_fraction_subsample_of_activity_labels(
        self, activity_labels_df: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> None:
        # Test the _random_fraction method
        fraction = 0.5
        subsampled_df = SubsampleEndOfSeqActivityLabels._random_fraction(
            activity_labels_df, fraction
        )
        assert len(subsampled_df) <= len(activity_labels_df) * fraction

    def test_trackId_stratified_subsample_of_activity_labels(
        self, activity_labels_df: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> None:
        # Test the _trackId_stratified method
        fraction = 0.5
        num_track_ids = len(activity_labels_df.trackId.unique())
        subsampled_df = SubsampleEndOfSeqActivityLabels._trackId_stratified(
            activity_labels_df, fraction
        )
        subsampled_num_track_ids = len(subsampled_df.trackId.unique())
        assert subsampled_num_track_ids == num_track_ids
        assert len(subsampled_df) <= len(activity_labels_df) * fraction

    def test_random_trackId_fraction_subsample_of_activity_labels(
        self, activity_labels_df: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> None:
        # Test the _random_trackId_fraction method
        fraction = 0.5
        subsampled_df = SubsampleEndOfSeqActivityLabels._random_trackId_fraction(
            activity_labels_df, fraction
        )
        assert len(subsampled_df) <= len(activity_labels_df) * fraction
        assert (
            len(subsampled_df.trackId.unique())
            <= len(activity_labels_df.trackId.unique()) * fraction
        )

    def test_during_activity_subsample_of_activity_labels(
        self,
        activity_labels_df_with_many_different_activities: DataFrame[
            ActivityEndOfSequenceLabelDataModel
        ],
    ) -> None:
        # Test the _during_activity method
        subsampled_df = SubsampleEndOfSeqActivityLabels._during_activity(
            activity_labels_df_with_many_different_activities
        )
        expected_df = pd.DataFrame(
            {
                "trackId": ["1", "1", "1", "3"],
                "activity_type_name": ["a", "a", "a", "c"],
                "send": ["1", "2", "2", "3"],
            }
        )

        assert_frame_equal(subsampled_df, expected_df)

    def test_activity_boundaries_subsample_of_activity_labels(
        self, activity_labels_df: DataFrame[ActivityEndOfSequenceLabelDataModel]
    ) -> None:
        # Test the _activity_boundaries method
        subsampled_df = SubsampleEndOfSeqActivityLabels._activity_boundaries(
            activity_labels_df
        )
        assert subsampled_df is not None

    def test_every_nth_percentile_of_same_activity_subsample_of_activity_labels(
        self,
        activity_labels_df_with_many_different_activities: DataFrame[
            ActivityEndOfSequenceLabelDataModel
        ],
    ) -> None:
        # Test the _every_nth_percentile_of_same_activity method
        nth_percentile = 25
        subsampled_df = (
            SubsampleEndOfSeqActivityLabels._every_nth_percentile_of_same_activity(
                activity_labels_df_with_many_different_activities, nth_percentile
            )
        )
        assert subsampled_df is not None

    def test_every_nth_plus_all_anchored_and_moored_subsample_of_activity_labels(
        self,
        activity_labels_df_with_many_different_activities: DataFrame[
            ActivityEndOfSequenceLabelDataModel
        ],
    ) -> None:
        # Test the _every_nth_plus_all_anchored_and_moored method
        nth_percentile = 25
        subsampled_df = (
            SubsampleEndOfSeqActivityLabels._every_nth_plus_all_anchored_and_moored(
                activity_labels_df_with_many_different_activities, nth_percentile
            )
        )
        assert subsampled_df is not None
