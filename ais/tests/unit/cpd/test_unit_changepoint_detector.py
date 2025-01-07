"""Unit tests for the `changepoint_detector` module. """

from datetime import datetime, timedelta

import numpy as np
import pytest
from atlantes.cpd.changepoint_detector import (ChangepointDetector,
                                               SpeedChangeDetector,
                                               TimeGapDetector)


# add unit tests for the rest of the variables and ensure that the min time gap is less than max duration
class TestUnitChangepointDetector:
    """Unit tests for the `ChangepointDetector` class."""

    def test_get_time_based_changepoint_detector(self) -> None:
        """Test that the `get_time_based_changepoint_detector` method returns a `TimeGapDetector` object."""

        # Act
        time_gap_detector = ChangepointDetector._get_time_based_changepoint_detector()
        # Assert
        assert isinstance(time_gap_detector, TimeGapDetector)
        assert time_gap_detector.min_time_gap == timedelta(
            hours=2
        ), "Changing this constant will change cpd behavior"

    def test_get_sog_based_changepoint_detector(self) -> None:
        """Test that the `get_sog_based_changepoint_detector` method returns a `SpeedChangeDetector` object."""

        # Act
        sog_cumsum_detector = (
            ChangepointDetector._get_speed_based_changepoint_detector()
        )
        # Assert
        assert isinstance(sog_cumsum_detector, SpeedChangeDetector)
        assert (
            sog_cumsum_detector.base_sample_size == 5
        ), "Changing this constant will change cpd behavior"
        assert (
            sog_cumsum_detector.changepoint_probability_threshold == 0.001
        ), "Changing this constant will change cpd behavior"

    def test_min_messages(self) -> None:
        """Test that the `min_messages` property returns the correct value."""

        # Act
        min_messages = ChangepointDetector.MIN_MESSAGES
        # Assert
        assert min_messages == 5, "Changing this constant will change cpd behavior"

    def test_max_duration(self) -> None:
        """Test that the `max_duration` property returns the correct value."""

        # Act
        max_duration = ChangepointDetector.MAX_DURATION
        # Assert
        assert max_duration == timedelta(
            hours=24
        ), "Changing this constant will change cpd behavior"

    def test_max_num_messages(self) -> None:
        """Test that the `max_num_messages` property returns the correct value."""

        # Act
        max_num_messages = ChangepointDetector.MAX_NUM_MESSAGES
        # Assert
        assert (
            max_num_messages == 500
        ), "Changing this constant will change cpd behavior"


class TestUnitTimeGapDetector:
    """Unit tests for the `TimeGapDetector` class."""

    def test_is_change_point(self) -> None:
        """Test that the `is_change_point` method returns the correct boolean value."""

        # Arrange
        time_gap_detector = TimeGapDetector(min_time_gap=timedelta(hours=12))
        times = [
            datetime(2020, 12, 31, 23, 59),
            datetime(2021, 1, 1, 0, 0),
            datetime(2021, 1, 2, 12, 0),
        ]
        # Act
        is_changepoint1 = time_gap_detector.is_change_point(times)
        is_changepoint2 = time_gap_detector.is_change_point(times[:2])
        is_changepoint3 = time_gap_detector.is_change_point(times[1:])

        # Assert
        assert is_changepoint1 is True
        assert is_changepoint2 is False
        assert is_changepoint3 is True


class TestUnitSpeedChangeDetector:
    """Unit tests for the `SpeedChangeDetector` class."""

    def test_init_base_speed_distribution(self) -> None:
        """Test that the `init_base_sog_distribution` method returns the correct distribution."""

        # Arrange
        sog_cumsum_detector = SpeedChangeDetector(
            base_sample_size=5, changepoint_probability_threshold=0.001
        )
        sogs = np.array([1, 2, 3, 4, 5])

        # Act
        mean, std = sog_cumsum_detector._init_base_speed_distribution(sogs)

        # Assert
        assert mean == 3.0
        assert std == np.std(sogs)

    @pytest.mark.parametrize(
        "new_standardized_sum, expected_prob",
        [
            (-10, 0.0),
            (-5, 5.733031438470704e-07),
            (-1.0, 0.31731050786291415),
            (-0.5, 0.6170750774519738),
            (0.0, 1.0),
            (0.21, 0.8336676730351154),
            (1.0, 0.31731050786291415),
            (3.5, 0.0004652581580710802),
        ],
    )
    def test_get_prob(self, new_standardized_sum: float, expected_prob: float) -> None:
        """Test that the `get_prob` method returns the correct probability."""

        # Arrange
        sog_cumsum_detector = SpeedChangeDetector(
            base_sample_size=5, changepoint_probability_threshold=0.001
        )

        # Act
        prob = sog_cumsum_detector._get_prob(new_standardized_sum)

        # Assert
        assert prob == expected_prob

    def test_get_base_sample(self) -> None:
        """Test that the `get_base_sample` method returns the correct base sample."""

        # Arrange
        sog_cumsum_detector = SpeedChangeDetector(
            base_sample_size=5, changepoint_probability_threshold=0.001
        )
        sogs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Act
        base_sample = sog_cumsum_detector._get_base_sample(sogs)
        base_sample2 = sog_cumsum_detector._get_base_sample(sogs[:7])

        # Assert
        assert np.array_equal(base_sample, np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(base_sample2, np.array([1, 2, 3, 4, 5]))

    def test_is_change_point(self) -> None:
        """Test that the `is_change_point` method returns the correct boolean value."""

        # Arrange
        sog_cumsum_detector = SpeedChangeDetector(
            base_sample_size=5, changepoint_probability_threshold=0.001
        )
        # less than base size observations
        sogs_less_then_base_size = np.array(
            [
                1,
                2,
                3,
                400,
            ]
        )
        sogs1 = [0.2, 0.3, 0.3, 0.4, 0.6, 0.7, 0.5, 0.8, 0.9, 1.0, 7.0]

        # Drifting edge case
        # Act
        is_changepoint1 = sog_cumsum_detector.is_change_point(sogs_less_then_base_size)
        is_changepoint3 = sog_cumsum_detector.is_change_point(sogs1[:7])
        is_changepoint2 = sog_cumsum_detector.is_change_point(sogs1)

        # Assert
        assert is_changepoint1 is False
        assert is_changepoint2 is True
        assert is_changepoint3 is False
