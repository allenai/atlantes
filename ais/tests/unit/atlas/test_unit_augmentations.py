"""Unit tests for the augmentations module."""

from typing import Callable

import pandas as pd
import pytest
from atlantes.atlas.augmentations import (AugmentationRegistry,
                                          random_context_length,
                                          random_message_dropout)


@pytest.fixture
def sample_trajectory_df() -> pd.DataFrame:
    """Fixture for a sample trajectory DataFrame"""
    trajectory_df = pd.DataFrame(
        {
            "mmsi": [1, 1, 1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
            ),
            "latitude": [0, 1, 2, 3, 4],
            "longitude": [0, 1, 2, 3, 4],
        }
    )
    return trajectory_df


@pytest.fixture
def test_augmentation_registry() -> AugmentationRegistry:
    """Fixture for the AugmentationRegistry"""
    return AugmentationRegistry()


@pytest.fixture
def sample_augmentation_func() -> Callable:
    """Fixture for a sample augmentation function"""

    def sample_augmentation_func(trajectory_df: pd.DataFrame) -> pd.DataFrame:
        return trajectory_df

    return sample_augmentation_func


def test_random_context_length_happy_path(sample_trajectory_df: pd.DataFrame) -> None:
    """Test the random_context_length augmentation"""

    augmented_df = random_context_length(sample_trajectory_df, 2, 5)
    assert len(augmented_df) == 4
    assert augmented_df["timestamp"].iloc[0] == pd.Timestamp("2021-01-02")


def test_random_context_length_min_greater_than_max(sample_trajectory_df: pd.DataFrame
) -> None:
    """Test the random_context_length augmentation with min_context_length > max_context_length"""

    with pytest.raises(ValueError):
        random_context_length(sample_trajectory_df, 5, 2)


def test_random_context_length_min_equal_to_max(sample_trajectory_df: pd.DataFrame
) -> None:
    """Test the random_context_length augmentation with min_context_length == max_context_length"""

    with pytest.raises(ValueError):
        random_context_length(sample_trajectory_df, 2, 2)


def test_random_message_dropout_happy_path(sample_trajectory_df: pd.DataFrame) -> None:
    """Test the random_message_dropout augmentation"""
    augmented_df = random_message_dropout(sample_trajectory_df, 0.1, 0.5, 1)
    assert len(augmented_df) == 4
    # always keep most recent message
    assert augmented_df["timestamp"].iloc[-1] == pd.Timestamp("2021-01-05")


def test_random_message_dropout_not_enough_messages(
    sample_trajectory_df: pd.DataFrame,
) -> None:
    """Test the random_message_dropout augmentation with not enough messages to drop"""

    augmented_df = random_message_dropout(sample_trajectory_df, 0.5, 0.5, 10)
    pd.testing.assert_frame_equal(augmented_df, sample_trajectory_df)


class TestAugmentationRegistry:
    """Tests for the AugmentationRegistry class"""

    def test_register_augmentation_happy_path(
        self,
        test_augmentation_registry: AugmentationRegistry,
        sample_augmentation_func: Callable,
    ) -> None:
        """Test the register_augmentation method"""

        test_augmentation_registry.register_augmentation(
            sample_augmentation_func, "sample_augmentation_func"
        )
        assert "sample_augmentation_func" in test_augmentation_registry.registry

    def test_register_augmentation_duplicate(
        self,
        test_augmentation_registry: AugmentationRegistry,
        sample_augmentation_func: Callable,
    ) -> None:
        """Test the register_augmentation method with a duplicate augmentation"""

        test_augmentation_registry.register_augmentation(
            sample_augmentation_func, "sample_augmentation_func"
        )
        with pytest.raises(ValueError):
            test_augmentation_registry.register_augmentation(
                sample_augmentation_func, "sample_augmentation_func"
            )

    def test_register_augmentation_wrong_first_param(
        self,
        test_augmentation_registry: AugmentationRegistry,
    ) -> None:
        """Test the register_augmentation method with an augmentation with the wrong first parameter"""

        def wrong_first_param(not_trajectory_df: pd.DataFrame) -> pd.DataFrame:
            return not_trajectory_df

        with pytest.raises(ValueError):
            test_augmentation_registry.register_augmentation(
                wrong_first_param, "wrong_first_param"
            )

    def test_get_augmentation_happy_path(
        self,
        test_augmentation_registry: AugmentationRegistry,
        sample_augmentation_func: Callable,
    ) -> None:
        """Test the get_augmentation method"""

        test_augmentation_registry.register_augmentation(
            sample_augmentation_func, "sample_augmentation_func"
        )
        augmentation = test_augmentation_registry.get_augmentation(
            "sample_augmentation_func"
        )
        assert augmentation == sample_augmentation_func

    def test_get_augmentation_not_registered(
        self, test_augmentation_registry: AugmentationRegistry
    ) -> None:
        """Test the get_augmentation method with an augmentation that is not registered"""

        with pytest.raises(ValueError):
            test_augmentation_registry.get_augmentation("not_registered")
