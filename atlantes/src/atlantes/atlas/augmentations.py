"""Module for data augmentations for ATLAS model

These are applied after preprocessing and before batching.
"""

import inspect
from typing import Callable

import numpy as np
import pandas as pd
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.log_utils import get_logger
from atlantes.utils import BaseRegistry
from pandera.typing import DataFrame

logger = get_logger(__name__)


class AugmentationRegistry(BaseRegistry):
    """Registry for data augmentations"""

    def register_augmentation(self, func: Callable, name: str) -> None:
        """Register an augmentation in the augmentation registry"""
        sig = inspect.signature(func)
        first_param = next(iter(sig.parameters.values()))
        if first_param.name != "trajectory_df":
            raise ValueError(f"The first parameter of {name} must be 'trajectory_df'")
        super().register(func, name)

    def get_augmentation(self, name: str) -> Callable:
        """Get an augmentation from the registry"""
        return super().get(name)


augmentation_registry = AugmentationRegistry()


def random_context_length(
    trajectory_df: DataFrame[TrackfileDataModelTrain],
    min_context_length: int,
    max_context_length: int,
) -> DataFrame[TrackfileDataModelTrain]:
    """Randomly select a context length for the trajectory"""
    max_context_length = min(max_context_length, len(trajectory_df))
    if min_context_length >= max_context_length:
        raise ValueError(
            f"min_context_length must be less than max_context_length {min_context_length} >= {max_context_length}"
        )
    # +  1 since since np.random.randint is [low, high)
    max_context_length = max_context_length + 1
    context_length = np.random.randint(min_context_length, max_context_length)
    return trajectory_df.iloc[-context_length:]


def random_message_dropout(
    trajectory_df: DataFrame[TrackfileDataModelTrain],
    min_dropout_rate: float,
    max_dropout_rate: float,
    min_messages: int,
) -> DataFrame[TrackfileDataModelTrain]:
    """Randomly drop messages from the trajectory

    Always Preserve the last (most recent) message in the trajectory.
    """
    if min_dropout_rate > max_dropout_rate:
        raise ValueError(
            f"min_dropout_rate must be less or equal max_dropout_rate {min_dropout_rate} >= {max_dropout_rate}"
        )
    dropout_rate = np.random.uniform(min_dropout_rate, max_dropout_rate)
    if len(trajectory_df) * (1 - dropout_rate) < min_messages:
        logger.warning(
            f"Not enough messages to drop, dropping none. {len(trajectory_df)} * {dropout_rate} < {min_messages} \
                consider lowering the max dropout rate"
        )
        return trajectory_df
    else:
        return pd.concat(
            [
                trajectory_df.iloc[:-1].sample(frac=1 - dropout_rate),
                trajectory_df.iloc[-1:],
            ]
        )


augmentation_registry.register_augmentation(
    random_context_length, "random_context_length"
)
augmentation_registry.register_augmentation(
    random_message_dropout, "random_message_dropout"
)
