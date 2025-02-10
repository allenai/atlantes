"""Activity preprocessor class

# TODO: Deal with the subpath at the end being removed or the last points so that we can
provide the right classification for a given time
"""

from __future__ import annotations

from atlantes.atlas.ais_dataset import ActivityDatasetEndOfSequence
from atlantes.atlas.atlas_utils import get_atlas_activity_inference_config
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.datamodels import PreprocessedActivityData
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame

logger = get_logger("atlas_activity_preprocessor")


# ONLY TAKE effect during testing
NUM_REPLICAS = 1
NUM_CPUS = 1
NUM_GPUS = 0


class AtlasActivityPreprocessor:
    """Class for preprocessing for Atlas activity classification"""

    inference_config = get_atlas_activity_inference_config()
    dataset_config = inference_config["data"]

    @classmethod
    def preprocess(
        cls, track_data: DataFrame[TrackfileDataModelTrain]
    ) -> PreprocessedActivityData:
        """Preprocess the activity data for classification using the Atlantes system"""
        # Preprocessing is performed in the getitem method identically to how it is done in the training pipeline
        activity_dataset = ActivityDatasetEndOfSequence(
            dataset_config=cls.dataset_config,
            mode="online",
            in_memory_data=[track_data],
        )
        return PreprocessedActivityData(**activity_dataset[0])