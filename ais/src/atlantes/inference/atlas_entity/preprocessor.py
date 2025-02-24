""" Entity preprocessor class and ray serve deployment

# TODO: Deal with the subpath at the end being removed or the last points so that we can
provide the right classification for a given time
"""

from __future__ import annotations

from atlantes.atlas.ais_dataset import AISTrajectoryEntityDataset
from atlantes.atlas.atlas_utils import get_atlas_entity_inference_config
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.datamodels import PreprocessedEntityData
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame

logger = get_logger("atlas_entity_preprocessor")


# ONLY TAKE effect during Local deployment or Testing
NUM_REPLICAS = 1
NUM_CPUS = 1
NUM_GPUS = 0


class AtlasEntityPreprocessor:
    """ "Class for preprocessing for AIS trajectory entity classification"""

    inference_config = get_atlas_entity_inference_config()
    dataset_config = inference_config["data"]

    @classmethod
    def preprocess(
        cls, track_data: DataFrame[TrackfileDataModelTrain]
    ) -> PreprocessedEntityData:
        """Preprocess the AIS trajectory data for entity classification using the Atlantes system

        Parameters
        ----------
        track_data : DataFrame[TrackfileDataModelTrain]
            Raw track data to preprocess

        Returns
        -------
        PreprocessedEntityData
            Preprocessed data ready for model inference

        Raises
        ------
        ValueError
            If the input data does not match the expected schema
        """
        # Preprocessing is performed in the getitem method identically to how it is done in the training pipeline
        entity_dataset = AISTrajectoryEntityDataset(
            mode="online",
            in_memory_data=[track_data],
            dataset_config=cls.dataset_config,
        )
        preprocessed = entity_dataset[0]
        return PreprocessedEntityData(**preprocessed)
