from __future__ import annotations

from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.datamodels import EntityPostprocessorOutput
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import AtlasEntityPostProcessor
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.inference.common import AtlasInferenceError
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame

logger = get_logger("atlas_entity_classifier")


class AtlasEntityClassifier:
    """Class for identifying the entity of a trajectory using the Atlantes system for AIS behavior classification

    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to identify the entity of the trajectory.
    The entity will be either a vessel or equipment/buoy.
    """

    def __init__(
        self,
        preprocessor: AtlasEntityPreprocessor,
        model: AtlasEntityModel,
        postprocessor: AtlasEntityPostProcessor,
    ) -> None:
        """Load the preprocessor, model, and postprocessor for the entity classifier"""
        self.preprocessor: AtlasEntityPreprocessor = preprocessor
        self.model: AtlasEntityModel = model
        self.postprocessor: AtlasEntityPostProcessor = postprocessor

    def run_pipeline(
        self, track_data: list[DataFrame[TrackfileDataModelTrain]]
    ) -> list[EntityPostprocessorOutput]:
        """
        ATLAS entity requires data in TrackfileDataModelTrain format

        Pandera Validation occurs during preprocessing

        Parameters
        ----------
        track_data : DataFrame[TrackfileDataModelTrain]
            see atlantes.atlas.schemas.TrackfileDataModelTrain for the required columns
        Returns
        -------
        list[EntityPostprocessorOutput]
            Returns a list of EntityPostprocessorOutput objects

        """
        try:
            # Preprocessing
            preprocessed_data = []
            for track in track_data:
                try:
                    preprocessed = self.preprocessor.preprocess(track)
                    preprocessed_data.append(preprocessed)
                except Exception as e:
                    logger.warning(f"Error preprocessing track: {e}")
                    continue

            if len(preprocessed_data) == 0:
                logger.warning("No preprocessed data to run inference on")
                return []

            # Model inference
            classifications = self.model.run_inference(preprocessed_data)

            # Postprocessing
            results = []
            for classification in classifications:
                try:
                    result = self.postprocessor.postprocess(classification)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error postprocessing entity output: {e}")
                    continue

            # Return a list of the enum item name (lowered) and a dict of inference details
            return results
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
