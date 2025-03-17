from __future__ import annotations

from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.datamodels import EntityPostprocessorOutput
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import AtlasEntityPostProcessor
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.inference.common import AtlasInferenceError
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame
from pydantic import BaseModel, Field

logger = get_logger("atlas_entity_classifier")

class PipelineOutput(BaseModel):
    predictions: list[EntityPostprocessorOutput] = Field(default_factory=list)
    num_failed_preprocessing: int = Field(default=0)
    num_failed_postprocessing: int = Field(default=0)

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
    ) -> PipelineOutput:
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
            pipeline_output = PipelineOutput()
            preprocessed_data = []
            logger.error(f"A1 preprocessing {len(track_data)} tracks")
            for track in track_data:
                try:
                    logger.error("A2")
                    preprocessed = self.preprocessor.preprocess(track)
                    logger.error("A3")
                    preprocessed_data.append(preprocessed)
                except Exception as e:
                    logger.error("A4")
                    logger.warning(f"Error preprocessing track: {e}")
                    pipeline_output.num_failed_preprocessing += 1
                    continue
            logger.error("A5")
            if len(preprocessed_data) == 0:
                logger.error("A6")
                logger.warning("No preprocessed data to run inference on")
                return pipeline_output
            logger.error("A7")
            classifications = self.model.run_inference(preprocessed_data)
            logger.error("A8")
            results = []
            for classification in classifications:
                try:
                    logger.error("A9")
                    result = self.postprocessor.postprocess(classification)
                    logger.error("A10")
                    results.append(result)
                except Exception as e:
                    logger.error("A11")
                    logger.warning(f"Error postprocessing entity output: {e}")
                    pipeline_output.num_failed_postprocessing += 1
                    continue
            logger.error("A12")
            return pipeline_output
        except Exception as e:
            logger.error("A13")
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
