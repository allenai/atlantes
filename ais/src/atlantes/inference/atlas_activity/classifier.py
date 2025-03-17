from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
)
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame
from pydantic import BaseModel, Field

logger = get_logger("atlas_activity_classifier")

class PipelineOutput(BaseModel):
    predictions: list[tuple[str, dict]] = Field(default_factory=list)
    num_failed_preprocessing: int = Field(default=0)
    num_failed_postprocessing: int = Field(default=0)

class AtlasActivityClassifier:
    """Class for classifying the activity of a trajectory using the Atlantes system for AIS behavior classification
    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to classify the activity of the trajectory.
    The activity will be one of the predefined activity classes."""

    def __init__(
        self,
        preprocessor: AtlasActivityPreprocessor,
        model: AtlasActivityModel,
        postprocessor: AtlasActivityPostProcessor,
    ) -> None:
        """Load the preprocessor, model, and postprocessor for the activity classifier DeploymentHandle"""
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def run_pipeline(
        self, track_data: list[DataFrame[TrackfileDataModelTrain]]
    ) -> PipelineOutput:
        """
        ATLAS activity requires data in TrackfileDataModelTrain format
        Pandera Validation occurs during preprocessing
        Parameters
        ----------
        track_data : DataFrame[TrackfileDataModelTrain]
            see atlantes.atlas.schemas.TrackfileDataModelTrain for the required columns
        Returns
        -------
        tuple[str, dict]
            Returns a tuple of the predicted activity class and a dict of inference details e.g confidence, outputs
        """
        try:
            pipeline_output = PipelineOutput()
            preprocessed_data = []
            for track in track_data:
                try:
                    preprocessed = self.preprocessor.preprocess(track)
                    preprocessed_data.append(preprocessed)
                except Exception as e:
                    logger.warning(f"Error preprocessing track: {e}")
                    pipeline_output.num_failed_preprocessing += 1
                    continue

            if len(preprocessed_data) == 0:
                logger.warning("No preprocessed data to run inference on")
                return pipeline_output
            classifications = self.model.run_inference(preprocessed_data)

            for classification in classifications:
                try:
                    postprocessed = self.postprocessor.postprocess(classification)
                    pipeline_output.predictions.append(postprocessed)
                except Exception as e:
                    logger.warning(f"Error postprocessing classification: {e}")
                    pipeline_output.num_failed_postprocessing += 1
                    continue
            return pipeline_output
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
