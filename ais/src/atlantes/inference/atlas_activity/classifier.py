from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
    PostprocessFailure,
    Prediction,
    PreprocessFailure,
    TrackData,
)
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame
from pydantic import BaseModel, Field

logger = get_logger("atlas_activity_classifier")


class PipelineInput(BaseModel):
    track_id: str
    track_data: DataFrame[TrackfileDataModelTrain]

    @staticmethod
    def from_track_data(track_data: TrackData) -> "PipelineInput":
        return PipelineInput(
            track_id=track_data.track_id,  # unique identifier for the track
            track_data=DataFrame(track_data.track_data),
        )


class PipelineOutput(BaseModel):
    # predictions: (classification, details, track_id)
    predictions: list[Prediction] = Field(default_factory=list)
    preprocess_failures: list[PreprocessFailure] = Field(default_factory=list)
    postprocess_failures: list[PostprocessFailure] = Field(default_factory=list)


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

    def run_pipeline(self, inputs: list[PipelineInput]) -> PipelineOutput:
        """
        ATLAS activity requires data in TrackfileDataModelTrain format
        Pandera Validation occurs during preprocessing
        Parameters
        ----------
        inputs : list[PipelineInput]
            List of inputs containing track_id and track_data
        Returns
        -------
        PipelineOutput
            Contains predictions with activity class, inference details, and track_id
        """
        try:
            pipeline_output = PipelineOutput()
            preprocessed_data = []
            track_ids = []

            for input_data in inputs:
                track_id = input_data.track_id
                try:
                    preprocessed = self.preprocessor.preprocess(input_data.track_data)
                    preprocessed_data.append(preprocessed)
                    track_ids.append(track_id)
                except Exception as e:
                    logger.warning(f"Error preprocessing {input_data.track_id=}: {e}")
                    failure = PreprocessFailure(track_id=track_id, error=str(e))
                    pipeline_output.preprocess_failures.append(failure)
                    continue

            classifications = self.model.run_inference(preprocessed_data)

            for classification, track_id in zip(classifications, track_ids):
                try:
                    activity, details = self.postprocessor.postprocess(classification)
                    pipeline_output.predictions.append(
                        Prediction(
                            track_id=track_id,
                            classification=activity,
                            details=details,
                        )
                    )
                except Exception as e:
                    c = classification[0]
                    logger.warning(f"Error postprocessing {track_id=}, {c=}: {e}")
                    failure = PostprocessFailure(
                        track_id=track_id,
                        classification=c.name,
                        error=str(e),
                    )
                    pipeline_output.postprocess_failures.append(failure)
                    continue
            return pipeline_output
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
