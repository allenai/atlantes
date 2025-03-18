from __future__ import annotations

from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import AtlasEntityPostProcessor
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
    AtlasModelTrackInputs,
    PostprocessFailure,
    Prediction,
    PreprocessFailure,
)
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame
from pydantic import BaseModel, Field

logger = get_logger("atlas_entity_classifier")


class PipelineInput(BaseModel):
    track_id: str
    track_data: DataFrame[TrackfileDataModelTrain]

    @staticmethod
    def from_track_data(track_data: AtlasModelTrackInputs) -> "PipelineInput":
        return PipelineInput(
            track_id=track_data.track_id,  # unique identifier for the track
            track_data=DataFrame(track_data.track_data),
        )


class PipelineOutput(BaseModel):
    predictions: list[Prediction] = Field(default_factory=list)
    preprocess_failures: list[PreprocessFailure] = Field(default_factory=list)
    postprocess_failures: list[PostprocessFailure] = Field(default_factory=list)


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

    def run_pipeline(self, inputs: list[PipelineInput]) -> PipelineOutput:
        """
        ATLAS entity requires data in TrackfileDataModelTrain format

        Pandera Validation occurs during preprocessing

        Parameters
        ----------
        inputs : list[PipelineInput]
            List of inputs containing track_id and track_data
        Returns
        -------
        PipelineOutput
            Contains predictions with entity class, inference details, and track_id
        """
        try:
            pipeline_output = PipelineOutput()
            preprocessed_data = []
            valid_track_ids = []

            for input_data in inputs:
                track_id = input_data.track_id
                try:
                    preprocessed = self.preprocessor.preprocess(input_data.track_data)
                    preprocessed_data.append(preprocessed)
                    # preprocess() returns an error if the track_data is not valid.
                    # this is necessary for the zip() below to work
                    valid_track_ids.append(track_id)
                except Exception as e:
                    logger.warning(f"Error preprocessing {input_data.track_id=}: {e}")
                    preprocess_failure = PreprocessFailure(
                        track_id=track_id, error=str(e)
                    )
                    pipeline_output.preprocess_failures.append(preprocess_failure)
                    continue

            classifications = self.model.run_inference(preprocessed_data)

            for classification, track_id in zip(classifications, valid_track_ids):
                try:
                    postprocessed = self.postprocessor.postprocess(classification)
                    pipeline_output.predictions.append(
                        Prediction(
                            track_id=track_id,
                            classification=postprocessed.entity_class,
                            details=postprocessed.entity_classification_details._asdict(),
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Error postprocessing {track_id=}, {classification=}: {e}"
                    )
                    postprocess_failure = PostprocessFailure(
                        track_id=track_id,
                        classification=classification.predicted_class.name,
                        error=str(e),
                    )
                    pipeline_output.postprocess_failures.append(postprocess_failure)
                    continue
            return pipeline_output
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
