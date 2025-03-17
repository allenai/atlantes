from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
)
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame

logger = get_logger("atlas_activity_classifier")


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
    ) -> list[tuple[str, dict]] | None:
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
            classifications = self.model.run_inference(preprocessed_data)

            results = []
            for classification in classifications:
                try:
                    result = self.postprocessor.postprocess(classification)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error postprocessing classification: {e}")
                    continue
            return results
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
