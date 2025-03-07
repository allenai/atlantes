

from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
)
from pandera.errors import SchemaError
from pandera.typing import DataFrame


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

    def _apply_postprocessing(
        self,
        activity_class_details_metadata_tuples: list[
            tuple[AtlasActivityLabelsTraining, dict, dict]
        ],
    ) -> list[tuple[str, dict]]:
        results = []
        for (
            activity_class_details_metadata_tuple
        ) in activity_class_details_metadata_tuples:
            results.append(
                self.postprocessor.postprocess(activity_class_details_metadata_tuple)
            )
        return results

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
            preprocessed_data = [
                self.preprocessor.preprocess(track) for track in track_data
            ]

            classifications = self.model.run_inference(preprocessed_data)
            if classifications is None:
                return None

            results = [
                self.postprocessor.postprocess(classification)
                for classification in classifications
            ]
            return results
        except SchemaError as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
