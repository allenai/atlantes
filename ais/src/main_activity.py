"""Deployment Pipeline for the Atlas Activity Model
# TODO: Make all steps have simple pydantic response objects
"""

import os

import pandas as pd
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import (
    AtlasActivityPreprocessor,
)
from atlantes.inference.common import AtlasInferenceError, ATLASRequest, ATLASResponse
from atlantes.log_utils import get_logger
from flask import Flask, request
from pandera.errors import SchemaError
from pandera.typing import DataFrame

app = Flask(__name__)

logger = get_logger("atlas_activity_api")


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


preprocessor = AtlasActivityPreprocessor()
model = AtlasActivityModel()
postprocessor = AtlasActivityPostProcessor()
classifier = AtlasActivityClassifier(
    preprocessor=preprocessor,
    model=model,
    postprocessor=postprocessor,
)


@app.route("/info", methods=["GET"])
def index():
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    return {"git_commit_hash": git_commit_hash}


@app.route("/classify", methods=["POST"])
def classify():
    try:
        atlas_request = ATLASRequest(**request.json)
        result = classifier.run_pipeline(
            [pd.DataFrame(track) for track in atlas_request.tracks]
        )
        return ATLASResponse(predictions=result).model_dump(mode="json")
    except Exception as e:
        logger.exception(f"Error while running inference: {e}")
        return {"error": "inference request failed"}, 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", default=8080))
    app.run(host="0.0.0.0", port=port)
