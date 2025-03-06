"""Deployment Pipeline for the Atlas Activity Model
# TODO: Make all steps have simple pydantic response objects
"""

import os

import uvicorn
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    AtlasInferenceError,
    ATLASRequest,
    ATLASResponse,
)
from atlantes.log_utils import get_logger
from fastapi import FastAPI, HTTPException
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from pydantic import BaseModel

logger = get_logger("atlas_activity_api")

app = FastAPI()
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


class Info(BaseModel):
    model_type: str
    git_commit_hash: str


@app.get("/info", response_model=Info)
def index():
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    info = Info(model_type="entity", git_commit_hash=git_commit_hash)
    logger.info(f"Received request for {info=}")
    return info


@app.post("/classify", response_model=ATLASResponse)
def classify(request: ATLASRequest):
    try:
        tracks = [DataFrame(track) for track in request.tracks]
        result = classifier.run_pipeline(tracks) or []
        return ATLASResponse(predictions=result).model_dump(mode="json")
    except Exception as e:
        logger.exception("Error while running inference")
        raise HTTPException(status_code=500, detail="inference request failed") from e


if __name__ == "__main__":
    port = int(os.getenv("PORT", default=8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
