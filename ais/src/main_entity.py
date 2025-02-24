"""ATLAS inference service Entity
"""

from __future__ import annotations

import os

import uvicorn
from atlantes.inference.atlas_entity.datamodels import (
    EntityPostprocessorInput,
    EntityPostprocessorOutput,
    PreprocessedEntityData,
)
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor,
    KnownShipTypeAndBuoyName,
)
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.inference.common import AtlasInferenceError, ATLASRequest, ATLASResponse
from fastapi import FastAPI
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from pydantic import BaseModel

from .atlantes.atlas.schemas import TrackfileDataModelTrain
from .atlantes.log_utils import get_logger

logger = get_logger("atlas_entity_api")

app = FastAPI()


class AtlasEntityClassifier:
    """Class for identifying the entity of a trajectory using the Atlantes system for AIS behavior classification

    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to identify the entity of the trajectory.
    tHe entity will be either a vessel or equipment/buoy.

    Note: This pipeline does not support batching
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

    def _apply_preprocessing(
        self, track_data: list[DataFrame[TrackfileDataModelTrain]]
    ) -> list[PreprocessedEntityData]:
        return [self.preprocessor.preprocess(track) for track in track_data]

    def _apply_model(
        self, preprocessed_data: list[PreprocessedEntityData]
    ) -> list[EntityPostprocessorInput]:
        """Run inference on the preprocessed data using the mode"""
        return self.model.run_inference(preprocessed_data)

    def _apply_postprocessing(
        self,
        entity_outputs_with_details_metadata_tuples: list[EntityPostprocessorInput],
    ) -> list[EntityPostprocessorOutput]:
        """Postprocess the AIS trajectory data for entity classification
        using the Atlantes system"""
        return [
            self.postprocessor.postprocess(entity_output)
            for entity_output in entity_outputs_with_details_metadata_tuples
        ]

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
        tuple[str, dict]
            Returns a tuple of the predicted entity class and a dict of inference details e.g confidence, outputs

        """
        try:
            preprocessed_data = self._apply_preprocessing(track_data)
            classifications = self._apply_model(preprocessed_data)
            results = self._apply_postprocessing(classifications)
            # Return a list of the enum item name (lowered) and a dict of inference details.
            return results
        except SchemaError as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
        except KnownShipTypeAndBuoyName as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e


preprocessor = AtlasEntityPreprocessor()
model = AtlasEntityModel()
postprocessor = AtlasEntityPostProcessor()
classifier = AtlasEntityClassifier(
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
    return Info(model_type="entity", git_commit_hash=git_commit_hash)


@app.post("/classify", response_model=ATLASResponse)
async def classify(request: ATLASRequest):
    try:
        tracks = [DataFrame(track) for track in request.tracks]
        results = classifier.run_pipeline(tracks)
        predictions = [result.serialize() for result in results]
        return ATLASResponse(predictions=predictions)
    except Exception as e:
        logger.exception(f"Error while running inference: {e}")
        return {"error": "inference request failed"}, 500

if __name__ == "__main__":
    PORT = int(os.getenv("ATLAS_ENTITY_PORT", default=8001))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
