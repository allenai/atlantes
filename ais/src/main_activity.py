"""Deployment Pipeline for the Atlas Activity Model
# TODO: Make all steps have simple pydantic response objects
"""

import os

import uvicorn
from atlantes.inference.atlas_activity.classifier import AtlasActivityClassifier
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    ATLASRequest,
    ATLASResponse,
)
from atlantes.log_utils import get_logger
from fastapi import FastAPI, HTTPException
from pandera.typing import DataFrame
from pydantic import BaseModel

logger = get_logger("atlas_activity_api")

app = FastAPI()

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
def index() -> Info:
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    info = Info(model_type="entity", git_commit_hash=git_commit_hash)
    logger.info(f"Received request for {info=}")
    return info


@app.post("/classify", response_model=ATLASResponse)
def classify(request: ATLASRequest) -> dict:
    try:
        tracks = [DataFrame(track) for track in request.tracks]
        result = classifier.run_pipeline(tracks) or []
        return ATLASResponse(predictions=result).model_dump(mode="json")
    except Exception as e:
        logger.exception("Error while running inference")
        raise HTTPException(status_code=500, detail="inference request failed") from e


if __name__ == "__main__":
    port = int(os.getenv("PORT", default=8000))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104
