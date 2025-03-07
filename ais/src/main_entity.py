"""ATLAS inference service Entity
"""

from __future__ import annotations

import os

import uvicorn
from atlantes.inference.atlas_entity.classifier import AtlasEntityClassifier
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor,
)
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.inference.common import (
    ATLASRequest,
    ATLASResponse,
)
from atlantes.log_utils import get_logger
from fastapi import FastAPI, HTTPException
from pandera.typing import DataFrame
from pydantic import BaseModel

logger = get_logger("atlas_entity_api")

app = FastAPI()

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
def index() -> Info:
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    info = Info(model_type="entity", git_commit_hash=git_commit_hash)
    logger.info(f"Received request for {info=}")
    return info


@app.post("/classify", response_model=ATLASResponse)
def classify(request: ATLASRequest) -> dict:
    try:
        tracks = [DataFrame(track) for track in request.tracks]
        results = classifier.run_pipeline(tracks)
        predictions = [result.serialize() for result in results]
        return ATLASResponse(predictions=predictions).model_dump(mode="json")
    except Exception as e:
        logger.exception("Error while running inference")
        raise HTTPException(status_code=500, detail="inference request failed") from e

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", default=8001))
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # nosec B104
