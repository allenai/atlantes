"""API for the Atlas Entity Model"""

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
    InfoResponse,
)
from atlantes.log_utils import get_logger
from fastapi import FastAPI, HTTPException, Response
from pandera.typing import DataFrame
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

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

@app.get("/info", response_model=InfoResponse)
def index() -> InfoResponse:
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    info = InfoResponse(model_type="entity", git_commit_hash=git_commit_hash)
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

@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", default=8001))
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # nosec B104
