"""API for the Atlas Activity Model"""

import os

import uvicorn
from atlantes.inference.atlas_activity.classifier import AtlasActivityClassifier
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import (
    ATLASRequest,
    ATLASResponse,
    InfoResponse,
)
from atlantes.log_utils import get_logger
from fastapi import FastAPI, HTTPException, Response
from pandera.typing import DataFrame
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

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


@app.get("/info", response_model=InfoResponse)
def index() -> InfoResponse:
    git_commit_hash = os.getenv("GIT_COMMIT_HASH", default="unknown")
    info = InfoResponse(model_type="activity", git_commit_hash=git_commit_hash)
    logger.info(f"Received request for {info=}")
    return info


@app.post("/classify", response_model=ATLASResponse)
def classify(request: ATLASRequest) -> dict:
    try:
        tracks = [DataFrame(track) for track in request.tracks]
        output = classifier.run_pipeline(tracks)
        return ATLASResponse(
            predictions=output.predictions,
            num_failed_preprocessing=output.num_failed_preprocessing,
            num_failed_postprocessing=output.num_failed_postprocessing,
        ).model_dump(mode="json")
    except Exception as e:
        logger.exception("Error while running inference")
        raise HTTPException(status_code=500, detail="inference request failed") from e

@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    port = int(os.getenv("PORT", default=8000))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104
