"""Common functions and objects for inference.
"""

from pydantic import BaseModel


class TrackData(BaseModel):
    track_id: str
    track_data: list[dict]


class ATLASRequest(BaseModel):
    """Request object for ATLAS"""

    track_data: list[TrackData]


class PreprocessFailure(BaseModel):
    track_id: str
    error: str


class PostprocessFailure(BaseModel):
    track_id: str
    classification: str
    error: str

class Prediction(BaseModel):
    track_id: str
    classification: str
    details: dict

class ATLASResponse(BaseModel):
    """Request object for ATLAS"""

    predictions: list[Prediction]
    preprocess_failures: list[PreprocessFailure]
    postprocess_failures: list[PostprocessFailure]


class AtlasInferenceError(Exception):
    """Raised when an error occurs during ATLAS inference."""
    pass

class InfoResponse(BaseModel):
    model_type: str
    git_commit_hash: str
