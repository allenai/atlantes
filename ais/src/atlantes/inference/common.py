"""Common functions and objects for inference.
"""

from pydantic import BaseModel


class ATLASRequest(BaseModel):
    """Request object for ATLAS"""

    tracks: list[list[dict]]


class ATLASResponse(BaseModel):
    """Request object for ATLAS"""

    predictions: list[tuple[str, dict]]
    num_failed_preprocessing: int
    num_failed_postprocessing: int


class AtlasInferenceError(Exception):
    """Raised when an error occurs during ATLAS inference."""
    pass

class InfoResponse(BaseModel):
    model_type: str
    git_commit_hash: str
