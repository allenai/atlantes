"""Common functions and objects for inference.
"""

from pydantic import BaseModel


class ATLASRequest(BaseModel):
    """Request object for ATLAS"""

    track: list[dict]


class ATLASResponse(BaseModel):
    """Request object for ATLAS"""

    predictions: tuple[str, dict]


class AtlasInferenceError(Exception):
    """Raised when an error occurs during ATLAS inference."""
    pass
