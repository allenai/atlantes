"""Data models for the atlas entity application

These are used to define the input and output of the entity postprocessor and other
steps as needed
"""

from typing import NamedTuple, Optional

import torch
from atlantes.atlas.atlas_utils import AtlasEntityLabelsTrainingWithUnknown
from pydantic import BaseModel, Field, field_validator


class PreprocessedEntityMetadata(BaseModel):
    """Metadata about the preprocessed entity"""

    flag_code: str
    binned_ship_type: int
    ais_type: int
    entity_name: str
    trackId: str
    file_location: Optional[str] = None
    mmsi: str
    track_length: int


class PreprocessedEntityInputs(BaseModel):
    """Model inputs after preprocessing"""

    traj_tensor: torch.Tensor = Field(description="Preprocessed trajectory features")
    spatiotemporal_intervals: torch.Tensor = Field(
        description="Spatiotemporal intervals tensor"
    )

    class Config:
        arbitrary_types_allowed = True


class PreprocessedEntityTargets(BaseModel):
    """Model targets after preprocessing"""

    class_id: torch.Tensor = Field(description="Entity class target labels")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("class_id")
    @classmethod
    def validate_class_id(cls, v: torch.Tensor) -> torch.Tensor:
        """Validate the class id"""
        if v.shape[0] != 0:
            raise ValueError("Class id array must be empty")
        return v


class PreprocessedEntityData(BaseModel):
    """Complete preprocessed data for entity classification"""

    metadata: PreprocessedEntityMetadata
    inputs: PreprocessedEntityInputs
    targets: PreprocessedEntityTargets

    class Config:
        arbitrary_types_allowed = True


class EntityPostprocessorInputDetails(NamedTuple):
    """Details of the entity classification"""

    model: str
    confidence: float
    outputs: list[float]


class EntityPostprocessorOutputDetails(NamedTuple):
    """Details of the entity classification"""

    predicted_classification: str
    model: str
    confidence: float
    outputs: list[float]
    postprocessed_classification: str
    postprocess_rule_applied: bool
    confidence_threshold: float


class EntityMetadata(NamedTuple):
    """Metadata of the entity"""

    binned_ship_type: Optional[int]
    ais_type: int  # AIS category reported by the AIS message
    flag_code: str
    entity_name: str
    track_length: int
    mmsi: str
    trackId: str
    file_location: Optional[str]


class EntityPostprocessorOutput(NamedTuple):
    """Output of the entity postprocessor"""

    track_id: str
    entity_class: str
    entity_classification_details: EntityPostprocessorOutputDetails

    def serialize(self) -> tuple[str, dict, str]:
        postprocessed_classification_details = self.entity_classification_details
        entity_classification_details = {
            "predicted_classification": postprocessed_classification_details.predicted_classification,
            "model": postprocessed_classification_details.model,
            "confidence": postprocessed_classification_details.confidence,
            "outputs": postprocessed_classification_details.outputs,
            "postprocessed_classification": postprocessed_classification_details.postprocessed_classification,
            "postprocess_rule_applied": postprocessed_classification_details.postprocess_rule_applied,
            "confidence_threshold": postprocessed_classification_details.confidence_threshold,
        }
        return (
            self.entity_class,
            entity_classification_details,
            self.track_id,
        )


class EntityPostprocessorInput(NamedTuple):
    """Input to the entity postprocessor"""

    predicted_class: AtlasEntityLabelsTrainingWithUnknown
    entity_classification_details: EntityPostprocessorInputDetails
    metadata: EntityMetadata
