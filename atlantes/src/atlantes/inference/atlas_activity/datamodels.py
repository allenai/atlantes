"""Data models for the atlas activity inference pipeline"""

from typing import Any, Optional

import numpy as np
from pandas import DataFrame as PandasDataFrame
from pydantic import BaseModel, field_validator


class PreprocessorBaseModel(BaseModel):
    """Base class for preprocessor models"""

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def validate_numpy_arrays(v: Any) -> np.ndarray:
        """Validate the numpy arrays"""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be a numpy array")
        return v

    @staticmethod
    def validate_pandas_dataframe(v: Any) -> PandasDataFrame:
        """Validate the pandas dataframe"""
        if not isinstance(v, PandasDataFrame):
            raise ValueError("Must be a pandas DataFrame")
        return v


class ActivityInput(PreprocessorBaseModel):
    """Pydantic model for the activity input"""

    traj_array: np.ndarray
    spatiotemporal_intervals: np.ndarray

    @field_validator("traj_array", "spatiotemporal_intervals")
    @classmethod
    def validate_arrays(cls, v: np.ndarray) -> np.ndarray:
        """Validate numpy arrays"""
        return cls.validate_numpy_arrays(v)


class ActivityMetadata(PreprocessorBaseModel):
    """Pydantic model for the activity metadata"""

    flag_code: str
    binned_ship_type: int
    entity_name: str
    trackId: str
    file_location: Optional[str]
    send_time: Optional[str]
    track_length: int
    most_recent_data: PandasDataFrame
    dataset_membership: str

    @field_validator("most_recent_data")
    @classmethod
    def validate_most_recent_data(cls, v: PandasDataFrame) -> PandasDataFrame:
        """Validate the most recent data"""
        return cls.validate_pandas_dataframe(v)


class PreprocessedActivityData(PreprocessorBaseModel):
    """Pydantic model for the preprocessed activity data"""

    inputs: ActivityInput
    metadata: ActivityMetadata
    enough_context: bool
    activity_label: np.ndarray

    @field_validator("activity_label")
    @classmethod
    def validate_activity_label(cls, v: np.ndarray) -> np.ndarray:
        """Validate the activity label"""
        array = cls.validate_numpy_arrays(v)
        if array.shape[0] != 0:
            raise ValueError("Activity label array must be empty")
        return array
