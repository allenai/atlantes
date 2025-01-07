"""Pandera schemas for ATLAS data.

Use schema.validate(df) instead of datamodel(df) for explicit validation

Use DataFrameModel with check types and type annotations where possible
Verify dataframe model works in parallel when using DDP and multiple workers
"""

from typing import Optional

import pandas as pd
from atlantes.datautils import BaseDataModel
from pandera.typing import Index, Series


class TrackfileDataModelTrain(BaseDataModel):
    """DataModel for trackfiles for training

    These include all columns that would be loaded into memory
    for training with the ATLAS model and dataset classes
    """

    lat: Series[float]
    lon: Series[float]
    send: Series[pd.Timestamp]
    sog: Series[float]
    cog: Series[float]
    nav: Series[int]
    subpath_num: Optional[Series[int]]  # TODO: eventually we should remove from entity as well
    dist2coast: Series[float]
    name: Series[str]
    flag_code: Series[str]
    category: Series[int]
    trackId: Series[str]
    mmsi: Series[int]


class TrajectoryLengthsDataModel(BaseDataModel):
    """DataModel for trajectory lengths files

    DataModel for file containing the lengths of trajectories
    """

    Length: Series[int]
    Path: Series[str]


TrajectoryLengthsSchema = TrajectoryLengthsDataModel.to_schema()


class EntityClassLabelDataModel(BaseDataModel):
    """DataModel for entity class label files

    DataModel for file containing the entity class labels
    """

    Path: Index[str]
    entity_class_label: Series[int]

class ActivityEndOfSequenceLabelDataModel(BaseDataModel):
    """DataModel for activity labels for end of sequence activity

    This contains the columns for the df that is read/written and contains the labels
    for the end of sequence activity task
    """

    raw_paths: Series[list[str]]
    downsampled_path: Optional[Series[str]]
    trackId: Series[str]
    activity_type_name: Series[str]
    send: Series[pd.Timestamp]
    lat: Optional[Series[float]]
    lon: Optional[Series[float]]


class FeedbackActivityLabelDataModel(ActivityEndOfSequenceLabelDataModel):
    """DataModel for feedback activity labels

    DataModel for file containing the feedback activity labels
    """
    predicted_activity_label: Series[str]
    subpath_id: Series[str]
    subpath_start_time: Series[pd.Timestamp]


class ActivityEndOfSequenceLabelNoContextDataModel(BaseDataModel):
    """DataModel for activity labels for end of sequence activity
    before adding context
    """

    Path: Series[str]
    downsampled_path: Optional[Series[str]]
    trackId: Series[str]
    activity_type_name: Series[str]
    send: Series[pd.Timestamp]
    lat: Optional[Series[float]]
    lon: Optional[Series[float]]
