"""Pandera Schemas for human annotation data."""

import pandas as pd
import pandera as pa
from atlantes.datautils import BaseDataModel
from pandera.typing import Index, Series


class AnnotatedDownsampledAISTrackDataFrameModel(BaseDataModel):
    """Pandera dataframe model for the annotated downsampled AIS track dataframe."""

    # Need to coerce specific format # add something to confirm date format
    send: Series[pd.Timestamp] = pa.Field(coerce=True)  # does not work wrong object
    lat: Series[float]
    lon: Series[float]
    mmsi: Series[int]
    category: Series[int]
    trackId: Series[str]
    nav: Series[int] = pa.Field(coerce=True)
    name: Series[str] = pa.Field(coerce=True)
    sog: Series[float]
    cog: Series[float]
    annotator_uid: Series[str]
    annotator_username: Series[str]
    annotated_at: Series[str]
    object_type_name: Series[str]
    object_type_subtype: Series[str]
    object_type_confidence: Series[float] = pa.Field(
        coerce=True, nullable=True
    )  # does not coerce int to float
    activity_type_name: Series[str]
    activity_type_subtype: Series[str] = pa.Field(nullable=True)
    activity_type_confidence: Series[float] = pa.Field(
        coerce=True
    )  # does not coerce int to float


class AnnotationIdentifierDataModel(BaseDataModel):
    """Pandera Schema for the identifying charecteristics of an annotation"""

    trackId: Series[str]
    month: Series[int]
    year: Series[int]
    annotator_username: Series[str]
    path: Series[str]


class TrackIdentifierDataModel(BaseDataModel):
    """Pandera Schema for the identifying a downsampled track to the original raw data"""

    trackId: Series[str]
    month: Series[int]
    year: Series[int]


class SingleTrackActivityTypeLabelUnreducedDataModel(BaseDataModel):
    """Pandera dataframe model for the annotations per subpath of an individual track

    This is for BEFORE redcuing the labels to a single label per subpath
    Multiple lables per subpath are allowed
    Missing Subpath nums are allowed
    """

    activity_types: Series[list[str]]
    subpath_num: Index[int]


class SingleTrackActivityTypeLabelDataModel(BaseDataModel):
    """Pandera dataframe model for the annotations per subpath of an individual track

    This is for AFTER redcuing the labels to a single label per subpath
    Multiple lables per subpath are not allowed
    Missing Subpath nums are not allowed
    """

    activity_type: Series[str]
    subpath_num: Index[int]


class LocationDataModel(BaseDataModel):
    """Pandera dataframe model for the location data model."""

    lat: Series[float]
    lon: Series[float]


class TrackfileDataModel(BaseDataModel):
    """DataModel for trackfiles for training

    These include all columns that would be loaded into memory
    for training with the ATLAS model and dataset classes
    """

    # geometry: Series[Point] not supported by pandera
    mmsi: Series[int]
    category: Series[int]
    trackId: Series[str]
    flag_code: Series[str]
    vessel_class: Series[str]
    dist2coast: Series[float]
    lat: Series[float]
    lon: Series[float]
    send: Series[pd.Timestamp] = pa.Field(coerce=True)
    sog: Series[float]
    cog: Series[float]
    nav: Series[int]
    subpath_num: Series[int]
    name: Series[str] = pa.Field(coerce=True)


class TrackMetadataIndex(BaseDataModel):
    file_name: Series[str]
    year: Series[str]
    flag_code: Series[str]
    month: Series[str]
    ais_type: Series[int]
    trackId: Series[str]
    Path: Series[str]
