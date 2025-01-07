"""Unit tests for the entity preprocessor

Tests the preprocessing functionality for AIS trajectory entity classification
"""

import numpy as np
import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import ATLAS_COLUMNS_WITH_META
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.datamodels import PreprocessedEntityData
from atlantes.inference.atlas_entity.preprocessor import \
    AtlasEntityPreprocessor
from pandera.errors import SchemaError as PanderaSchemaError
from pandera.typing import DataFrame


@pytest.fixture(scope="class")
def sample_track_data(
    test_ais_df1: pd.DataFrame,
) -> DataFrame[TrackfileDataModelTrain]:
    """Create test track data for preprocessing."""
    return test_ais_df1[ATLAS_COLUMNS_WITH_META].head(1000).copy()


def test_preprocess_valid_input(
    sample_track_data: DataFrame[TrackfileDataModelTrain]
) -> None:
    """Test preprocessing with valid input data"""
    # Preprocess the data
    result = AtlasEntityPreprocessor.preprocess(sample_track_data)

    # Check output type
    assert isinstance(result, PreprocessedEntityData)


def test_preprocess_empty_input() -> None:
    """Test preprocessing with empty input data"""
    empty_df = pd.DataFrame(columns=ATLAS_COLUMNS_WITH_META)

    with pytest.raises(IndexError):
        AtlasEntityPreprocessor.preprocess(empty_df)


def test_preprocess_missing_columns() -> None:
    """Test preprocessing with missing required columns"""
    invalid_data = pd.DataFrame({
        "send": pd.date_range(start="2020-01-01", periods=5, freq="H"),
        "lat": np.random.uniform(low=0, high=90, size=5),
        # Missing other required columns
    })

    with pytest.raises(PanderaSchemaError):
        AtlasEntityPreprocessor.preprocess(invalid_data)
