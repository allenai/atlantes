"""Test for the AtlasActivityPreprocessor class."""

import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import ATLAS_COLUMNS_WITH_META
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.datamodels import \
    PreprocessedActivityData
from atlantes.inference.atlas_activity.preprocessor import \
    AtlasActivityPreprocessor
from pandera.typing import DataFrame


@pytest.fixture(scope="class")
def test_track_data(test_ais_df1: pd.DataFrame) -> DataFrame[TrackfileDataModelTrain]:
    """Create test track data for preprocessing."""
    return test_ais_df1[ATLAS_COLUMNS_WITH_META].head(1000).copy()


class TestAtlasActivityPreprocessor:
    """Test the AtlasActivityPreprocessor class."""

    def test_preprocess_returns_preprocessed_activity_data(
        self, test_track_data: DataFrame[TrackfileDataModelTrain]
    ) -> None:
        """Test that preprocess returns a PreprocessedActivityData object."""
        preprocessed_data = AtlasActivityPreprocessor.preprocess(test_track_data)
        assert isinstance(preprocessed_data, PreprocessedActivityData)

    def test_preprocess_inputs_has_required_fields(
        self, test_track_data: DataFrame[TrackfileDataModelTrain]
    ) -> None:
        """Test that preprocessed inputs has required fields."""
        preprocessed_data = AtlasActivityPreprocessor.preprocess(test_track_data)
        assert hasattr(preprocessed_data.inputs, "traj_array")
        assert hasattr(preprocessed_data.inputs, "spatiotemporal_intervals")

    def test_preprocess_metadata_has_required_fields(
        self, test_track_data: DataFrame[TrackfileDataModelTrain]
    ) -> None:
        """Test that preprocessed metadata has required fields."""
        preprocessed_data = AtlasActivityPreprocessor.preprocess(test_track_data)
        required_fields = [
            "flag_code",
            "binned_ship_type",
            "entity_name",
            "trackId",
            "file_location",
            "send_time",
            "track_length",
            "most_recent_data",
            "dataset_membership",
        ]
        for field in required_fields:
            assert hasattr(preprocessed_data.metadata, field)

    def test_preprocess_activity_label_shape_is_empty(
        self, test_track_data: DataFrame[TrackfileDataModelTrain]
    ) -> None:
        """Test that activity label has correct shape."""
        preprocessed_data = AtlasActivityPreprocessor.preprocess(test_track_data)
        assert len(preprocessed_data.activity_label.shape) == 1

    def test_preprocess_handles_empty_dataframe(self) -> None:
        """Test that preprocess handles empty dataframe appropriately."""
        empty_df = pd.DataFrame(columns=ATLAS_COLUMNS_WITH_META)
        with pytest.raises(Exception):
            AtlasActivityPreprocessor.preprocess(empty_df)
