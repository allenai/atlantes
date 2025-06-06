"""Integration tests for the entity pipeline."""

import numpy as np
import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import (
    ATLAS_COLUMNS_WITH_META,
    get_atlas_entity_inference_config,
)
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.classifier import (
    AtlasEntityClassifier,
    PipelineInput,
)
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor,
    KnownShipTypeAndBuoyName,
)
from atlantes.inference.atlas_entity.preprocessor import AtlasEntityPreprocessor
from atlantes.log_utils import get_logger
from pandera.typing import DataFrame

logger = get_logger(__name__)


@pytest.fixture(scope="class")
def in_memory_ais_track_df(
    test_ais_df1: pd.DataFrame,
) -> DataFrame[TrackfileDataModelTrain]:
    """Build an in-memory AIS track stream with CPD subpaths."""
    test_ais_df1_inference = test_ais_df1[ATLAS_COLUMNS_WITH_META].head(4000).copy()
    test_ais_df1_inference = DataFrame[TrackfileDataModelTrain](test_ais_df1_inference)
    assert isinstance(test_ais_df1_inference, DataFrame)
    return test_ais_df1_inference


@pytest.fixture(scope="class")
def buoy_df(test_buoy_df: pd.DataFrame) -> DataFrame[TrackfileDataModelTrain]:
    """Build an in-memory AIS track stream."""
    test_buoy_df_inference = test_buoy_df[ATLAS_COLUMNS_WITH_META].copy()
    test_buoy_df_inference = DataFrame[TrackfileDataModelTrain](test_buoy_df_inference)
    assert isinstance(test_buoy_df_inference, DataFrame)
    return test_buoy_df_inference


@pytest.fixture(scope="class")
def entity_classifier_pipeline() -> AtlasEntityClassifier:
    """Return the entity classifier pipeline."""
    return AtlasEntityClassifier(
        AtlasEntityPreprocessor(),
        AtlasEntityModel(),
        AtlasEntityPostProcessor(),
    )


class TestEntityClassifier:
    """Tests for the pipelines."""

    def test_entity_pipeline(
        self,
        in_memory_ais_track_df: DataFrame[TrackfileDataModelTrain],
        entity_classifier_pipeline: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        tracks = [PipelineInput(track_id="test", track_data=in_memory_ais_track_df)]
        output = entity_classifier_pipeline.run_pipeline(tracks)
        assert len(output.predictions) == 1, f"expected 1 prediction, got {output=}"

    def test_entity_pipeline_vessel(
        self,
        in_memory_ais_track_df: DataFrame[TrackfileDataModelTrain],
        entity_classifier_pipeline: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        tracks = [PipelineInput(track_id="test", track_data=in_memory_ais_track_df)]
        output = entity_classifier_pipeline.run_pipeline(tracks)
        assert len(output.predictions) == 1, f"expected 1 prediction, got {output=}"
        prediction = output.predictions[0]
        predicted_class_name = prediction.classification
        predicted_class_details = prediction.details
        inference_config = get_atlas_entity_inference_config()
        model_id = inference_config["model"]["ATLAS_ENTITY_MODEL_ID"]
        assert isinstance(predicted_class_name, str)
        assert predicted_class_name == "vessel"  # Buoy or vessel
        assert predicted_class_details["model"] == model_id
        assert (
            predicted_class_details["confidence"] > 0.9
            and predicted_class_details["confidence"] <= 1.0
        )
        assert isinstance(predicted_class_details["outputs"], list)
        assert len(predicted_class_details["outputs"]) == 2
        assert predicted_class_details["postprocessed_classification"] == "vessel"
        assert predicted_class_details["postprocess_rule_applied"] is False

    def test_entity_pipeline_buoy(
        self,
        buoy_df: DataFrame[TrackfileDataModelTrain],
        entity_classifier_pipeline: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        tracks = [PipelineInput(track_id="test", track_data=buoy_df)]
        output = entity_classifier_pipeline.run_pipeline(tracks)
        assert len(output.predictions) == 1, f"expected 1 prediction, got {output=}"
        inference_config = get_atlas_entity_inference_config()
        model_id = inference_config["model"]["ATLAS_ENTITY_MODEL_ID"]
        prediction = output.predictions[0]
        predicted_class_name = prediction.classification
        predicted_class_details = prediction.details
        logger.info(predicted_class_name)
        logger.info(predicted_class_details)
        logger.info(np.array(predicted_class_details["outputs"]).shape)
        assert isinstance(predicted_class_name, str)
        assert predicted_class_name == "buoy"
        assert predicted_class_details["model"] == model_id
        assert (
            predicted_class_details["confidence"] > 0.7
            and predicted_class_details["confidence"] <= 1.0
        )
        assert isinstance(predicted_class_details["outputs"], list)
        assert predicted_class_details["postprocessed_classification"] == "buoy"
        assert predicted_class_details["postprocess_rule_applied"] is True

    def test_entity_postprocessor_error_handling(
        self,
        buoy_df: DataFrame[TrackfileDataModelTrain],
        entity_classifier_pipeline: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        modified_buoy_df = buoy_df.copy()
        # pretend that something with a buoy name is listed as a fishing vessel
        modified_buoy_df.loc[:, "category"] = [30] * len(modified_buoy_df)
        try:
            tracks = [PipelineInput(track_id="test", track_data=modified_buoy_df)]
            entity_classifier_pipeline.run_pipeline(tracks)
        except Exception as e:
            assert isinstance(e.__cause__, KnownShipTypeAndBuoyName)
