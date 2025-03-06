"""Integration tests for the activity pipeline."""

import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import (
    ATLAS_COLUMNS_WITH_META,
    get_atlas_activity_inference_config,
)
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.inference.common import ATLASResponse
from atlantes.log_utils import get_logger

from ais.src.main_activity import AtlasActivityClassifier

logger = get_logger(__name__)


@pytest.fixture(scope="class")
def in_memory_ais_track_df(test_ais_df1: pd.DataFrame) -> pd.DataFrame:
    """Build an in-memory AIS track stream with CPD subpaths."""
    test_ais_df1_inference = test_ais_df1[ATLAS_COLUMNS_WITH_META].head(4000).copy()
    return test_ais_df1_inference


@pytest.fixture(scope="class")
def activity_classifier_pipeline() -> AtlasActivityClassifier:
    """Return the activity classifier pipeline."""
    return AtlasActivityClassifier(
        AtlasActivityPreprocessor(),
        AtlasActivityModel(),
        AtlasActivityPostProcessor(),
    )


class TestActivityClassifier:
    """Tests for the pipelines."""

    def test_activity_pipeline(
        self,
        in_memory_ais_track_df: pd.DataFrame,
        activity_classifier_pipeline: AtlasActivityClassifier,
    ) -> None:
        """Test the activity pipeline."""
        tracks = [in_memory_ais_track_df.to_dict(orient="records")]
        batched_output = activity_classifier_pipeline.run_pipeline(tracks)
        predicted_class_name, predicted_details = batched_output[0].predictions
        # Check that the output is as expected

        # Asserting based on last subpath outputs for type checking
        assert isinstance(batched_output, ATLASResponse)
        assert isinstance(predicted_class_name, str)
        assert isinstance(predicted_details, dict)
        assert predicted_details.keys() == {
            "model",
            "confidence",
            "outputs",
            "model_version",
            "original_classification",
            "postprocessed_classification",
            "rule_applied",
        }
        assert isinstance(predicted_details["model"], str)
        assert (
            predicted_details["confidence"] > 0.0
            and predicted_details["confidence"] <= 1.0
        )
        assert isinstance(predicted_details["outputs"], list)
        assert isinstance(
            predicted_details["confidence"], float
        )  # SHould just be a float but would need to change contract
        assert len(predicted_details["outputs"]) == 4  # num classes

        # Asserting for Real-time Simulated outputs
        logger.warning("This is not a regression test, the output may change.")

        inference_config = get_atlas_activity_inference_config()
        model_id = inference_config["model"]["ATLAS_ACTIVITY_MODEL_ID"]
        assert predicted_details["model"] == model_id

    @pytest.mark.parametrize(
        "df_list, expected_label",
        [
            ("obvious_anchored_df_list", "anchored"),
            ("obvious_moored_df_list", "moored"),
            ("obvious_transit_df_list", "transiting"),
            ("near_shore_df_list", "unknown"),
            ("stationary_df_list", "moored"),
            ("non_fishing_or_unknown_df_list", "transiting"),
            ("transiting_df_list", "transiting"),
            ("fishing_df_list", "fishing"),
            ("high_traffic_port_df_list", "unknown"),
            ("offshore_supply_vessels_df_list", "transiting"),
        ],
        ids=[
            "Pipeline for obvious anchored vessels",
            "Pipeline for obvious moored vessels",
            "Pipeline for obvious transiting vessels",
            "Pipeline for near shore vessels",
            "Pipeline for stationary vessels",
            "Pipeline for non-fishing or unknown vessels",
            "Pipeline for unknown vessels likely transiting",
            "Pipeline for fishing vessels",
            "Pipeline for known high traffic port false positives",
            "Pipeline for offshore marine infrastructure transiting vessels",
        ],
    )
    def test_activity_pipeline_postprocessing(
        self,
        activity_classifier_pipeline: AtlasActivityClassifier,
        request: pytest.FixtureRequest,
        df_list: list[pd.DataFrame],
        expected_label: str,
    ) -> None:
        """Test the activity pipeline with different debugging data."""

        # Retrieve the df_list fixture name
        df_list_name = request.node.callspec.params["df_list"]

        # Fetch the fixture for the df_list (list of DataFrames)
        df_list = request.getfixturevalue(df_list_name)

        output_lst = []
        for i in range(0, len(df_list)):
            tracks = [df_list[i].to_dict(orient="records")]
            output = activity_classifier_pipeline.run_pipeline(tracks)
            output_lst.append(output[0].predictions)

        predicted_class_names_lst = [
            predicted_class_name for predicted_class_name, _ in output_lst
        ]
        predicted_class_details_lst = [
            predicted_class_details for _, predicted_class_details in output_lst
        ]

        logger.info(predicted_class_details_lst)
        logger.info(predicted_class_names_lst)

        # Track failures with the specific dataframe filename (df_list_name)
        assert all(
            [
                predicted_class_name == expected_label
                for predicted_class_name in predicted_class_names_lst
            ]
        ), f"All predicted class names should be {expected_label} for {df_list_name}, instead got {predicted_class_names_lst}"

        assert all(
            [
                "original_classification" in predicted_class_details.keys()
                for predicted_class_details in predicted_class_details_lst
            ]
        )
        assert all(
            [
                "postprocessed_classification" in predicted_class_details.keys()
                for predicted_class_details in predicted_class_details_lst
            ]
        )
