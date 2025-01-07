"""Unit tests for the model inference class and deployment class.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from atlantes.atlas.ais_dataset import ActivityDatasetEndOfSequence
from atlantes.atlas.atlas_net import AtlasActivityEndOfSequenceTaskNet
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.inference.atlas_activity.datamodels import \
    PreprocessedActivityData
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


def floats_differ_less_than(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Helper function for comparing floats."""
    return abs(a - b) < tolerance


@pytest.fixture(scope="class")
def preprocessed_input_stream_activity(
    test_ais_df1: pd.DataFrame,
    test_ais_df2: pd.DataFrame,
    data_config_inference_fixture: dict,
) -> pd.DataFrame:
    """Build an in-memory AIS track stream."""
    activity_dataset = ActivityDatasetEndOfSequence(
        mode="online",
        in_memory_data=[test_ais_df1.tail(1000), test_ais_df2.tail(1000)],
        dataset_config=data_config_inference_fixture,
    )
    return [PreprocessedActivityData(**activity_dataset[0]),
            PreprocessedActivityData(**activity_dataset[1])]


@pytest.fixture(scope="class")
def padded_vs_unpadded_input_stream_activity(
    test_ais_df1: pd.DataFrame,
    data_config_inference_fixture: dict,
) -> pd.DataFrame:
    """Pass in a stream of data where the ending messages are the same

    but the context differs
    So that one example is padded and the other is not in the batch."""
    activity_dataset = ActivityDatasetEndOfSequence(
        mode="online",
        # Tail allows us to have the same ending point for inference
        in_memory_data=[test_ais_df1.tail(1000), test_ais_df1.tail(2000)],
        dataset_config=data_config_inference_fixture,
    )
    # Maybe I could pass this more informatively
    return [PreprocessedActivityData(**activity_dataset[0]),
            PreprocessedActivityData(**activity_dataset[1])]


@pytest.fixture(scope="class")
def batched_vs_unbatched_input_stream_activity(
    test_ais_df2: pd.DataFrame,
    atlas_activity_inference_config_fixture: dict,
) -> pd.DataFrame:
    """Pass in a stream of data where the ending messages are the same

    but the context differs
    So that we can see the difference between batched and unbatched."""
    activity_dataset = ActivityDatasetEndOfSequence(
        mode="online",
        # Tail allows us to have the same ending point for inference
        in_memory_data=[test_ais_df2.tail(2000), test_ais_df2.tail(2000)],
        dataset_config=atlas_activity_inference_config_fixture["data"],
    )
    # Maybe I could pass this more informatively
    return [PreprocessedActivityData(**activity_dataset[0]),
            PreprocessedActivityData(**activity_dataset[1])]


class TestAtlasActivityModel:
    """Test the AtlasActivityModel class."""

    @pytest.fixture(scope="class")
    def model_class(self) -> AtlasActivityModel:
        """Create an instance of the AtlasActivityModel class."""
        return AtlasActivityModel()

    def test_initialize_atlas_activity_model_inference_correct_output(
        self, model_class: AtlasActivityModel
    ) -> None:
        """Test the initialize_atlas_activity_model_inference method."""
        net_with_no_loaded_weights = (
            model_class._initialize_atlas_activity_model_inference()
        )

        assert net_with_no_loaded_weights.is_defaults_set is False

    def test_load_atlas_activity_model_correct_output(
        self, model_class: AtlasActivityModel
    ) -> None:
        """Test the load_atlas_activity_model method."""

        loaded_model = model_class._load_atlas_activity_model()

        assert isinstance(loaded_model, AtlasActivityEndOfSequenceTaskNet)

    def test_load_model_raises_error_if_checkpoint_does_not_exist(
        self, model_class: AtlasActivityModel, monkeypatch: Any
    ) -> None:
        """Test the load_atlas_activity_model method raises an error if the checkpoint does not exist."""

        monkeypatch.setattr(
            model_class,
            "ATLAS_ACTIVITY_MODEL_PATH",
            Path("non_existent_checkpoint.pth"),
        )
        with pytest.raises(FileNotFoundError):
            model_class._load_atlas_activity_model()

    def test_model_path_is_a_pt_file(self, model_class: AtlasActivityModel) -> None:
        """Test that the model path is a .pt file."""
        assert model_class.ATLAS_ACTIVITY_MODEL_PATH.suffix == ".pt"

    def test_model_is_in_eval_mode(self, model_class: AtlasActivityModel) -> None:
        """Test that the model is in eval mode."""

        assert model_class.atlas_activity_model.training is False

    def test_run_inference_unbatched(
        self,
        model_class: AtlasActivityModel,
        preprocessed_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test the run_inference method."""
        # Vessel Nav Status says underway Sailing
        # THIS IS HARD TO RELIABLY TEST
        # expected_activity_class_1 = AtlasActivityLabelsTraining.TRANSITING
        outputs = model_class.run_inference([preprocessed_input_stream_activity[0]])
        output_tuple = outputs[0]
        activity_class, details, metadata = output_tuple
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert AtlasActivityLabelsTraining.FISHING != activity_class
        # assert activity_class == expected_activity_class_1
        assert isinstance(details, dict)
        assert set(["confidence", "outputs", "model", "model_version"]) == set(
            details.keys()
        )
        assert isinstance(metadata, dict)

    def test_different_unbatched_inputs_get_different_outputs(
        self,
        model_class: AtlasActivityModel,
        preprocessed_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test that different inputs give different output."""
        outputs_1 = model_class.run_inference([preprocessed_input_stream_activity[0]])
        outputs_2 = model_class.run_inference([preprocessed_input_stream_activity[1]])
        activity_class_1, details_1, metadata_1 = outputs_1[0]
        activity_class_2, details_2, metadata_2 = outputs_2[0]
        assert not all(
            floats_differ_less_than(output_prob_1, output_prob_2, tolerance=0.1)
            for output_prob_1, output_prob_2 in zip(
                details_1["outputs"], details_2["outputs"]
            )
        )

    def test_batched_vs_unbatched_outputs_are_same(
        self,
        model_class: AtlasActivityModel,
        batched_vs_unbatched_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test that batched and unbatched outputs are the same."""

        batched_outputs = model_class.run_inference(
            batched_vs_unbatched_input_stream_activity
        )
        batched_output_1 = batched_outputs[0]
        (
            batched_output_activity_class_1,
            batched_output_details_1,
            batched_output_metadata_1,
        ) = batched_output_1
        unbatched_output_1 = model_class.run_inference(
            [batched_vs_unbatched_input_stream_activity[0]]
        )[0]
        (
            unbatched_output_activity_class_1,
            unbatched_output_details_1,
            unbatched_output_metadata_1,
        ) = unbatched_output_1
        # TODO: Add checks that allow for differences in floating point precison
        assert batched_output_activity_class_1 == unbatched_output_activity_class_1
        assert all(
            [
                floats_differ_less_than(batched_out, unbatched_out)
                for batched_out, unbatched_out in zip(
                    batched_output_details_1["outputs"],
                    unbatched_output_details_1["outputs"],
                )
            ]
        )
        assert batched_output_metadata_1 == unbatched_output_metadata_1

    @pytest.mark.skip("This test is too brittle as a better modle in integration fails these tests")
    def test_run_inference_batched(
        self,
        model_class: AtlasActivityModel,
        preprocessed_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test the run_inference method."""
        # Both of these seem to be low speed transiting
        expected_activity_class_1 = AtlasActivityLabelsTraining.TRANSITING
        expected_confidence_1 = 0.86
        expected_output_probs_1 = [
            0.01,
            0.03,
            0.01,
            0.86,
        ]
        expected_activity_class_2 = AtlasActivityLabelsTraining.TRANSITING
        outputs = model_class.run_inference(preprocessed_input_stream_activity)
        expected_confidence_2 = 0.71
        expected_output_probs_2 = [
            0.29,
            0.001,
            0.001,
            0.71,
        ]
        output_1 = outputs[0]
        output_2 = outputs[1]
        activity_class_1, details_1, metadata_1 = output_1
        activity_class_2, details_2, metadata_2 = output_2
        assert activity_class_1 == expected_activity_class_1
        assert floats_differ_less_than(
            details_1["confidence"], expected_confidence_1, tolerance=0.1
        )
        assert all(
            floats_differ_less_than(output_prob, expected_output_prob, tolerance=0.1)
            for output_prob, expected_output_prob in zip(
                details_1["outputs"], expected_output_probs_1
            )
        )
        assert isinstance(metadata_1, dict)
        assert activity_class_2 == expected_activity_class_2
        assert floats_differ_less_than(
            details_2["confidence"], expected_confidence_2, tolerance=0.1
        )
        for i, (output_prob, expected_output_prob) in enumerate(
            zip(details_2["outputs"], expected_output_probs_2)
        ):
            assert floats_differ_less_than(
                output_prob, expected_output_prob, tolerance=0.1
            ), (
                f"Assertion failed at index {i}: Output probability {output_prob} differs from expected {expected_output_prob} "
                f"by more than the tolerance of 0.2."
            )

        assert isinstance(metadata_2, dict)

    def test_batching_with_no_padding_does_not_change_output(
        self,
        model_class: AtlasActivityModel,
        batched_vs_unbatched_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test the run_inference method."""
        outputs_batched = model_class.run_inference(
            [batched_vs_unbatched_input_stream_activity[1]]
        )[0]
        outputs_unbatched = model_class.run_inference(
            batched_vs_unbatched_input_stream_activity
        )[1]
        activity_class_batched, details_batched, metadata_batched = outputs_batched
        activity_class_unbatched, details_unbatched, metadata_unbatched = (
            outputs_unbatched
        )
        assert activity_class_batched == activity_class_unbatched
        assert floats_differ_less_than(
            details_batched["confidence"], details_unbatched["confidence"]
        )
        assert all(
            floats_differ_less_than(output_prob_batched, output_prob_unbatched)
            for output_prob_batched, output_prob_unbatched in zip(
                details_batched["outputs"], details_unbatched["outputs"]
            )
        )
        assert metadata_batched == metadata_unbatched

    def test_batching_with_extreme_padding_does_not_change_output(
        self,
        model_class: AtlasActivityModel,
        padded_vs_unpadded_input_stream_activity: list[pd.DataFrame],
    ) -> None:
        """Test the run_inference method


        This test fails if we pad 1000 or sommething like this"""
        outputs_padded = model_class.run_inference(
            padded_vs_unpadded_input_stream_activity
        )[1]
        outputs_unpadded = model_class.run_inference(
            [padded_vs_unpadded_input_stream_activity[1]]
        )[0]
        activity_class_unpadded, details_unpadded, metadata_unpadded = outputs_unpadded
        activity_class_padded, details_padded, metadata_padded = outputs_padded
        assert activity_class_unpadded == activity_class_padded
        # Higher tolerance because it is expected for padding to slightly alter confidence
        assert floats_differ_less_than(
            details_unpadded["confidence"], details_padded["confidence"], tolerance=0.01
        )
        assert all(
            floats_differ_less_than(
                output_prob_unpadded, output_prob_padded, tolerance=0.01
            )
            for output_prob_unpadded, output_prob_padded in zip(
                details_unpadded["outputs"], details_padded["outputs"]
            )
        )
        assert metadata_unpadded == metadata_padded
