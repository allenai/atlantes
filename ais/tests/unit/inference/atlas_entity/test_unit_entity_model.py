"""Unit tests for the model inference class and deployment class.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from atlantes.atlas.ais_dataset import AISTrajectoryEntityDataset
from atlantes.atlas.atlas_net import AtlasEntity
from atlantes.atlas.atlas_utils import AtlasEntityLabelsTrainingWithUnknown
from atlantes.inference.atlas_entity.datamodels import (EntityMetadata,
                                                        PreprocessedEntityData)
from atlantes.inference.atlas_entity.model import AtlasEntityModel
from atlantes.log_utils import get_logger
from atlantes.utils import floats_differ_less_than

logger = get_logger(__name__)


@pytest.fixture(scope="class")
def preprocessed_input_stream_entity(
    test_ais_df1: pd.DataFrame,
    test_ais_df2: pd.DataFrame,
    atlas_entity_inference_config_fixture: dict,
) -> pd.DataFrame:
    """Build an in-memory AIS track stream."""
    entity_dataset = AISTrajectoryEntityDataset(
        mode="online",
        in_memory_data=[test_ais_df1.head(4000), test_ais_df2],
        dataset_config=atlas_entity_inference_config_fixture["data"],
    )
    return [PreprocessedEntityData(**entity_dataset[0]),
            PreprocessedEntityData(**entity_dataset[1])]


class TestAtlasEntityModel:
    """Test the AtlasEntityModel class."""

    @pytest.fixture(scope="class")
    def model_class(
        self,
    ) -> AtlasEntityModel:
        """Create an instance of the AtlasEntityModel class."""
        return AtlasEntityModel()

    def test_initialize_atlas_entity_model_inference_correct_output(
        self, model_class: AtlasEntityModel
    ) -> None:
        """Test the initialize_atlas_entity_model_inference method."""

        net_with_no_loaded_weights = (
            model_class._initialize_atlas_entity_model_inference()
        )

        assert net_with_no_loaded_weights.is_defaults_set is False

    def test_load_atlas_entity_model_correct_output(
        self, model_class: AtlasEntityModel
    ) -> None:
        """Test the load_atlas_entity_model method."""

        loaded_model = model_class._load_atlas_entity_model()

        assert isinstance(loaded_model, AtlasEntity)

    def test_load_model_raises_error_if_checkpoint_does_not_exist(
        self, model_class: AtlasEntityModel, monkeypatch: Any
    ) -> None:
        """Test the load_atlas_entity_model method raises an error if the checkpoint does not exist."""

        monkeypatch.setattr(
            model_class,
            "ATLAS_ENTITY_MODEL_PATH",
            Path("non_existent_checkpoint.pth"),
        )
        with pytest.raises(FileNotFoundError):
            model_class._load_atlas_entity_model()

    def test_model_path_is_a_pt_file(self, model_class: AtlasEntityModel) -> None:
        """Test that the model path is a .pt file."""
        assert model_class.ATLAS_ENTITY_MODEL_PATH.suffix == ".pt"

    def test_model_is_in_eval_mode(self, model_class: AtlasEntityModel) -> None:
        """Test that the model is in eval mode."""

        assert model_class.atlas_entity_model.training is False

    def test_run_inference(
        self,
        model_class: AtlasEntityModel,
        preprocessed_input_stream_entity: list[pd.DataFrame],
    ) -> None:
        """Test the run_inference method on a single input."""
        expected_entity_class_1 = AtlasEntityLabelsTrainingWithUnknown.VESSEL
        expected_confidence_1 = 0.99
        expected_prob_outputs_1 = [0.99, 0.01]

        entity_outputs = model_class.run_inference(
            [preprocessed_input_stream_entity[0]]
        )[0]

        entity_class_1, details_1, metadata_1 = entity_outputs
        assert entity_class_1 == expected_entity_class_1
        assert floats_differ_less_than(details_1.confidence, expected_confidence_1)
        assert all(
            floats_differ_less_than(prob_output, expected_prob_output)
            for prob_output, expected_prob_output in zip(
                details_1.outputs, expected_prob_outputs_1
            )
        )
        assert isinstance(metadata_1, EntityMetadata)
