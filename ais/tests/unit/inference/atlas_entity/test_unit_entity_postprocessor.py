"""Unit tests for the entity postprocessor class."""

from typing import Any

import pytest
from atlantes.atlas.atlas_utils import AtlasEntityLabelsTrainingWithUnknown
from atlantes.inference.atlas_entity.datamodels import (
    EntityMetadata,
    EntityPostprocessorInput,
    EntityPostprocessorInputDetails,
)
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor,
    KnownShipTypeAndBuoyName,
)


class TestAtlasEntityPostProcessor:
    """Unit tests for the AtlasEntityPostProcessor class."""

    @pytest.fixture(scope="class")
    def entity_postprocessor_class(self) -> AtlasEntityPostProcessor:
        """Create an instance of the AtlasEntityPostProcessor class."""
        return AtlasEntityPostProcessor()

    @pytest.mark.parametrize(
        "binned_binned_ship_type",
        [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
    )
    def test_is_binned_ship_type_known_is_true_for_known_types(
        self,
        binned_binned_ship_type: int,
        entity_postprocessor_class: AtlasEntityPostProcessor,
    ) -> None:
        """Test the is_binned_ship_type_known method.

        See atlantes/config/AIS_CATEGORIES for the ship types.
        """
        assert entity_postprocessor_class.is_binned_ship_type_known(
            binned_binned_ship_type
        )

    def test_is_binned_ship_type_known_is_false_for_unknown_types(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the is_binned_ship_type_known method."""
        unknown_binned_ship_type = 0
        assert not entity_postprocessor_class.is_binned_ship_type_known(
            unknown_binned_ship_type
        )

    def test_is_binned_ship_type_known_returns_bool(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the is_binned_ship_type_known method returns a bool."""
        unknown_binned_ship_type = 0
        result = entity_postprocessor_class.is_binned_ship_type_known(
            unknown_binned_ship_type
        )
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    def test_check_confidence_threshold_returns_bool(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test that check_confidence_threshold method returns a bool."""
        confidence = 0.7
        predicted_class = AtlasEntityLabelsTrainingWithUnknown.VESSEL
        result, _ = entity_postprocessor_class.check_confidence_threshold(
            confidence, predicted_class
        )
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    @pytest.mark.parametrize("undefined_binned_ship_type", [-1, 210, 24, 1000])
    def test_is_binned_ship_type_known_raises_error_for_undefined_types(
        self,
        undefined_binned_ship_type: int,
        entity_postprocessor_class: AtlasEntityPostProcessor,
    ) -> None:
        """Test the is_binned_ship_type_known method."""
        with pytest.raises(ValueError):
            entity_postprocessor_class.is_binned_ship_type_known(
                undefined_binned_ship_type
            )

    def test_postprocess_happy_path(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method."""
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                ais_type=9999,
                mmsi="123456789",
                entity_name="unknown",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        entity_class = output.entity_class
        details = output.entity_classification_details
        assert (
            entity_class == "vessel"
            and details.postprocessed_classification == "vessel"
            and details.postprocess_rule_applied is False
        )

    def test_postprocess_with_only_ais_type_not_binned_ship_type(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method with only ais_type not binned_ship_type."""
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=None,
                ais_type=70,
                mmsi="123456789",
                entity_name="Cargo Ship",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        entity_postprocessor_output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        assert (
            entity_postprocessor_output.entity_class == "vessel"
            and entity_postprocessor_output.entity_classification_details.postprocessed_classification
            == "vessel"
            and entity_postprocessor_output.entity_classification_details.postprocess_rule_applied
            is True
        )

    def test_postprocess_with_only_unknown_ais_type_not_binned_ship_type(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method with only ais_type not binned_ship_type."""
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=None,
                ais_type=9999,
                mmsi="123456789",
                entity_name="NET, 10%",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        entity_postprocessor_output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        assert (
            entity_postprocessor_output.entity_class == "buoy"
            and entity_postprocessor_output.entity_classification_details.postprocessed_classification
            == "buoy"
            and entity_postprocessor_output.entity_classification_details.postprocess_rule_applied
            is True
        )

    def test_postprocess_unknown_for_low_confidence(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method turns low confidence predictions to unknown"""
        monkeypatch.setattr(
            entity_postprocessor_class,
            "ENTITY_CLASS_CONFIDENCE_THRESHOLD",
            0.8,
        )
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.6, outputs=[0.6, 0.4]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                ais_type=9999,
                mmsi="17890234523",
                entity_name="unknown",
                track_length=800,
                file_location=None,
                trackId="B:17890234523",
                flag_code="ARG",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        assert output.entity_class == "unknown"

    def test_postprocess_buoyish_name_overrides_confidence_thresholding(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method

        Test that buoyish names override the confidence thresholding."""
        monkeypatch.setattr(
            entity_postprocessor_class,
            "ENTITY_CLASS_CONFIDENCE_THRESHOLD",
            0.8,
        )
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                ais_type=9999,
                entity_name="buoy",
                track_length=800,
                file_location=None,
                trackId="B:123456789",
                flag_code="USA",
            ),
        )
        entity_outputs_with_details_metadata_tuples_2 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                entity_name="Net-18%",
                ais_type=9999,
                track_length=800,
                file_location=None,
                trackId="B:123456789",
                flag_code="USA",
            ),
        )
        output_1 = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        entity_class_1 = output_1.entity_class
        output_2 = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_2
        )
        entity_class_2 = output_2.entity_class
        assert entity_class_1 == "buoy" and entity_class_2 == "buoy"

    def test_postprocess_not_enough_messages_rule_applied(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.6, outputs=[0.6, 0.4]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="987654321",
                ais_type=9999,
                entity_name="unknown",
                track_length=499,
                file_location=None,
                trackId="A:987654321",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        assert output.entity_class == "unknown"

    def test_buoyish_name_overrides_not_enough_messages(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                ais_type=9999,
                mmsi="987654321",
                entity_name="Net-18%",
                track_length=499,
                file_location=None,
                trackId="A:987654321",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        assert output.entity_class == "buoy"

    def test_buoy_puns_names_are_not_postprocessed(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method."""
        united_states_mmsi = "338148848"
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi=united_states_mmsi,
                ais_type=9999,
                entity_name="Nauti Buoys",
                track_length=800,
                file_location=None,
                trackId="A:338148848",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        entity_class = output.entity_class
        details = output.entity_classification_details
        assert entity_class == "vessel" and details.postprocess_rule_applied is False

    def test_postprocess_known_binned_ship_type_overrides_confidence_thresholding(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method. catches buoyish names"""
        monkeypatch.setattr(
            entity_postprocessor_class,
            "ENTITY_CLASS_CONFIDENCE_THRESHOLD",
            0.8,
        )
        entity_outputs_with_details_metadata_tuples = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.6, outputs=[0.6, 0.4]
            ),
            metadata=EntityMetadata(
                binned_ship_type=2,
                mmsi="123456789",
                entity_name="unknown",
                ais_type=9999,
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples
        )
        assert output.entity_class == "vessel"

    def test_postprocess_rule_application_recorded_in_details(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        monkeypatch.setattr(
            entity_postprocessor_class,
            "ENTITY_CLASS_CONFIDENCE_THRESHOLD",
            0.8,
        )
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                entity_name="buoy",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
                ais_type=9999,
            ),
        )
        entity_outputs_with_details_metadata_tuples_2 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.4, outputs=[0.4, 0.6]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                entity_name="unknown",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
                ais_type=9999,
            ),
        )
        entity_outputs_with_details_metadata_tuples_3 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=8,
                mmsi="123456789",
                entity_name="random",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
                ais_type=9999,
            ),
        )
        output_1 = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        output_2 = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_2
        )
        output_3 = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_3
        )
        assert output_1.entity_classification_details.postprocess_rule_applied
        assert output_2.entity_classification_details.postprocess_rule_applied
        assert output_3.entity_classification_details.postprocess_rule_applied

    def test_postprocess_records_confidence_thresholding_in_details(
        self, entity_postprocessor_class: AtlasEntityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        confidence_threshold = 0.8
        monkeypatch.setattr(
            entity_postprocessor_class,
            "ENTITY_CLASS_CONFIDENCE_THRESHOLD",
            confidence_threshold,
        )
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                ais_type=9999,
                entity_name="random",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        assert (
            output.entity_classification_details.confidence_threshold
            == confidence_threshold
        )

    def test_postprocess_no_postprocessing_applied(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method. catches buoyish names"""
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=0,
                mmsi="123456789",
                ais_type=9999,
                entity_name="random",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
            ),
        )
        output = entity_postprocessor_class.postprocess(
            entity_outputs_with_details_metadata_tuples_1
        )
        assert output.entity_classification_details.postprocess_rule_applied is False
        assert output.entity_class == "vessel"

    def test_postprocess_raises_error_for_known_binned_ship_type_and_buoy_name(
        self, entity_postprocessor_class: AtlasEntityPostProcessor
    ) -> None:
        """Test the postprocess method."""
        entity_outputs_with_details_metadata_tuples_1 = EntityPostprocessorInput(
            predicted_class=AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            entity_classification_details=EntityPostprocessorInputDetails(
                model="test", confidence=0.9, outputs=[0.9, 0.1]
            ),
            metadata=EntityMetadata(
                binned_ship_type=8,
                mmsi="123456789",
                entity_name="buoy",
                track_length=800,
                file_location=None,
                trackId="A:123456789",
                flag_code="USA",
                ais_type=9999,
            ),
        )
        with pytest.raises(KnownShipTypeAndBuoyName):
            entity_postprocessor_class.postprocess(
                entity_outputs_with_details_metadata_tuples_1
            )
