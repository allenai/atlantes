"""Test the entity postprocessor GRPC server."""

# mypy: ignore-errors
from typing import Any, Optional

import grpc
from atlantes.atlas.atlas_utils import AtlasEntityLabelsTrainingWithUnknown
from atlantes.inference.atlas_entity.grpc_serve import (
    entitypostprocessor_pb2,
    entitypostprocessor_pb2_grpc,
)
from atlantes.inference.atlas_entity.grpc_serve.config import (
    ENTITY_POSTPROCESSOR_GRPC_ADDRESS,
    ENTITY_POSTPROCESSOR_GRPC_PORT,
)
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


class TestEntityPostprocessorServeGrpc:

    @staticmethod
    def postprocess(
        stub: entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub,
        predicted_class: Optional[int],
        entity_classification_details: Optional[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> tuple[
        AtlasEntityLabelsTrainingWithUnknown,
        dict[str, Any],
    ]:
        """Postprocess the entity data through the GRPC server."""
        metadata_proto = entitypostprocessor_pb2.EntityMetadata(
            ais_type=metadata["ais_type"],
            flag_code=metadata["flag_code"],
            entity_name=metadata["entity_name"],
            mmsi=metadata["mmsi"],
            trackId=metadata["trackId"],
        )
        if entity_classification_details:
            entity_classification_details_proto = (
                entitypostprocessor_pb2.EntityPostprocessorInputDetails(
                    model=entity_classification_details["model"],
                    confidence=entity_classification_details["confidence"],
                    outputs=entity_classification_details["outputs"],
                )
            )
        else:
            entity_classification_details_proto = None

        if (
            predicted_class is not None
            and entity_classification_details_proto is not None
        ):
            logger.info("Predicted class and entity classification details passed in")
            inputs = entitypostprocessor_pb2.EntityPostprocessorInput(
                predicted_class=predicted_class,
                metadata=metadata_proto,
                entity_classification_details=entity_classification_details_proto,
            )
        elif predicted_class is not None:
            logger.info("Predicted class passed in, no entity classification details")
            inputs = entitypostprocessor_pb2.EntityPostprocessorInput(
                predicted_class=predicted_class,
                metadata=metadata_proto,
            )
        elif entity_classification_details_proto is not None:
            logger.info("Entity classification details passed in, no predicted class")
            inputs = entitypostprocessor_pb2.EntityPostprocessorInput(
                metadata=metadata_proto,
                entity_classification_details=entity_classification_details_proto,
            )
        else:
            logger.info("No predicted class or entity classification details passed in")
            inputs = entitypostprocessor_pb2.EntityPostprocessorInput(
                metadata=metadata_proto,
            )

        response = stub.Postprocess(inputs)
        if response.entity_class == 2:
            response.entity_class = -100
        logger.info(f"Response: {response.entity_class}")
        return (
            AtlasEntityLabelsTrainingWithUnknown(response.entity_class),
            response.entity_classification_details,
        )

    def test_postprocess_without_details(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = -100
        entity_classification_details = None
        metadata = {
            "flag_code": "USA",
            "ais_type": 9999,  # Unknown THIS IS NOT AIS CATEGORY we cannot convert earlier
            "entity_name": "NET GEAR,78%",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
            "track_length": 10,
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            logger.info(f"Entity class: {entity_class}")
            logger.info(
                f"Entity classification details: {entity_classification_details}"
            )
            assert entity_class == AtlasEntityLabelsTrainingWithUnknown.BUOY

    def test_postprocess_with_details(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = -100
        entity_classification_details = {
            "model": "test_model",
            "confidence": 0.5,
            "outputs": [0.5, 0.5],
        }
        metadata = {
            "flag_code": "USA",
            "ais_type": 70,
            "entity_name": "Vessel Boat",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
            "track_length": 10,
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            logger.info(f"Entity class: {entity_class}")
            logger.info(
                f"Entity classification details: {entity_classification_details}"
            )
            logger.info(
                f"Postprocess rule applied: {entity_classification_details.postprocess_rule_applied}"
            )
            assert entity_class == AtlasEntityLabelsTrainingWithUnknown.VESSEL

    def test_postprocess_without_class(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = None
        entity_classification_details = {
            "model": "test_model",
            "confidence": 0.5,
            "outputs": [0.5, 0.5],
        }
        metadata = {
            "flag_code": "USA",
            "ais_type": 70,
            "entity_name": "Vessel Boat",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
            "track_length": 10,
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            logger.info(f"Entity class: {entity_class}")
            logger.info(
                f"Entity classification details: {entity_classification_details}"
            )
            logger.info(
                f"Postprocess rule applied: {entity_classification_details.postprocess_rule_applied}"
            )
            assert entity_class == AtlasEntityLabelsTrainingWithUnknown.VESSEL

    def test_postprocess_without_class_and_details(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = None
        entity_classification_details = None
        metadata = {
            "flag_code": "USA",
            "ais_type": 70,
            "entity_name": "Vessel Boat",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
            "track_length": 10,
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            assert entity_class == AtlasEntityLabelsTrainingWithUnknown.VESSEL

    def test_postprocess_without_track_length(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = None
        entity_classification_details = None
        metadata = {
            "flag_code": "USA",
            "ais_type": 70,
            "entity_name": "Vessel Boat",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            assert entity_class == AtlasEntityLabelsTrainingWithUnknown.VESSEL

    def test_postprocess_known_ship_type_and_buoy_name(self) -> None:
        """Test that postprocess KnownShipTypeAndBuoyName error returns INVALID_ARGUMENT"""
        predicted_class = AtlasEntityLabelsTrainingWithUnknown.VESSEL.value
        entity_classification_details = None
        metadata = {
            "mmsi": "123456789",
            "entity_name": "buoy",
            "track_length": 800,
            "file_location": None,
            "trackId": "A:123456789",
            "flag_code": "USA",
            "ais_type": 36, # sailing
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            try:
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            except grpc.RpcError as e:
                assert e.code() == grpc.StatusCode.INVALID_ARGUMENT
                assert "The ship type is known and the name indicates it is a buoy" in str(e)


    def test_ensure_output_details_are_present(self) -> None:
        """Test the postprocess function in the GRPC server."""
        predicted_class = None
        entity_classification_details = None
        metadata = {
            "flag_code": "USA",
            "ais_type": 70,
            "entity_name": "Vessel Boat",  # in case entity name is an int
            "trackId": "B:1234567890:57828525252:4252528528",
            "mmsi": "1234567890",
            "track_length": 10,
        }
        with grpc.insecure_channel(
            f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = entitypostprocessor_pb2_grpc.EntityPostprocessorServiceStub(channel)
            entity_class, entity_classification_details = (
                TestEntityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, entity_classification_details, metadata
                )
            )
            logger.info(
                f"Entity classification details: {entity_classification_details}"
            )
            logger.info(
                f"postprocessed_classification: {entity_classification_details.postprocessed_classification}"
            )
            logger.info(
                f"postprocess_rule_applied: {entity_classification_details.postprocess_rule_applied}"
            )
            logger.info(
                f"confidence_threshold: {entity_classification_details.confidence_threshold}"
            )
            logger.info(f"model: {entity_classification_details.model}")
            logger.info(f"confidence: {entity_classification_details.confidence}")
            logger.info(f"outputs: {entity_classification_details.outputs}")
            logger.info(
                f"predicted_classification: {entity_classification_details.predicted_classification}"
            )
            assert entity_classification_details.predicted_classification == "unknown"
            assert entity_classification_details.postprocess_rule_applied is True
            assert entity_classification_details.confidence_threshold == 0.5
            assert entity_classification_details.model == "postprocessor"
            assert entity_classification_details.confidence == 1.0
            assert entity_classification_details.outputs == [0.5, 0.5]
            assert (
                entity_classification_details.postprocessed_classification == "vessel"
            )


# add with and without class passed in test
