"""Serve the entity postprocessor as a GRPC server.

This allows us to do entity classification in obvious cases or when we lack
enough information to run a model."""

# mypy: ignore-errors
from concurrent import futures

import grpc
from atlantes.atlas.atlas_utils import AtlasEntityLabelsTrainingWithUnknown
from atlantes.inference.atlas_entity.datamodels import (
    EntityMetadata,
    EntityPostprocessorInput,
    EntityPostprocessorInputDetails,
)
from atlantes.inference.atlas_entity.grpc_serve import (
    entitypostprocessor_pb2,
    entitypostprocessor_pb2_grpc,
)
from atlantes.inference.atlas_entity.grpc_serve.config import (
    ENTITY_POSTPROCESSOR_GRPC_ADDRESS,
    ENTITY_POSTPROCESSOR_GRPC_PORT,
)
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor,
    KnownShipTypeAndBuoyName,
)
from atlantes.log_utils import get_logger
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

MAX_WORKERS = 10

logger = get_logger(__name__)


class EntityPostprocessorServicer(
    entitypostprocessor_pb2_grpc.EntityPostprocessorServiceServicer
):
    DEFAULT_MODEL_NAME = "postprocessor"
    DEFAULT_CONFIDENCE = 1.0
    DEFAULT_OUTPUTS = [0.5, 0.5]
    DEFAULT_TRACK_LENGTH = 1

    def __init__(self) -> None:
        self.postprocessor = AtlasEntityPostProcessor()
        self.proto_enum_mapping = {
            "vessel": entitypostprocessor_pb2.AtlasEntityLabelsTrainingWithUnknown.VESSEL,
            "buoy": entitypostprocessor_pb2.AtlasEntityLabelsTrainingWithUnknown.BUOY,
            "unknown": entitypostprocessor_pb2.AtlasEntityLabelsTrainingWithUnknown.UNKNOWN,
        }

    def Postprocess(
        self,
        request: entitypostprocessor_pb2.EntityPostprocessorInput,
        context: grpc.ServicerContext,
    ) -> entitypostprocessor_pb2.EntityPostprocessorOutput:
        """Postprocess Entity Data to make a buoy vessel classification without track data."""

        # Parse Optional Fields
        if request.HasField("predicted_class"):
            predicted_class = AtlasEntityLabelsTrainingWithUnknown(
                request.predicted_class
            )
        else:
            predicted_class = AtlasEntityLabelsTrainingWithUnknown.UNKNOWN
        binned_ship_type = (
            request.metadata.binned_ship_type
            if request.metadata.HasField("binned_ship_type")
            else None
        )
        file_location = (
            request.metadata.file_location
            if request.metadata.HasField("file_location")
            else None
        )
        track_length = (
            request.metadata.track_length
            if request.metadata.HasField("track_length")
            else self.DEFAULT_TRACK_LENGTH
        )

        if not request.HasField("entity_classification_details"):
            entity_classification_details = EntityPostprocessorInputDetails(
                model=self.DEFAULT_MODEL_NAME,
                confidence=self.DEFAULT_CONFIDENCE,
                outputs=self.DEFAULT_OUTPUTS,
            )
        else:
            entity_classification_details = EntityPostprocessorInputDetails(
                model=request.entity_classification_details.model,
                confidence=request.entity_classification_details.confidence,
                outputs=request.entity_classification_details.outputs,
            )

        # Prepare Internal Request
        entity_metadata = EntityMetadata(
            binned_ship_type=binned_ship_type,
            ais_type=request.metadata.ais_type,
            flag_code=request.metadata.flag_code,
            entity_name=request.metadata.entity_name,
            track_length=track_length,
            mmsi=request.metadata.mmsi,
            trackId=request.metadata.trackId,
            file_location=file_location,
        )
        internal_request = EntityPostprocessorInput(
            predicted_class=predicted_class,
            entity_classification_details=entity_classification_details,
            metadata=entity_metadata,
        )
        try:
            internal_response = self.postprocessor.postprocess(internal_request)
        except KnownShipTypeAndBuoyName as e:
            logger.warning(f"KnownShipTypeAndBuoyName: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Failed to postprocess entity data")
            raise e
        logger.info(
            f"Internal response: {internal_response.entity_classification_details}"
        )
        output_classification_details_proto = entitypostprocessor_pb2.EntityPostprocessorOutputDetails(
            predicted_classification=internal_response.entity_classification_details.predicted_classification,
            model=internal_response.entity_classification_details.model,
            confidence=internal_response.entity_classification_details.confidence,
            outputs=internal_response.entity_classification_details.outputs,
            postprocessed_classification=internal_response.entity_classification_details.postprocessed_classification,
            postprocess_rule_applied=internal_response.entity_classification_details.postprocess_rule_applied,
            confidence_threshold=internal_response.entity_classification_details.confidence_threshold,
        )
        output_entity_class = self.proto_enum_mapping[
            internal_response.entity_class.lower()
        ]
        return entitypostprocessor_pb2.EntityPostprocessorOutput(
            entity_class=output_entity_class,
            entity_classification_details=output_classification_details_proto,
        )


def serve() -> None:
    """Starts the server."""
    logger.info("Starting entity postprocessor GRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    entitypostprocessor_pb2_grpc.add_EntityPostprocessorServiceServicer_to_server(
        EntityPostprocessorServicer(), server
    )

    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    # Health borrowed from: https://github.com/grpc/grpc/blob/master/examples/python/xds/server.py
    health_servicer = health.HealthServicer(
        experimental_thread_pool=futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ),
    )

    # Add the health servicer to the server.
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port(
        f"{ENTITY_POSTPROCESSOR_GRPC_ADDRESS}:{ENTITY_POSTPROCESSOR_GRPC_PORT}"
    )
    server.start()
    # Mark the service as healthy once it is started.
    health_servicer.set('ChangepointService', health_pb2.HealthCheckResponse.SERVING)
    logger.info("Entity postprocessor GRPC server started.")
    server.wait_for_termination()
    logger.info("Entity postprocessor GRPC server exiting.")


if __name__ == "__main__":
    serve()
