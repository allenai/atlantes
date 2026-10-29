"""Serve the activity postprocessor as a GRPC server.

This lets us run the activity postprocessing rules (transiting, anchored,
moored, confidence thresholding, geofencing, etc.) as a standalone service,
decoupled from the model inference pipeline. The server reconstructs the
trajectory DataFrame from the request and calls the exact same
``AtlasActivityPostProcessor.postprocess`` entrypoint the in-process pipeline
uses, so every rule behaves identically."""

# mypy: ignore-errors
from concurrent import futures

import grpc
import pandas as pd
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.datautils import NAV_NAN
from atlantes.inference.atlas_activity.grpc_serve import (
    activitypostprocessor_pb2,
    activitypostprocessor_pb2_grpc,
)
from atlantes.inference.atlas_activity.grpc_serve.config import (
    ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS,
    ACTIVITY_POSTPROCESSOR_GRPC_PORT,
)
from atlantes.inference.atlas_activity.postprocessor import (
    DIST_TO_COAST_THRESHOLD_METERS,
    AtlasActivityPostProcessor,
)
from atlantes.log_utils import get_logger
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

MAX_WORKERS = 10

logger = get_logger(__name__)


class ActivityPostprocessorServicer(
    activitypostprocessor_pb2_grpc.ActivityPostprocessorServiceServicer
):
    DEFAULT_MODEL_NAME = "postprocessor"
    DEFAULT_CONFIDENCE = 1.0
    DEFAULT_OUTPUTS: list[float] = []
    DEFAULT_DIST2COAST = DIST_TO_COAST_THRESHOLD_METERS * 10

    def __init__(self) -> None:
        self.postprocessor = AtlasActivityPostProcessor()
        self.proto_enum_mapping = {
            "fishing": activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.FISHING,
            "anchored": activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.ANCHORED,
            "moored": activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.MOORED,
            "transiting": activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.TRANSITING,
            "unknown": activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.UNKNOWN,
        }

    def _build_message_df(
        self, messages: list[activitypostprocessor_pb2.AisMessage]
    ) -> pd.DataFrame:
        """Reconstruct the ``most_recent_data`` DataFrame from the request messages.

        ``nav`` and ``dist2coast`` are only materialized as columns when at least
        one message provides them; otherwise they are omitted so the
        postprocessor's own column defaults apply, matching real trajectory data.
        """
        any_nav = any(m.HasField("nav") for m in messages)
        any_dist2coast = any(m.HasField("dist2coast") for m in messages)
        rows = []
        for m in messages:
            row = {
                "sog": m.sog,
                "rel_cog": m.rel_cog,
                "cog": m.cog,
                "lat": m.lat,
                "lon": m.lon,
                "send": m.send,
            }
            if any_nav:
                row["nav"] = m.nav if m.HasField("nav") else NAV_NAN
            if any_dist2coast:
                row["dist2coast"] = (
                    m.dist2coast if m.HasField("dist2coast") else self.DEFAULT_DIST2COAST
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def Postprocess(
        self,
        request: activitypostprocessor_pb2.ActivityPostprocessorInput,
        context: grpc.ServicerContext,
    ) -> activitypostprocessor_pb2.ActivityPostprocessorOutput:
        """Postprocess a single activity classification using trajectory context."""
        if len(request.metadata.most_recent_data) == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "metadata.most_recent_data must contain at least one message",
            )

        try:
            activity_class = AtlasActivityLabelsTraining(request.predicted_class)
        except ValueError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"predicted_class must be a training label, got {request.predicted_class}",
            )

        if request.HasField("activity_classification_details"):
            details = {
                "model": request.activity_classification_details.model,
                "confidence": request.activity_classification_details.confidence,
                "outputs": list(request.activity_classification_details.outputs),
            }
        else:
            details = {
                "model": self.DEFAULT_MODEL_NAME,
                "confidence": self.DEFAULT_CONFIDENCE,
                "outputs": list(self.DEFAULT_OUTPUTS),
            }

        message_df = self._build_message_df(list(request.metadata.most_recent_data))
        metadata = {
            "most_recent_data": message_df,
            "binned_ship_type": request.metadata.binned_ship_type,
        }

        try:
            postprocessed_class_name, output_details = self.postprocessor.postprocess(
                (activity_class, details, metadata)
            )
        except Exception as e:
            logger.exception("Failed to postprocess activity data")
            raise e

        logger.info(f"Postprocessed activity classification details: {output_details}")
        output_classification_details_proto = (
            activitypostprocessor_pb2.ActivityPostprocessorOutputDetails(
                model=output_details["model"],
                confidence=output_details["confidence"],
                outputs=output_details["outputs"],
                original_classification=output_details["original_classification"],
                postprocessed_classification=output_details["postprocessed_classification"],
                rule_applied=output_details["rule_applied"],
            )
        )
        return activitypostprocessor_pb2.ActivityPostprocessorOutput(
            activity_class=self.proto_enum_mapping[postprocessed_class_name],
            activity_classification_details=output_classification_details_proto,
        )


def serve() -> None:
    """Starts the server."""
    logger.info("Starting activity postprocessor GRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    activitypostprocessor_pb2_grpc.add_ActivityPostprocessorServiceServicer_to_server(
        ActivityPostprocessorServicer(), server
    )

    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    # Health borrowed from: https://github.com/grpc/grpc/blob/master/examples/python/xds/server.py
    health_servicer = health.HealthServicer(
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
    )

    # Add the health servicer to the server.
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port(
        f"{ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS}:{ACTIVITY_POSTPROCESSOR_GRPC_PORT}"
    )
    server.start()
    # Mark the service as healthy once it is started.
    health_servicer.set("ActivityPostprocessorService", health_pb2.HealthCheckResponse.SERVING)
    logger.info("Activity postprocessor GRPC server started.")
    server.wait_for_termination()
    logger.info("Activity postprocessor GRPC server exiting.")


if __name__ == "__main__":
    serve()
