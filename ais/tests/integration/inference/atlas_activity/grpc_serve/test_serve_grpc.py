"""Test the activity postprocessor GRPC server."""

# mypy: ignore-errors
from typing import Any, Optional

import grpc
import pytest
from atlantes.atlas.atlas_utils import AtlasActivityLabelsWithUnknown
from atlantes.inference.atlas_activity.grpc_serve import (
    activitypostprocessor_pb2,
    activitypostprocessor_pb2_grpc,
)
from atlantes.inference.atlas_activity.grpc_serve.config import (
    ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS,
    ACTIVITY_POSTPROCESSOR_GRPC_PORT,
)
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


class TestActivityPostprocessorServeGrpc:

    @staticmethod
    def postprocess(
        stub: activitypostprocessor_pb2_grpc.ActivityPostprocessorServiceStub,
        predicted_class: int,
        messages: list[dict[str, Any]],
        binned_ship_type: int,
        activity_classification_details: Optional[dict[str, Any]] = None,
    ) -> tuple[AtlasActivityLabelsWithUnknown, Any]:
        """Postprocess the activity data through the GRPC server."""
        ais_messages = [
            activitypostprocessor_pb2.AisMessage(**message) for message in messages
        ]
        metadata_proto = activitypostprocessor_pb2.ActivityMetadata(
            most_recent_data=ais_messages,
            binned_ship_type=binned_ship_type,
            trackId="A:123456789",
        )
        if activity_classification_details is not None:
            details_proto = activitypostprocessor_pb2.ActivityPostprocessorInputDetails(
                model=activity_classification_details["model"],
                confidence=activity_classification_details["confidence"],
                outputs=activity_classification_details["outputs"],
            )
            inputs = activitypostprocessor_pb2.ActivityPostprocessorInput(
                predicted_class=predicted_class,
                metadata=metadata_proto,
                activity_classification_details=details_proto,
            )
        else:
            inputs = activitypostprocessor_pb2.ActivityPostprocessorInput(
                predicted_class=predicted_class,
                metadata=metadata_proto,
            )

        response = stub.Postprocess(inputs)
        logger.info(f"Response activity class: {response.activity_class}")
        return (
            AtlasActivityLabelsWithUnknown(response.activity_class),
            response.activity_classification_details,
        )

    def test_postprocess_transiting(self) -> None:
        """A very fast vessel is reclassified as transiting."""
        predicted_class = (
            activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.FISHING
        )
        messages = [
            {
                "sog": 7.0,
                "rel_cog": 1.0,
                "cog": 90.0,
                "lat": 10.0,
                "lon": 10.0,
                "send": "2024-01-01T00:00:00Z",
            }
        ]
        with grpc.insecure_channel(
            f"{ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS}:{ACTIVITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = activitypostprocessor_pb2_grpc.ActivityPostprocessorServiceStub(
                channel
            )
            activity_class, details = TestActivityPostprocessorServeGrpc.postprocess(
                stub, predicted_class, messages, binned_ship_type=2
            )
            logger.info(f"Activity class: {activity_class}, details: {details}")
            assert activity_class == AtlasActivityLabelsWithUnknown.TRANSITING
            assert details.original_classification == "fishing"
            assert details.postprocessed_classification == "transiting"
            assert details.rule_applied != ""

    def test_postprocess_low_confidence_unknown(self) -> None:
        """A low-confidence fishing classification is downgraded to unknown."""
        predicted_class = (
            activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.FISHING
        )
        # Moderate speed (above the displacement short-circuit, below the
        # transiting thresholds) with a wandering course so no movement rule
        # fires before the confidence-threshold rule.
        messages = [
            {
                "sog": 4.2,
                "rel_cog": 30.0,
                "cog": cog,
                "lat": 10.0 + i * 0.05,
                "lon": 10.0 + i * 0.05,
                "send": f"2024-01-01T00:0{i}:00Z",
            }
            for i, cog in enumerate([10.0, 80.0, 150.0, 220.0, 300.0])
        ]
        details = {"model": "atlas-activity", "confidence": 0.1, "outputs": [0.1, 0.9]}
        with grpc.insecure_channel(
            f"{ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS}:{ACTIVITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = activitypostprocessor_pb2_grpc.ActivityPostprocessorServiceStub(
                channel
            )
            activity_class, out_details = (
                TestActivityPostprocessorServeGrpc.postprocess(
                    stub,
                    predicted_class,
                    messages,
                    binned_ship_type=2,
                    activity_classification_details=details,
                )
            )
            logger.info(f"Activity class: {activity_class}, details: {out_details}")
            assert activity_class == AtlasActivityLabelsWithUnknown.UNKNOWN
            assert out_details.original_classification == "fishing"
            assert out_details.postprocessed_classification == "unknown"
            assert out_details.model == "atlas-activity"
            assert out_details.confidence == pytest.approx(0.1)

    def test_postprocess_empty_messages_invalid_argument(self) -> None:
        """An empty trajectory returns INVALID_ARGUMENT."""
        predicted_class = (
            activitypostprocessor_pb2.AtlasActivityLabelsTrainingWithUnknown.FISHING
        )
        with grpc.insecure_channel(
            f"{ACTIVITY_POSTPROCESSOR_GRPC_ADDRESS}:{ACTIVITY_POSTPROCESSOR_GRPC_PORT}"
        ) as channel:
            stub = activitypostprocessor_pb2_grpc.ActivityPostprocessorServiceStub(
                channel
            )
            try:
                TestActivityPostprocessorServeGrpc.postprocess(
                    stub, predicted_class, [], binned_ship_type=2
                )
                raise AssertionError("Expected INVALID_ARGUMENT")
            except grpc.RpcError as e:
                assert e.code() == grpc.StatusCode.INVALID_ARGUMENT
