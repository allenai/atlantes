from concurrent import futures

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from atlantes.cpd.serve import changepoint_pb2
from atlantes.cpd.serve import changepoint_pb2_grpc
from atlantes.cpd.changepoint_detector import ChangepointDetector
from atlantes.cpd.constants import ChangepointOutput
from atlantes.cpd.serve.config import CPD_ADDRESS, CPD_PORT
from atlantes.log_utils import get_logger

_THREAD_POOL_SIZE = 256

logger = get_logger(__name__)


class ChangepointServicer(changepoint_pb2_grpc.ChangepointServiceServicer):

    def IsChangepoint(
            self,
            request: changepoint_pb2.ChangepointInput,  # type: ignore
            context: grpc.ServicerContext
    ) -> changepoint_pb2.ChangepointOutput:  # type: ignore
        """Check if there is a changepoint in the data."""
        cpd_output: ChangepointOutput = ChangepointDetector.detect_changepoint(
            request.sogs,
            [t.ToDatetime() for t in request.times]
        )
        return changepoint_pb2.ChangepointOutput(  # type: ignore
            is_changepoint=cpd_output.is_changepoint,
            changepoint_reason=cpd_output.changepoint_reason.value
        )


def serve() -> None:
    """Starts the server."""
    logger.info("Starting CPD GRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    changepoint_pb2_grpc.add_ChangepointServiceServicer_to_server(
        ChangepointServicer(), server
    )

    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    # Health borrowed from: https://github.com/grpc/grpc/blob/master/examples/python/xds/server.py
    health_servicer = health.HealthServicer(
        experimental_thread_pool=futures.ThreadPoolExecutor(
            max_workers=_THREAD_POOL_SIZE
        ),
    )

    # Add the health servicer to the server.
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port(f"{CPD_ADDRESS}:{CPD_PORT}")
    server.start()

    # Mark the service as healthy once it is started.
    health_servicer.set('ChangepointService', health_pb2.HealthCheckResponse.SERVING)

    logger.info(f"CPD GRPC server started at {CPD_ADDRESS}:{CPD_PORT}.")
    server.wait_for_termination()
    logger.info("CPD GRPC server exiting.")

if __name__ == '__main__':
    serve()
