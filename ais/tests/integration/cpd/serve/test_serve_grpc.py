import random
from datetime import datetime, timedelta

import grpc
from atlantes.cpd.constants import ChangepointReasons
from atlantes.cpd.serve import changepoint_pb2, changepoint_pb2_grpc
from atlantes.cpd.serve.config import CPD_ADDRESS, CPD_PORT
from atlantes.log_utils import get_logger
from google.protobuf.timestamp_pb2 import Timestamp

logger = get_logger(__name__)


class TestCpdServeGrpc:

    @staticmethod
    def is_changepoint(
            stub: changepoint_pb2_grpc.ChangepointServiceStub,
            sogs: list[float],
            times: list[datetime]
    ) -> tuple[bool, ChangepointReasons]:
        """Check if there is a changepoint in the data."""

        def to_Timestamp(t: datetime) -> Timestamp:
            _t = Timestamp()
            _t.FromDatetime(t)
            return _t

        inputs = changepoint_pb2.ChangepointInput(  # type: ignore
            sogs=sogs,
            times=[to_Timestamp(t) for t in times]
        )
        response = stub.IsChangepoint(inputs)

        return response.is_changepoint, ChangepointReasons(response.changepoint_reason)

    def test_is_changepoint(self) -> None:
        """Run the client"""
        how_many = 100
        times = [datetime.now() + timedelta(minutes=i*10) for i in range(how_many)]
        sogs = [random.random() for _ in range(how_many)] # nosec B330

        with grpc.insecure_channel(f'{CPD_ADDRESS}:{CPD_PORT}') as channel:
            stub = changepoint_pb2_grpc.ChangepointServiceStub(channel)
            changepoint, reason = self.is_changepoint(stub, sogs, times)

            logger.info(f'Is Changepoint: {changepoint}')
            logger.info(f'Reason: {reason}')
            # TODO: add an assert here it is not a changepoint
