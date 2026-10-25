"""Unit tests for `ChangepointDetector.first_changepoint` and the `FirstChangepoint` gRPC method."""

import random
from concurrent import futures
from datetime import datetime, timedelta

import grpc
import pytest
from atlantes.cpd.changepoint_detector import ChangepointDetector
from atlantes.cpd.constants import ChangepointReasons
from atlantes.cpd.serve import changepoint_pb2, changepoint_pb2_grpc
from atlantes.cpd.serve.serve_grpc import ChangepointServicer
from google.protobuf.timestamp_pb2 import Timestamp

START = datetime(2026, 1, 1)


def _feed_one_at_a_time(
    sogs: list[float], times: list[datetime], start_index: int = 1
) -> tuple[int, ChangepointReasons] | None:
    """What a caller does without the batch call: ask about every prefix in turn."""
    for index in range(max(start_index, 1), len(sogs)):
        output = ChangepointDetector.detect_changepoint(sogs[: index + 1], times[: index + 1])
        if output.is_changepoint:
            return index, output.changepoint_reason
    return None


def _trajectory(
    rng: random.Random, length: int, step: timedelta, gap_at: int | None = None, jump_at: int | None = None
) -> tuple[list[float], list[datetime]]:
    sogs: list[float] = []
    times: list[datetime] = []
    now = START
    for i in range(length):
        if gap_at is not None and i == gap_at:
            now += timedelta(hours=3)
        else:
            now += step
        base = 0.2 if jump_at is None or i < jump_at else 12.0
        sogs.append(base + rng.random() * 0.1)
        times.append(now)
    return sogs, times


def _timestamp(t: datetime) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(t)
    return ts


def _request(sogs: list[float], times: list[datetime], start_index: int = 0) -> changepoint_pb2.FirstChangepointInput:  # type: ignore
    return changepoint_pb2.FirstChangepointInput(  # type: ignore
        sogs=sogs, times=[_timestamp(t) for t in times], start_index=start_index
    )


class TestFirstChangepoint:
    """`first_changepoint` must agree with feeding observations to `detect_changepoint` one at a time."""

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_feeding_observations_one_at_a_time(self, seed: int) -> None:
        rng = random.Random(seed)  # nosec B311
        length = rng.choice([2, 3, 5, 6, 20, 120, 520])
        step = timedelta(minutes=rng.choice([1, 10, 30, 90]))
        gap_at = rng.choice([None, length // 2]) if length > 2 else None
        jump_at = rng.choice([None, length // 3]) if length > 8 else None
        sogs, times = _trajectory(rng, length, step, gap_at, jump_at)
        start_index = rng.choice([0, 1, 2, length // 2])

        assert ChangepointDetector.first_changepoint(sogs, times, start_index) == _feed_one_at_a_time(
            sogs, times, start_index
        )

    def test_returns_none_for_a_steady_trajectory(self) -> None:
        sogs, times = _trajectory(random.Random(1), 50, timedelta(minutes=10))  # nosec B311

        assert ChangepointDetector.first_changepoint(sogs, times) is None

    def test_finds_a_time_gap(self) -> None:
        sogs, times = _trajectory(random.Random(2), 30, timedelta(minutes=10), gap_at=12)  # nosec B311

        assert ChangepointDetector.first_changepoint(sogs, times) == (12, ChangepointReasons.TIME)

    def test_finds_a_speed_change(self) -> None:
        sogs, times = _trajectory(random.Random(3), 60, timedelta(minutes=10), jump_at=30)  # nosec B311

        found = ChangepointDetector.first_changepoint(sogs, times)

        assert found is not None
        index, reason = found
        assert reason == ChangepointReasons.SOG
        assert 30 <= index < 60

    def test_finds_the_message_limit(self) -> None:
        sogs = [0.2] * 520
        times = [START + timedelta(seconds=30 * i) for i in range(520)]

        assert ChangepointDetector.first_changepoint(sogs, times) == (
            ChangepointDetector.MAX_NUM_MESSAGES,
            ChangepointReasons.MAX_NUM_MESSAGES_EXCEEDED,
        )

    def test_start_index_skips_earlier_observations(self) -> None:
        sogs, times = _trajectory(random.Random(5), 30, timedelta(minutes=10), gap_at=12)  # nosec B311

        assert ChangepointDetector.first_changepoint(sogs, times, start_index=12) == (12, ChangepointReasons.TIME)
        # Starting after the gap skips it. Whatever is found next must be at or after the start index, and must
        # match asking about each prefix in turn (the gap stays in the buffer, so the speed test can still fire).
        after = ChangepointDetector.first_changepoint(sogs, times, start_index=13)
        assert after == _feed_one_at_a_time(sogs, times, start_index=13)
        assert after is None or after[0] >= 13

    @pytest.mark.parametrize("start_index", [-5, 0, 1])
    def test_start_index_below_one_is_treated_as_one(self, start_index: int) -> None:
        sogs, times = _trajectory(random.Random(6), 10, timedelta(minutes=10), gap_at=1)  # nosec B311

        assert ChangepointDetector.first_changepoint(sogs, times, start_index) == (1, ChangepointReasons.TIME)

    def test_a_single_observation_has_no_changepoint(self) -> None:
        assert ChangepointDetector.first_changepoint([1.0], [START]) is None


class TestFirstChangepointService:
    """The servicer wiring and the generated gRPC code."""

    def test_servicer_reports_no_changepoint_as_minus_one(self) -> None:
        sogs, times = _trajectory(random.Random(7), 30, timedelta(minutes=10))  # nosec B311

        response = ChangepointServicer().FirstChangepoint(_request(sogs, times), None)  # type: ignore[arg-type]

        assert response.changepoint_index == -1
        assert response.changepoint_reason == ChangepointReasons.NO_CHANGEPOINT.value

    def test_servicer_reports_the_first_changepoint(self) -> None:
        sogs, times = _trajectory(random.Random(8), 30, timedelta(minutes=10), gap_at=12)  # nosec B311

        response = ChangepointServicer().FirstChangepoint(_request(sogs, times), None)  # type: ignore[arg-type]

        assert response.changepoint_index == 12
        assert response.changepoint_reason == ChangepointReasons.TIME.value

    def test_round_trip_over_grpc_agrees_with_is_changepoint(self) -> None:
        sogs, times = _trajectory(random.Random(9), 40, timedelta(minutes=10), gap_at=25)  # nosec B311
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        changepoint_pb2_grpc.add_ChangepointServiceServicer_to_server(ChangepointServicer(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = changepoint_pb2_grpc.ChangepointServiceStub(channel)
                first = stub.FirstChangepoint(_request(sogs, times))
                before = stub.IsChangepoint(
                    changepoint_pb2.ChangepointInput(  # type: ignore
                        sogs=sogs[: first.changepoint_index],
                        times=[_timestamp(t) for t in times[: first.changepoint_index]],
                    )
                )
                at = stub.IsChangepoint(
                    changepoint_pb2.ChangepointInput(  # type: ignore
                        sogs=sogs[: first.changepoint_index + 1],
                        times=[_timestamp(t) for t in times[: first.changepoint_index + 1]],
                    )
                )
        finally:
            server.stop(grace=None)

        assert first.changepoint_index == 25
        assert not before.is_changepoint
        assert at.is_changepoint
        assert at.changepoint_reason == first.changepoint_reason

    @pytest.mark.parametrize("sogs_count, times_count", [(5, 4), (4, 5), (3, 0), (0, 3)])
    def test_rejects_sogs_and_times_of_different_lengths(self, sogs_count: int, times_count: int) -> None:
        sogs = [0.2] * sogs_count
        times = [START + timedelta(minutes=10 * i) for i in range(times_count)]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        changepoint_pb2_grpc.add_ChangepointServiceServicer_to_server(ChangepointServicer(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = changepoint_pb2_grpc.ChangepointServiceStub(channel)
                with pytest.raises(grpc.RpcError) as error:
                    stub.FirstChangepoint(_request(sogs, times))
        finally:
            server.stop(grace=None)

        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "same length" in error.value.details()

    def test_empty_request_has_no_changepoint(self) -> None:
        response = ChangepointServicer().FirstChangepoint(_request([], []), None)  # type: ignore[arg-type]

        assert response.changepoint_index == -1

    def test_observations_needed_covers_the_first_possible_changepoint(self) -> None:
        assert ChangepointDetector.observations_needed() == ChangepointDetector.MAX_NUM_MESSAGES + 1
        assert ChangepointDetector.observations_needed(0) == ChangepointDetector.MAX_NUM_MESSAGES + 1
        assert ChangepointDetector.observations_needed(700) == 701

    @pytest.mark.parametrize("start_index, expected_index", [(0, 500), (1, 500), (499, 500), (500, 500), (700, 700)])
    def test_a_long_request_gives_the_same_answer_as_its_leading_observations(
        self, start_index: int, expected_index: int
    ) -> None:
        count = 5000
        sogs = [0.2] * count
        times = [START + timedelta(seconds=30 * i) for i in range(count)]
        needed = ChangepointDetector.observations_needed(start_index)

        full = ChangepointServicer().FirstChangepoint(_request(sogs, times, start_index), None)  # type: ignore[arg-type]
        cut = ChangepointServicer().FirstChangepoint(  # type: ignore[arg-type]
            _request(sogs[:needed], times[:needed], start_index), None
        )

        assert full.changepoint_index == cut.changepoint_index == expected_index
        assert full.changepoint_reason == ChangepointReasons.MAX_NUM_MESSAGES_EXCEEDED.value

