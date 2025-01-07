"""Integration tests for the 'test_changepoint_detector' module. """

from datetime import datetime
from typing import Generator

import pandas as pd
from atlantes.cpd.changepoint_detector import ChangepointDetector as CPD
from atlantes.cpd.changepoint_detector import ChangepointOutput
from atlantes.cpd.constants import ChangepointReasons
from atlantes.datautils import DATE_FORMAT

# Make changes to them to cover each of the different cases we need to test


# Make edits to turn them into each of the different cases we need to test
def simulate_stream(data: list) -> Generator:
    """Simulate a stream of data."""
    for i in range(len(data)):
        yield data[:i]


class TestChangepointDetector:
    """Tests for the `ChangepointDetector` class."""

    def test_trackfile1(self, test_ais_df1: pd.DataFrame) -> None:
        """Test that the detector producces the appropriate changepoints for a given trackfile."""
        # Arrange
        sogs = test_ais_df1.sog.tolist()
        times = pd.to_datetime(test_ais_df1.send, format=DATE_FORMAT).tolist()
        sogs_stream = simulate_stream(sogs)
        times_stream = simulate_stream(times)
        # Act
        changepoint_outputs = []
        last_changepoint_idx = 0
        for i, (sogs, times) in enumerate(zip(sogs_stream, times_stream)):
            # Set inputs to start from the last changepoint
            sogs = sogs[last_changepoint_idx:]
            times = times[last_changepoint_idx:]
            changepoint_output: ChangepointOutput = CPD.detect_changepoint(sogs, times)
            if changepoint_output.is_changepoint:
                last_changepoint_idx = i
            changepoint_outputs.append(changepoint_output)
        some_expected_changepoint_idxs = [
            18,
            46,
            70,
            109,
            137,
            143,
            172,
            190,
            349,
            359,
            385,
            403,
            411,
            434,
            830,
            1497,
            2224,
            2960,
            3664,
            4351,
            5039,
            5781,
            7152,
            9839,
            10994,
            11021,
            11048,
            11063,
            11076,
            11156,
            11184,
            11587,
            11649,
            11702,
            11713,
            11819,
            12440,
            12383,
            14082,
            15179,
            16038,
            16326,
        ]
        max_duration_exceeded = [11587]
        max_num_messages_exceeded = [12383, 14082, 16038]
        time_based_changepoints = [
            830,
            1497,
            2224,
            2960,
            3664,
            4351,
            5039,
            5781,
            7152,
            9839,
            12440,
        ]
        # Assert
        for idx, changepoint_output in enumerate(changepoint_outputs):
            if idx in some_expected_changepoint_idxs:
                assert changepoint_output.is_changepoint, f"index: {idx}"
                # too many changepoints to enumerate
                if idx in time_based_changepoints:
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.TIME
                    ), f"index: {idx}"
                elif idx in max_duration_exceeded:
                    assert (
                        changepoint_output.changepoint_reason
                        == ChangepointReasons.MAX_DURATION_EXCEEDED
                    ), f"index: {idx}"
                elif idx in max_num_messages_exceeded:
                    assert (
                        changepoint_output.changepoint_reason
                        == ChangepointReasons.MAX_NUM_MESSAGES_EXCEEDED
                    ), f"index: {idx}"
                else:
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.SOG
                    ), f"index: {idx}"
            elif changepoint_output.is_changepoint:
                # not checking all changepoints because so many of them
                pass
            else:
                assert not changepoint_output.is_changepoint, f"index: {idx}"
                assert (
                    changepoint_output.changepoint_reason
                    == ChangepointReasons.NO_CHANGEPOINT
                    or changepoint_output.changepoint_reason
                    == ChangepointReasons.NOT_ENOUGH_MESSAGES
                ), f"index: {idx}"

    def test_trackfile2(self, test_ais_df2: pd.DataFrame) -> None:
        """Test that the detector producces the appropriate changepoints for a given trackfile."""
        # Arrange
        sogs = test_ais_df2.sog.tolist()
        times = pd.to_datetime(test_ais_df2.send, format=DATE_FORMAT).tolist()
        sogs_stream = simulate_stream(sogs)
        times_stream = simulate_stream(times)
        all_expected_changepoint_idxs = [
            10,
            40,
            48,
            79,
            85,
            107,
            119,
            125,
            139,
            147,
            155,
            162,
            177,
            245,
            257,
            269,
            292,
            304,
            310,
            327,
            353,
            372,
            408,
            410,
            551,
            558,
            574,
            580,
            590,
            599,
            609,
            622,
            638,
            663,
            673,
            690,
            713,
            739,
            788,
            794,
            802,
            813,
            821,
            1006,
            1013,
            1019,
            1022,
            1034,
        ]
        time_based_changepoints = [147, 327, 410, 551, 802, 1022]

        # Act
        last_changepoint_idx = 0
        changepoint_outputs = []
        for i, (sogs, times) in enumerate(zip(sogs_stream, times_stream)):
            # Set inputs to start from the last changepoint
            sogs = sogs[last_changepoint_idx:]
            times = times[last_changepoint_idx:]
            changepoint_output: ChangepointOutput = CPD.detect_changepoint(sogs, times)
            if changepoint_output.is_changepoint:
                last_changepoint_idx = i
            changepoint_outputs.append(changepoint_output)

        # Assert
        for idx, changepoint_output in enumerate(changepoint_outputs):
            if idx in all_expected_changepoint_idxs:
                assert changepoint_output.is_changepoint
                if idx in time_based_changepoints:
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.TIME
                    ), f"index: {idx}"
                else:
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.SOG
                    ), f"index: {idx}"
            else:
                assert not changepoint_output.is_changepoint, f"index: {idx}"

                assert (
                    changepoint_output.changepoint_reason
                    == ChangepointReasons.NO_CHANGEPOINT
                    or changepoint_output.changepoint_reason
                    == ChangepointReasons.NOT_ENOUGH_MESSAGES
                ), f"index: {idx}"

    def test_trackfile3(self, test_ais_df3: pd.DataFrame) -> None:
        """Test that the detector producces the appropriate changepoints for a given trackfile."""
        # Arrange
        sogs = test_ais_df3.sog.tolist()
        times = pd.to_datetime(test_ais_df3.send, format=DATE_FORMAT).tolist()
        #
        sogs_stream = simulate_stream(sogs)
        times_stream = simulate_stream(times)
        all_expected_changepoint_idxs = [
            17,
            37,
            52,
            60,
            70,
            81,
            95,
            106,
            118,
            144,
            177,
            180,
            182,
            213,
            221,
            229,
            269,
            281,
            295,
            327,
            338,
            350,
            392,
            409,
            423,
            429,
            445,
            447,
            449,
            451,
            453,
            480,
            506,
            530,
            546,
            558,
            587,
            613,
            630,
            661,
            668,
            686,
            700,
            711,
            723,
            741,
            769,
            793,
            813,
            832,
            852,
            893,
            907,
            918,
            943,
            960,
            996,
            1012,
            1027,
            1089,
            1114,
            1133,
            1139,
            1222,
            1230,
            1240,
            1255,
            1280,
            1301,
            1324,
            1337,
            1347,
            1364,
            1373,
            1388,
            1411,
            1425,
        ]
        time_based_changepoints = [17, 180, 182, 213, 445, 447, 449, 451, 453]

        # Act
        changepoint_outputs = []
        last_changepoint_idx = 0
        for i, (sogs, times) in enumerate(zip(sogs_stream, times_stream)):
            # Set inputs to start from the last changepoint
            sogs = sogs[last_changepoint_idx:]
            times = times[last_changepoint_idx:]
            changepoint_output: ChangepointOutput = CPD.detect_changepoint(sogs, times)
            if changepoint_output.is_changepoint:
                last_changepoint_idx = i
            changepoint_outputs.append(changepoint_output)

        # Assert
        for idx, changepoint_output in enumerate(changepoint_outputs):
            if idx in all_expected_changepoint_idxs:
                if idx in time_based_changepoints:
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.TIME
                    ), f"index: {idx}"
                else:
                    # Otherwise, it should be a SOG changepoint for this trackfile
                    assert changepoint_output.is_changepoint, f"index: {idx}"
                    assert (
                        changepoint_output.changepoint_reason == ChangepointReasons.SOG
                    ), f"index: {idx}"
            else:
                assert not changepoint_output.is_changepoint, f"index: {idx}"
                assert (
                    changepoint_output.changepoint_reason
                    == ChangepointReasons.NO_CHANGEPOINT
                    or changepoint_output.changepoint_reason
                    == ChangepointReasons.NOT_ENOUGH_MESSAGES
                ), f"index: {idx}"

    def test_less_than_min_points(self, test_ais_df1: pd.DataFrame) -> None:
        """Test that the detector returns False when the number of points is less than the minimum."""
        test_df = test_ais_df1.iloc[: CPD.MIN_MESSAGES - 1]
        sogs = test_df.sog.tolist()
        times = pd.to_datetime(test_df.send, format=DATE_FORMAT).tolist()
        changepoint_output = CPD.detect_changepoint(sogs, times)
        assert not changepoint_output.is_changepoint
        assert (
            changepoint_output.changepoint_reason
            == ChangepointReasons.NOT_ENOUGH_MESSAGES
        )

    def test_prioritize_time_gap_as_reason(self) -> None:
        """Test that the detector returns True with a time gap reason,
        when the time gap occurs when the duration is greater than the maximum.

        Time gap should be prioritized"""
        # Create examples where there could be a time gap, max duration exceeded and sog changepoint
        # we want to prioritize time gap
        times = [
            datetime(2020, 12, 31, 23, 59),
            datetime(2021, 1, 1, 0, 0),
            datetime(2021, 1, 1, 1, 0),
            datetime(2021, 1, 1, 2, 0),
            datetime(2021, 1, 1, 3, 0),
            datetime(2021, 1, 1, 4, 0),
            datetime(2022, 1, 1, 5, 0),
        ]

        sogs = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 10]

        changepoint_output = CPD.detect_changepoint(sogs, times)
        assert changepoint_output.is_changepoint
        assert changepoint_output.changepoint_reason == ChangepointReasons.TIME
