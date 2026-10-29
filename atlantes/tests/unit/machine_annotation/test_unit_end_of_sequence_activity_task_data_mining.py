"""Unit tests for the eos_activity_task_data_mining module.


This module contains unit tests for the eos_activity_task_data_mining module."""

import tempfile
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from atlantes.machine_annotation.end_of_sequence_activity_task_data_mining import (
    AISActivityDataMiner, HighSpeedTransitingSearch, MetadataFilter,
    MidSpeedTransitingSearch, NonFishingVesselFilter, Searcher)


class TestFilterStrategies:
    """Test for the filter strategies."""

    def test_non_fishing_vessel_filter_strategy(self) -> None:
        """Test for the NonFishingVesselFilter class."""

        fishing_vessel_ais_type = 30
        unknown_vessel_type_1 = 1
        unknown_vessel_type_2 = 2
        unknown_vessel_type_3 = 3
        unknown_vessel_type_4 = 9999
        metadata_df = pd.DataFrame(
            {
                "ais_type": [
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    51,
                    fishing_vessel_ais_type,
                    53,
                    60,
                    61,
                    72,
                    73,
                    unknown_vessel_type_1,
                    unknown_vessel_type_2,
                    unknown_vessel_type_3,
                    unknown_vessel_type_4,
                ],
                "Path": [
                    "file1",
                    "file2",
                    "file3",
                    "file4",
                    "file5",
                    "file6",
                    "file7",
                    "file8",
                    "file9",
                    "file10",
                    "file11",
                    "file12",
                    "file13",
                    "file14",
                    "file15",
                    "file16",
                    "file17",
                ],
            }
        )
        result = NonFishingVesselFilter.filter(metadata_df)

        assert all("file" in path for path in result)
        assert isinstance(result, list)
        assert "file_8" not in result
        assert "file_14" not in result
        assert "file_15" not in result
        assert "file_16" not in result
        assert "file_17" not in result


class TestSearchStrategies:
    def test_high_speed_transiting_search(self) -> None:
        """Test for the HighSpeedTransitingSearch class."""
        df = pd.DataFrame(
            {
                "send": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "sog": [5, 15],
                "cog": [3, 80],
                "trackId": ["A", "A"],
                "lat": [43.2, 43.4],
                "lon": [72.2, 72.5],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_file_path = temp_file.name
            df.to_csv(temp_file_path, index=False)

            min_trajectory_length = 1
            max_num_messages_to_label = 2
            result = HighSpeedTransitingSearch.search(
                temp_file_path, min_trajectory_length, max_num_messages_to_label
            )

            assert len(result) == 1
            assert result.iloc[0]["activity_type_name"] == "transiting"
            assert result.iloc[0]["Path"] == temp_file_path

    def test_mid_speed_transiting_search(self) -> None:
        """Test for the MidSpeedTransitingSearch class."""
        df = pd.DataFrame(
            {
                "send": pd.to_datetime(["2022-01-01", "2023-01-02", "2023-01-02"]),
                "sog": [4.2, 10, 1.1],
                "cog": [3, 80, 82],
                "trackId": ["A", "A", "A"],
                "lat": [43.2, 43.4, 43.5],
                "lon": [72.2, 72.5, 72.6],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_file_path = temp_file.name
            df.to_csv(temp_file_path, index=False)

            min_trajectory_length = 1
            max_num_messages_to_label = 2
            result = MidSpeedTransitingSearch.search(
                temp_file_path, min_trajectory_length, max_num_messages_to_label
            )

            assert len(result) == 1
            assert result.iloc[0]["activity_type_name"] == "transiting"
            assert result.iloc[0]["Path"] == temp_file_path
            assert result.iloc[0]["send"] == pd.Timestamp("2022-01-01 00:00:00")


def test_metadata_filter_execute_method() -> None:
    """Test for the MetadataFilter class for executing the filter."""
    metadata_df = pd.DataFrame(
        {"ais_type": [30, 40, 50], "Path": ["file1", "file2", "file3"]}
    )

    with patch(
        "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.NonFishingVesselFilter.filter"
    ) as mock_filter:
        mock_filter.return_value = ["file1", "file2"]
        result = MetadataFilter.execute_filter("non_fishing_vessels", metadata_df)

    assert result == ["file1", "file2"]

    with pytest.raises(ValueError):
        MetadataFilter.execute_filter("unknown_strategy", metadata_df)


def test_searcher_execute() -> None:
    """Test for the Searcher class for executing the search."""
    with patch(
        "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.HighSpeedTransitingSearch.search"
    ) as mock_search:
        mock_search.return_value = pd.DataFrame(
            {
                "send": pd.to_datetime(["2023-01-01"]),
                "trackId": ["A"],
                "activity_type_name": ["transiting"],
                "Path": ["test_path"],
            }
        )
        result = Searcher.execute_search("high_speed_transiting", "test_path", 1, 2)

    assert len(result) == 1
    assert result.iloc[0]["activity_type_name"] == "transiting"

    with pytest.raises(ValueError):
        Searcher.execute_search("unknown_strategy", "test_path", 1, 2)


@patch(
    "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.Searcher.execute_search"
)
@patch("atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.dask.compute")
def test_distributed_acquisition(
    mock_dask_compute: Any, mock_execute_search: Any
) -> None:
    """Tests the distributed acquisition method of the AISActivityDataMiner class."""
    mock_execute_search.return_value = pd.DataFrame(
        {
            "send": pd.to_datetime(["2023-01-01"]),
            "trackId": ["A"],
            "activity_type_name": ["transiting"],
            "Path": ["test_path"],
        }
    )
    mock_dask_compute.return_value = [mock_execute_search.return_value]

    result = AISActivityDataMiner.distributed_acquisition(
        ["path1", "path2"], 2, 1, "high_speed_transiting"
    )

    assert isinstance(result, list)
    assert isinstance(result[0], pd.DataFrame)
    assert len(result) == 1
    assert len(result[0]) == 1


def test_synchronous_acquisition() -> None:
    """Tests the synchronous acquisition method of the AISActivityDataMiner class."""
    with patch(
        "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.Searcher.execute_search"
    ) as mock_execute_search:
        mock_execute_search.return_value = pd.DataFrame(
            {
                "send": pd.to_datetime(["2023-01-01"]),
                "trackId": ["A"],
                "activity_type_name": ["transiting"],
                "Path": ["test_path"],
            }
        )

        result = AISActivityDataMiner.synchronous_acquisition(
            ["path1", "path2"], 2, 1, "high_speed_transiting"
        )

    assert len(result) == 2
    assert len(result[0]) == 1
    assert isinstance(result, list)
    assert isinstance(result[0], pd.DataFrame)


@patch(
    "atlantes.human_annotation.prepare_annotations_for_training.pull_previous_months_context"
)
@patch(
    "atlantes.human_annotation.prepare_annotations_for_training.build_raw_paths_context_column"
)
def test_collate_information(
    mock_build_raw_paths: Any, mock_pull_previous_months: Any
) -> None:
    """Test for the collate_information method of the AISActivityDataMiner class."""
    machine_annotated_messages = [
        pd.DataFrame(
            {
                "send": pd.to_datetime(["2023-01-01"]),
                "trackId": ["A"],
                "activity_type_name": ["transiting"],
                "Path": ["test_path"],
            }
        )
    ]
    metadata_df = pd.DataFrame(
        {
            "trackId": ["A"],
            "month": [1],
            "year": [2023],
            "Path": ["test_path"],
            "ais_type": [30],
            "file_name": ["test_path"],
            "flag_code": ["USA"],
        }
    )
    mock_pull_previous_months.return_value = pd.Series(["path1", "path2"])
    mock_build_raw_paths.return_value = "some_raw_path"

    result = AISActivityDataMiner.collate_information(
        machine_annotated_messages, metadata_df
    )

    assert result.iloc[0]["raw_paths"] == ["test_path"] and len(result) == 1


# Test for AISActivityDataMiner mine data
@patch(
    "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.AISActivityDataMiner.collate_information"
)
@patch(
    "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.AISActivityDataMiner.synchronous_acquisition"
)
@patch(
    "atlantes.machine_annotation.end_of_sequence_activity_task_data_mining.MetadataFilter.execute_filter"
)
def test_mine_data_pipeline(
    mock_execute_filter: Any,
    mock_sync_acquisition: Any,
    mock_collate_information: Any,
) -> None:
    """Tests to make sure the mine data method composes the other methods correctly."""
    metadata_df = pd.DataFrame(
        {
            "trackId": ["A"],
            "month": [1],
            "year": [2023],
            "Path": ["test_path"],
            "ais_type": [30],
            "file_name": ["test_path"],
            "flag_code": ["USA"],
        }
    )
    mock_execute_filter.return_value = ["file1", "file2"]
    mock_sync_acquisition.return_value = [
        pd.DataFrame(
            {
                "send": pd.to_datetime(["2023-01-01"]),
                "trackId": ["A"],
                "activity_type_name": ["transiting"],
            }
        )
    ]
    mock_collate_information.return_value = pd.DataFrame(
        {
            "send": pd.to_datetime(["2023-01-01"]),
            "trackId": ["A"],
            "activity_type_name": ["transiting"],
            "raw_paths": [["some_raw_path", "another_raw_path"]],
        }
    )

    result = AISActivityDataMiner.mine_data(
        metadata_df, 1, 2, 2, "non_fishing_vessels", "high_speed_transiting", True
    )

    assert len(result) == 1
    assert result.iloc[0]["raw_paths"] == ["some_raw_path", "another_raw_path"]
    assert result.iloc[0]["activity_type_name"] == "transiting"
