""" Integration tests for the `gen_track_lengths_file` module. """

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from atlantes.atlas.gen_track_lengths_file import \
    write_traj_lengths_distributed
from atlantes.atlas.schemas import TrajectoryLengthsDataModel
from atlantes.log_utils import get_logger
from click.testing import CliRunner
from google.cloud import storage
from pandas._testing import assert_frame_equal
from pandera import DataFrameSchema

logger = get_logger(__name__)


@pytest.fixture(scope="class")
def root_dir_fixture() -> str:
    return "2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885"


@pytest.fixture(scope="class")
def trajectory_lengths_df_schema() -> DataFrameSchema:
    return TrajectoryLengthsDataModel.to_schema()


@pytest.fixture(scope="class")
def expected_track_lengths_df1() -> pd.DataFrame:

    data = [
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_01.csv",
            297,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_02.csv",
            3004,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_03.csv",
            1889,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_04.csv",
            223,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_05.csv",
            105,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_06.csv",
            130,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_07.csv",
            632,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_08.csv",
            516,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_09.csv",
            1854,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_10.csv",
            63,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_11.csv",
            2156,
        ],
        [
            "gs://ais-track-data/2022/all_2022_tracks_w_subpaths/30/ARG/0/B:701000168:1561242759:3021304:1149885/B:701000168:1561242759:3021304:1149885_12.csv",
            5192,
        ],
    ]

    df = pd.DataFrame(data, columns=["Path", "Length"])
    return df


@pytest.mark.skip(reason="Skipping this test as it is not needed for deployed code")
class TestWriteTrajLengthsDistributed:
    """
    This class contains unit tests for the `write_traj_lengths_distributed` function.
    """

    def test_write_traj_lengths_distributed_with_gcs(
        self,
        root_dir_fixture: str,
        expected_track_lengths_df1: pd.DataFrame,
        trajectory_lengths_df_schema: DataFrameSchema,
    ) -> None:
        """
        Test case for writing trajectory lengths using Google Cloud Storage.
        """
        BUCKET_NAME = "ais-track-data"
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "trajectory_lengths_test_local.csv")
            path_to_metadata_index = str(
                Path(__file__).parents[3]
                / "test-data"
                / "test_metadata_index_single_trackId.parquet"
            )
            result = runner.invoke(
                write_traj_lengths_distributed,
                [
                    "--metadata-index",
                    path_to_metadata_index,
                    "--output-path",
                    output_path,
                ],
            )
            logger.info(result.output)
            logger.info(result.exception)
            logger.info(result.exc_info)

            client = storage.Client()
            output_bucket = client.get_bucket(BUCKET_NAME)
            output_blob = output_bucket.blob(output_path)
            expected_output_path = f"gs://{BUCKET_NAME}/{output_path}"
            track_lengths_df = pd.read_csv(expected_output_path)
            validated_track_lengths_df = trajectory_lengths_df_schema.validate(
                track_lengths_df
            )
            assert result.exit_code == 0
            assert_frame_equal(validated_track_lengths_df, expected_track_lengths_df1)
            output_blob.delete()
