"""Testing configuration for AIS package."""

import logging
import os
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from google.cloud import storage
from shapely.geometry import Point

from atlantes.atlas.atlas_utils import (get_atlas_activity_inference_config,
                                        get_atlas_entity_inference_config)

logger = logging.getLogger(__name__)  # not using atlantes.utils logger for tests

# For local development copy the data back into test-data folder
# TEST_PROJECTS_FOLDER_PREFIX = os.environ.get(
#     "TEST_PROJECTS_FOLDER_PREFIX", Path(__file__).parent.parent
# )
TEST_PROJECTS_FOLDER_PREFIX = os.environ["TEST_PROJECTS_FOLDER_PREFIX"]
TEST_PROJECTS_FOLDER = os.path.join(
    TEST_PROJECTS_FOLDER_PREFIX, "test-data", "test-downsampled-annotation-projects"
)
TEST_GOLD_STANDARD_FOLDER = os.path.join(TEST_PROJECTS_FOLDER, "test_goldstandardproj")
GOLD_STANDARD_PROJECTS = ["test_goldstandardproj"]
GCP_TEST_BUCKET = "ais-track-data"
GCP_TEST_PREFIX = f"{TEST_PROJECTS_FOLDER}/completed".replace(f"gs://{GCP_TEST_BUCKET}/", "")
TEST_AIS_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/test-ais-tracks")
OBVIOUS_TRANSITING_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/obvious-transiting-tracks")
OBVIOUS_MOORED_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/obvious-moored-tracks")
OBVIOUS_ANCHORED_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/obvious-anchored-tracks")
NEAR_SHORE_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/near-shore-tracks")
NON_FISHING_NON_UNKNOWN_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/non-fishing-non-unknown-ves-tracks")
OBVIOUS_STATIONARY_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/stationary-vessel-tracks")
TRANSITING_DF_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/transiting-vessel-tracks")
FISHING_DATA_FOLDER = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/fishing-vessel-tracks")
HIGH_TRAFFIC_PORTS = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/high-traffic-port-tracks")
MARINE_INFRA_SUPPLY_VESSELS = os.path.join(TEST_PROJECTS_FOLDER_PREFIX, "test-data/offshore_marine_infra_transiting_df_list")


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set a fixed random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)

@pytest.fixture(scope="session")
def test_ais_data_folder() -> str:
    """Return the test AIS data folder."""
    return TEST_AIS_DATA_FOLDER

@pytest.fixture(scope="session")
def test_projects_folder() -> str:
    """Return the test projects folder."""
    return TEST_PROJECTS_FOLDER


@pytest.fixture(scope="class")
def data_config_inference_fixture() -> dict:
    """Return the data config for inference."""
    return get_atlas_activity_inference_config()["data"]


@pytest.fixture(scope="class")
def atlas_activity_inference_config_fixture() -> dict:
    """Return the atlas activity inference config."""
    return get_atlas_activity_inference_config()


@pytest.fixture(scope="class")
def atlas_entity_inference_config_fixture() -> dict:
    """Return the atlas entity inference config."""
    return get_atlas_entity_inference_config()


@pytest.fixture(scope="class")
def test_ais_df1() -> pd.DataFrame:
    """Create a DataFrame with AIS data."""
    path_to_data = os.path.join(
        TEST_AIS_DATA_FOLDER,
        "B:441667000:1629940564:1213915:464314_03.csv"
    )
    df = pd.read_csv(path_to_data)
    return df


@pytest.fixture(scope="class")
def test_ais_df2() -> pd.DataFrame:
    """Create a DataFrame with AIS data."""
    path_to_data = os.path.join(
        TEST_AIS_DATA_FOLDER,
        "B:477996578:1606957218:2939087:1124020_01.csv"
    )
    df = pd.read_csv(path_to_data)
    return df


@pytest.fixture(scope="class")
def test_ais_df3() -> pd.DataFrame:
    """Create a DataFrame with AIS data."""
    path_to_data = os.path.join(
        TEST_AIS_DATA_FOLDER,
        "B:622113226:1651932693:2100727:1213087_06.csv"
    )
    df = pd.read_csv(path_to_data)
    return df


@pytest.fixture(scope="class")
def test_buoy_df() -> pd.DataFrame:
    """Create a DataFrame with AIS data."""
    path_to_data = os.path.join(
        TEST_AIS_DATA_FOLDER,
        "B:1121214:1662176951:3036884:1285598_09.csv"
    )
    df = pd.read_csv(path_to_data)
    return df


@pytest.fixture(scope="module")
def gcp_test_projects_folder_name() -> Generator:
    yield GCP_TEST_PREFIX


# Scope should potentially be session here
@pytest.fixture(scope="class")
def track_df_base_fixture() -> pd.DataFrame:
    """Create a DataFrame Trackfile."""
    # Create a DataFrame with NaN values
    data = [
        [
            Point([-1716981.5, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:02:18Z",
            -15.423908,
            28.132294,
            0.1,
            122.6,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716984.0, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:05:18Z",
            -15.42393,
            28.132278,
            0.1,
            147.1,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716980.5, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:20:17Z",
            -15.423899,
            28.132301,
            0.3,
            89.8,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716982.375, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:23:16Z",
            -15.423915,
            28.132301,
            0.1,
            73.8,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716983.25, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:29:17Z",
            -15.4239235,
            28.132273,
            0.3,
            139.6,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716978.125, 3265666.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:35:19Z",
            -15.423877,
            28.132313,
            0.1,
            153.8,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716979.75, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:38:17Z",
            -15.423892,
            28.132282,
            0.1,
            181.6,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716987.0, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:44:19Z",
            -15.423957,
            28.132298,
            0.1,
            250.1,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716983.25, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:47:18Z",
            -15.4239235,
            28.132278,
            0.5,
            234.6,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716987.5, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:50:17Z",
            -15.423962,
            28.132303,
            0.1,
            240.5,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716986.0, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:56:17Z",
            -15.423948,
            28.132298,
            0.1,
            203.9,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716987.0, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T00:59:17Z",
            -15.423957,
            28.132284,
            0.4,
            219.4,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716985.125, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:02:18Z",
            -15.423939,
            28.132261,
            0.2,
            204.3,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716986.5, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:05:18Z",
            -15.423953,
            28.132265,
            0.3,
            228.3,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716990.75, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:14:15Z",
            -15.42399,
            28.13229,
            0.2,
            226.7,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716990.75, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:17:18Z",
            -15.42399,
            28.132292,
            0.8,
            240.0,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716985.125, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:20:18Z",
            -15.42394,
            28.132286,
            0.2,
            180.9,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716996.25, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:29:17Z",
            -15.42404,
            28.13229,
            0.1,
            187.8,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716989.125, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:32:18Z",
            -15.423977,
            28.132284,
            0.2,
            174.1,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716991.625, 3265666.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:35:18Z",
            -15.423998,
            28.13231,
            0.3,
            192.4,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716995.875, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:38:19Z",
            -15.424037,
            28.132284,
            0.4,
            229.9,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716976.375, 3265659.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:41:25Z",
            -15.4238615,
            28.132254,
            0.3,
            127.2,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716977.625, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:47:15Z",
            -15.423873,
            28.132263,
            0.3,
            155.3,
            "POL",
            "vessel",
            0.0,
            0,
        ],
        [
            Point([-1716981.375, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:50:19Z",
            -15.423905,
            28.132284,
            0.3,
            141.4,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716986.0, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:53:20Z",
            -15.423948,
            28.132275,
            0.0,
            167.3,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716982.75, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T01:59:18Z",
            -15.423919,
            28.132261,
            0.2,
            200.0,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716985.75, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:02:16Z",
            -15.423946,
            28.132275,
            0.0,
            202.8,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716982.625, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:11:19Z",
            -15.423917,
            28.132296,
            0.0,
            200.5,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716979.75, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:14:18Z",
            -15.423892,
            28.132284,
            0.3,
            201.1,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716978.875, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:17:16Z",
            -15.423883,
            28.132296,
            0.2,
            130.2,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716980.25, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:23:26Z",
            -15.423897,
            28.132292,
            0.2,
            142.9,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716979.0, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:29:19Z",
            -15.423885,
            28.132298,
            0.6,
            175.3,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716974.25, 3265666.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:32:18Z",
            -15.423841,
            28.13231,
            0.2,
            86.3,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716984.125, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:35:15Z",
            -15.423932,
            28.1323,
            0.1,
            114.7,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716969.375, 3265669.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:47:17Z",
            -15.423799,
            28.132341,
            0.1,
            91.2,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716966.625, 3265673.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:50:15Z",
            -15.423774,
            28.13236,
            0.0,
            105.9,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716974.25, 3265666.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:53:27Z",
            -15.423841,
            28.132309,
            0.2,
            144.2,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716980.125, 3265666.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T02:59:18Z",
            -15.423895,
            28.132309,
            0.2,
            149.9,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716982.625, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T03:02:17Z",
            -15.423917,
            28.132298,
            0.3,
            123.0,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716987.375, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T03:05:16Z",
            -15.42396,
            28.1323,
            0.2,
            165.0,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716988.25, 3265664.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T03:11:18Z",
            -15.423968,
            28.132296,
            0.5,
            171.7,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716988.125, 3265661.0]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T03:17:17Z",
            -15.423966,
            28.132282,
            0.1,
            118.0,
            "POL",
            "vessel",
            0.0,
            1,
        ],
        [
            Point([-1716989.75, 3265662.25]),
            261002142,
            9999,
            9999,
            "B:261002142:1661635081:1999222:1289293",
            "Unknown",
            "2022-12-01T03:20:16Z",
            -15.423982,
            28.132284,
            0.2,
            133.6,
            "POL",
            "vessel",
            0.0,
            1,
        ],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "geometry",
            "mmsi",
            "category",
            "nav",
            "trackId",
            "name",
            "send",
            "lon",
            "lat",
            "sog",
            "cog",
            "flag_code",
            "vessel_class",
            "dist2coast",
            "subpath_num",
        ],
    )
    return df


@pytest.fixture(scope="class")
def obvious_moored_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=OBVIOUS_MOORED_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {OBVIOUS_MOORED_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def obvious_anchored_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=OBVIOUS_ANCHORED_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {OBVIOUS_ANCHORED_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def transiting_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=TRANSITING_DF_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {TRANSITING_DF_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def obvious_transit_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=OBVIOUS_TRANSITING_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {OBVIOUS_TRANSITING_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def near_shore_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=NEAR_SHORE_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {NEAR_SHORE_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def stationary_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=OBVIOUS_STATIONARY_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {OBVIOUS_STATIONARY_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def non_fishing_or_unknown_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=NON_FISHING_NON_UNKNOWN_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {NON_FISHING_NON_UNKNOWN_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def fishing_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=FISHING_DATA_FOLDER))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {FISHING_DATA_FOLDER}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def high_traffic_port_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=HIGH_TRAFFIC_PORTS))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {HIGH_TRAFFIC_PORTS}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]


@pytest.fixture(scope="class")
def offshore_supply_vessels_df_list() -> list[pd.DataFrame]:
    """Return a list of track data that is obviously transiting at the end."""
    client = storage.Client()
    bucket = client.bucket(GCP_TEST_BUCKET)
    blobs = list(bucket.list_blobs(prefix=MARINE_INFRA_SUPPLY_VESSELS))
    if len(blobs) == 0:
        raise FileNotFoundError(f"Could not find any files in {MARINE_INFRA_SUPPLY_VESSELS}")
    blobs_paths = sorted([f"gs://{GCP_TEST_BUCKET}/{blob.name}" for blob in blobs])
    return [pd.read_parquet(blob_path) for blob_path in blobs_paths]
