"""Shared fixtures for the atlantes_cpd tests."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_DATA_DIR = Path(os.environ.get("CPD_TEST_DATA_DIR", Path(__file__).parent / "data"))

TRACK_FILE_1 = "B:441667000:1629940564:1213915:464314_03.csv"
TRACK_FILE_2 = "B:477996578:1606957218:2939087:1124020_01.csv"
TRACK_FILE_3 = "B:622113226:1651932693:2100727:1213087_06.csv"


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set a fixed random seed for reproducibility."""
    np.random.seed(42)


def _load_track(file_name: str) -> pd.DataFrame:
    """Load an AIS track CSV, skipping the test if the file is not available."""
    path = TEST_DATA_DIR / file_name
    if not path.exists():
        pytest.skip(f"Test track {path} not found")
    return pd.read_csv(path)


@pytest.fixture(scope="class")
def test_ais_df1() -> pd.DataFrame:
    """AIS track used by the detector integration tests."""
    return _load_track(TRACK_FILE_1)


@pytest.fixture(scope="class")
def test_ais_df2() -> pd.DataFrame:
    """AIS track used by the detector integration tests."""
    return _load_track(TRACK_FILE_2)


@pytest.fixture(scope="class")
def test_ais_df3() -> pd.DataFrame:
    """AIS track used by the detector integration tests."""
    return _load_track(TRACK_FILE_3)
