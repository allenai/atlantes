"""Unit tests for the atlas_utils module."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import (COLORBLIND_FRIENDLY_COLOR_MAP, FIG_DPI,
                                        FIG_SIZE_IN_INCHES_SUBPATH,
                                        KNOTS_TO_MPS, MAX_VESSEL_SPEED,
                                        UNKNOWN_COG, UNKNOWN_SOG,
                                        AtlasEntityVesselTypeLabelClass,
                                        compute_solar_altitude,
                                        label_seq_by_subpath_activity,
                                        plot_trajectory, preprocess_trackfile)
from pandas._testing import assert_frame_equal


@pytest.fixture(scope="class")
def track_df_with_nan_values(track_df_base_fixture: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with NaN values."""
    #  Add NaN values to the DataFrame
    df = track_df_base_fixture.copy()
    df.loc[2, "sog"] = np.nan
    df.loc[4, "lat"] = np.nan
    df.loc[10, "lon"] = np.nan
    df.loc[12, "cog"] = np.nan
    return df


@pytest.fixture(scope="class")
def track_df_with_unknown_sog_cog(track_df_base_fixture: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with unknown SOG and COG."""
    # Add unknown SOG and COG values to the DataFrame
    df = track_df_base_fixture.copy()
    df.loc[1, "sog"] = UNKNOWN_SOG
    df.loc[3, "cog"] = UNKNOWN_COG
    return df


@pytest.fixture(scope="class")
def track_df_with_bad_lat_lons(track_df_base_fixture: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with bad latitudes and longitudes."""
    # Add bad latitudes and longitudes to the DataFrame
    df = track_df_base_fixture.copy()
    df.loc[5, "lat"] = 0.0
    df.loc[5, "lon"] = 0.0
    df.loc[8, "lat"] = 0.0
    df.loc[8, "lon"] = 0.0
    return df


@pytest.fixture(scope="class")
def track_df_with_too_high_sog(track_df_base_fixture: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with too high SOG."""
    # Add too high SOG values to the DataFrame
    df = track_df_base_fixture.copy()
    df.loc[6, "sog"] = MAX_VESSEL_SPEED + 1
    df.loc[15, "sog"] = MAX_VESSEL_SPEED + 1
    return df


class TestLabelSeqBySubpathActivity:
    def test_label_seq_by_subpath_activity(self) -> None:
        """Test label_seq_by_subpath_activity function."""
        # typical case
        subpath_labels = np.array([0, 1, 2, 0, 1])
        subpath_idxs = np.array([3, 6, 12, 15, 21])
        expected_output = np.array(
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        )
        output = label_seq_by_subpath_activity(subpath_idxs, subpath_labels)
        assert np.array_equal(output, expected_output)

        # test case where there is only one subpath
        subpath_labels = np.array([0])
        subpath_idxs = np.array([2])
        expected_output = np.array([0, 0, 0])
        output = label_seq_by_subpath_activity(subpath_idxs, subpath_labels)
        assert np.array_equal(output, expected_output)


class TestPreprocessTrackfile:
    """Test the preprocess_trackfile function."""

    def test_preprocess_trackfile_empty_df(
        self, track_df_base_fixture: pd.DataFrame
    ) -> None:
        """Test preprocess_trackfile function with empty DataFrame."""
        df = preprocess_trackfile(track_df_base_fixture.iloc[0:0])
        assert df.empty

    def test_preprocess_trackfile_nan_values(
        self,
        track_df_with_nan_values: pd.DataFrame,
        track_df_base_fixture: pd.DataFrame,
    ) -> None:
        """Test preprocess_trackfile function with NaN values."""
        expected_output = track_df_base_fixture.copy()
        expected_output.drop(index=[2, 4, 10, 12], inplace=True)
        expected_output.loc[:, "sog"] = expected_output.sog * KNOTS_TO_MPS

        df = preprocess_trackfile(track_df_with_nan_values)
        assert_frame_equal(df, expected_output)

    def test_preprocess_trackfile_unknown_sog_cog(
        self,
        track_df_with_unknown_sog_cog: pd.DataFrame,
        track_df_base_fixture: pd.DataFrame,
    ) -> None:
        """Test preprocess_trackfile function with unknown SOG and COG."""
        expected_output = track_df_base_fixture.copy()
        expected_output.drop(index=[1, 3], inplace=True)
        expected_output.loc[:, "sog"] = expected_output.sog * KNOTS_TO_MPS
        df = preprocess_trackfile(track_df_with_unknown_sog_cog)
        assert_frame_equal(df, expected_output)

    def test_preprocess_trackfile_bad_lat_lons(
        self,
        track_df_with_bad_lat_lons: pd.DataFrame,
        track_df_base_fixture: pd.DataFrame,
    ) -> None:
        """Test preprocess_trackfile function with bad latitudes and longitudes."""
        expected_output = track_df_base_fixture.copy()
        expected_output.drop(index=[5, 8], inplace=True)
        expected_output.loc[:, "sog"] = expected_output.sog * KNOTS_TO_MPS
        df = preprocess_trackfile(track_df_with_bad_lat_lons)
        assert_frame_equal(df, expected_output)

    def test_preprocess_trackfile_too_high_sog(
        self,
        track_df_with_too_high_sog: pd.DataFrame,
        track_df_base_fixture: pd.DataFrame,
    ) -> None:
        """Test preprocess_trackfile function with too high SOG."""
        expected_output = track_df_base_fixture.copy()
        expected_output.drop(index=[6, 15], inplace=True)
        expected_output.loc[:, "sog"] = expected_output.sog * KNOTS_TO_MPS
        df = preprocess_trackfile(track_df_with_too_high_sog)
        assert_frame_equal(df, expected_output)

    def test_preprocess_trackfile_base_fixture(
        self, track_df_base_fixture: pd.DataFrame
    ) -> None:
        """Test preprocess_trackfile function with base fixture."""
        expected_output = track_df_base_fixture.copy()
        expected_output.loc[:, "sog"] = expected_output.sog * KNOTS_TO_MPS

        df = preprocess_trackfile(track_df_base_fixture)
        assert_frame_equal(df, expected_output)


class TestPlotTrajectory:
    def test_dateline_crossing(self) -> None:
        """tests that the plot is"""

        fig, (ax_1, ax_2) = plt.subplots(
            1,
            2,
            figsize=(FIG_SIZE_IN_INCHES_SUBPATH * 2, FIG_SIZE_IN_INCHES_SUBPATH),
            dpi=FIG_DPI,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Longitude values around the dateline
        longitude = np.array([179, -179, 180, -180])
        latitude = np.array([0, 0, 1, 1])
        subpath_idxs = np.array([0, 1, 2, 3])
        labels = np.array([1, 2, 3, 4])
        label_to_class_name_dict = {1: "1", 2: "2", 3: "3", 4: "4"}
        title = "Activity TEST Plot"
        color_type = label_seq_by_subpath_activity(subpath_idxs, labels)

        # Test the plot_trajectory function with two axes
        ax_1 = plot_trajectory(
            ax_1,
            longitude,
            latitude,
            title,
            color_type,
            COLORBLIND_FRIENDLY_COLOR_MAP,
            label_to_class_name_dict=label_to_class_name_dict,
        )
        ax_2 = plot_trajectory(
            ax_2,
            longitude,
            latitude,
            title,
            color_type,
            COLORBLIND_FRIENDLY_COLOR_MAP,
            label_to_class_name_dict=label_to_class_name_dict,
        )

        # Verify that the legend is set up correctly
        assert len(ax_1.get_legend().get_texts()) == len(label_to_class_name_dict)
        assert len(ax_2.get_legend().get_texts()) == len(label_to_class_name_dict)

        # Verify that the center of the plot is at -180
        assert ax_1.projection.proj4_params.get("lon_0") == -180


@pytest.fixture(scope="class")
def sample_lat_lon_timestamps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixture for latitudes, longitudes, and timestamps."""
    latitudes = np.array([40.7128, 34.0522])  # New York, Los Angeles
    longitudes = np.array([-74.0060, -118.2437])
    timestamps = np.array(
        ["2024-09-04T12:00:00", "2024-09-04T18:00:00"], dtype="datetime64[ns]"
    )
    return latitudes, longitudes, timestamps


class TestComputeSolarAltitude:
    """Test the compute_solar_altitude function."""

    def test_solar_altitude_valid(
        self, sample_lat_lon_timestamps: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test solar altitude calculation with valid input."""
        latitudes, longitudes, timestamps = sample_lat_lon_timestamps

        # Call compute_solar_altitude
        altitudes = compute_solar_altitude(latitudes, longitudes, timestamps)

        # Ensure the altitudes are within valid range (-π/2 to π/2 radians)
        assert np.all(altitudes >= -1.5708)
        assert np.all(altitudes <= 1.5708)

    def test_solar_altitude_nighttime(self) -> None:
        """Test solar altitude during nighttime (should be negative)."""
        # Test case for New York
        ny_latitudes = np.array([40.7128])
        ny_longitudes = np.array([-74.0060])
        ny_timestamps = np.array(
            ["2024-09-04T02:00:00"], dtype="datetime64[ns]"
        )  # Nighttime (UTC - 4)

        # Call compute_solar_altitude
        ny_altitudes = compute_solar_altitude(
            ny_latitudes, ny_longitudes, ny_timestamps
        )

        # Ensure the altitude is negative during the night
        assert ny_altitudes[0] < 0

        # Test case for Seattle
        seattle_latitudes = np.array([47.6062])
        seattle_longitudes = np.array([-122.3321])
        seattle_timestamps = np.array(
            ["2024-06-21T19:00:00"], dtype="datetime64[ns]"
        )  # Noon (UTC - 7)

        # Call compute_solar_altitude
        seattle_altitudes = compute_solar_altitude(
            seattle_latitudes, seattle_longitudes, seattle_timestamps
        )

        # Ensure the altitude is positive during the day
        assert seattle_altitudes[0] > 0


@pytest.fixture
def total_classes() -> int:
    """Fixture to provide the total number of vessel types."""
    return len(AtlasEntityVesselTypeLabelClass)


@pytest.fixture
def all_members() -> list[AtlasEntityVesselTypeLabelClass]:
    """Fixture to provide all vessel type members."""
    return list(AtlasEntityVesselTypeLabelClass)
