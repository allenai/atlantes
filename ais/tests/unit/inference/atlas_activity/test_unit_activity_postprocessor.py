"""Unit tests for the activity postprocessor class."""

from typing import Any

import pandas as pd
import pytest
from atlantes.atlas.atlas_utils import (AtlasActivityLabelsTraining,
                                        AtlasActivityLabelsWithUnknown)
from atlantes.inference.atlas_activity.postprocessor import \
    AtlasActivityPostProcessor
from atlantes.utils import get_nav_status


class TestAtlasActivityPostProcessor:
    """Unit tests for the AtlasActivityPostProcessor class."""

    @pytest.fixture(scope="class")
    def activity_postprocessor_class(self) -> AtlasActivityPostProcessor:
        """Create an instance of the AtlasActivityPostProcessor class."""
        return AtlasActivityPostProcessor()

    def test_determine_postprocessed_activity_class_med_fast_and_straight_vessel(
        self, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test the determine_postprocessed_activity_class method."""
        most_recent_sog = 5.2  # m / s
        mean_rel_cog = 0.0
        original_activity_class = AtlasActivityLabelsTraining.FISHING
        confidence = 0.9

        postprocessed_activity_class, rule_applied = (
            activity_postprocessor_class.determine_postprocessed_activity_class(
                most_recent_sog, mean_rel_cog, original_activity_class, confidence
            )
        )
        expected_activity_class = AtlasActivityLabelsWithUnknown.TRANSITING
        assert postprocessed_activity_class == expected_activity_class
        assert isinstance(rule_applied, str)

    def test_determine_postprocessed_activity_class_very_fast_vessel(
        self, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test the determine_postprocessed_activity_class method."""
        most_recent_sog = 7.0  # m / s
        mean_rel_cog = 15.0  # degrees
        activity_class = AtlasActivityLabelsTraining.FISHING
        confidence = 0.9
        postprocessed_activity_class, rule_applied = (
            activity_postprocessor_class.determine_postprocessed_activity_class(
                most_recent_sog, mean_rel_cog, activity_class, confidence
            )
        )
        assert postprocessed_activity_class == AtlasActivityLabelsWithUnknown.TRANSITING
        assert isinstance(rule_applied, str)

    def test_determine_postprocessed_activity_class_med_fast_curvy_vessel(
        self, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test the determine_postprocessed_activity_class method for a medium-fast, curvy vessel."""
        most_recent_sog = 3.0  # m/s
        mean_rel_cog = 15.0  # degrees
        activity_class = AtlasActivityLabelsTraining.FISHING
        confidence = 0.9
        all_previous_messages = pd.DataFrame.from_dict(
            {"sog": [3.0, 3.1, 3.2, 3.3, 3.2]}
        )
        postprocessed_activity_class, rule_applied = (
            activity_postprocessor_class.determine_postprocessed_activity_class(
                most_recent_sog,
                mean_rel_cog,
                activity_class,
                confidence,
                dist2coast=2500,
                binned_ship_type=2,
                most_recent_messages=all_previous_messages,
            )
        )
        assert postprocessed_activity_class == AtlasActivityLabelsWithUnknown.FISHING
        assert isinstance(rule_applied, str)
        assert (
            rule_applied
            == activity_postprocessor_class.no_postprocessing_rule_applied._name
        )

    def test_confidence_thresholding(
        self, activity_postprocessor_class: AtlasActivityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        confidence_threshold = 0.8
        monkeypatch.setattr(
            activity_postprocessor_class,
            "ACTIVITY_CLASS_CONFIDENCE_THRESHOLD",
            confidence_threshold,
        )
        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.6},
            {
                "most_recent_data": pd.DataFrame({"sog": [3.0], "rel_cog": [6.0]}),
                "binned_ship_type": 2,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert activity_class == "unknown"

    def test_postprocessing_overides_confidence_thresholding(
        self, activity_postprocessor_class: AtlasActivityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""
        confidence_threshold = 0.8
        monkeypatch.setattr(
            activity_postprocessor_class,
            "ACTIVITY_CLASS_CONFIDENCE_THRESHOLD",
            confidence_threshold,
        )
        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.6},
            {
                "most_recent_data": pd.DataFrame({"sog": [5.0], "rel_cog": [0.0]}),
                "binned_ship_type": 0,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert activity_class == "transiting"

    def test_both_original_and_postprocessed_activity_classes_are_recorded(
        self, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test the postprocess method."""
        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.9},
            {
                "most_recent_data": pd.DataFrame({"sog": [2.0], "rel_cog": [18.0]}),
                "binned_ship_type": 0,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert "original_classification" in details.keys()
        assert "postprocessed_classification" in details.keys()

    def test_is_anchored_but_with_unknown_nav_status(
        self, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test that the vessel is not considered anchored if the nav status is unknown."""
        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.5},
            {
                "most_recent_data": pd.DataFrame(
                    {"sog": [0.0], "nav": [9999], "rel_cog": [0.0]}
                ),  # SOG 0, Unknown nav status, rel_cog 0.0
                "binned_ship_type": 2,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert (
            activity_class == "fishing"
        ), "Vessel should not be considered anchored with unknown nav status."

    @pytest.mark.skip(
        reason="This test is failing due to how inference configuration are set up. "
    )
    def test_is_anchored_with_correct_nav_status(
        self,
        activity_postprocessor_class: AtlasActivityPostProcessor,
    ) -> None:
        """Test that the vessel is considered anchored if SOG is zero and nav status is 'Anchored'."""

        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.9},
            {
                "most_recent_data": pd.DataFrame(
                    {
                        "sog": [0.0],
                        "nav": [get_nav_status("Anchored")],
                        "rel_cog": [0.0],
                    }
                ),  # SOG 0, Anchored nav status, rel_cog 0.0
                "binned_ship_type": 2,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert (
            activity_class == "anchored"
        ), "Vessel should be considered anchored with SOG 0 and anchored nav status."

    def test_is_moored_but_with_unknown_nav_status(
        self,
        activity_postprocessor_class: AtlasActivityPostProcessor,
    ) -> None:
        """Test that the vessel is not considered moored if the nav status is unknown."""

        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.9},
            {
                "most_recent_data": pd.DataFrame(
                    {"sog": [0.0], "nav": [9999], "rel_cog": [0.0]}
                ),  # SOG 0, Unknown nav status, rel_cog 0.0
                "binned_ship_type": 2,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert (
            activity_class == "fishing"
        ), "Vessel should not be considered moored with unknown nav status."

    @pytest.mark.skip(
        reason="This test is failing due to how inference configuration are set up."
    )
    def test_is_moored_with_correct_nav_status(
        self,
        activity_postprocessor_class: AtlasActivityPostProcessor,
    ) -> None:
        """Test that the vessel is considered moored if SOG is zero and nav status is 'Moored'."""

        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.5},
            {
                "most_recent_data": pd.DataFrame(
                    {"sog": [0.0], "nav": [get_nav_status("Moored")], "rel_cog": [0.0]}
                ),  # SOG 0, Moored nav status, rel_cog 0.0
                "binned_ship_type": 0,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert (
            activity_class == "moored"
        ), "Vessel should be considered moored with SOG 0 and moored nav status."

    def test_confidence_thresholding_unknown_vessel(
        self, activity_postprocessor_class: AtlasActivityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""

        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.4},  # in btw fishing (0.3) and unknown (0.5) thresholds
            {
                "most_recent_data": pd.DataFrame({"sog": [3.0], "rel_cog": [6.0]}),
                "binned_ship_type": 0,  # reserved, unknown
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert activity_class == "unknown"

    @pytest.mark.parametrize(
        "ais_type",
        [2],  # these are all known fishing vessel types, see AIS spec
    )
    def test_confidence_thresholding_fishing_vessel_30(
        self, ais_type: int, activity_postprocessor_class: AtlasActivityPostProcessor
    ) -> None:
        """Test the postprocess method."""

        # Prepare input tuple for the postprocess method
        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,  # The label
            {
                "confidence": 0.4
            },  # in between fishing (0.3) and unknown (0.5) thresholds
            {
                "most_recent_data": pd.DataFrame(
                    {"sog": [3.0], "rel_cog": [6.0], "binned_ship_type": ais_type}
                ),
                "binned_ship_type": ais_type,  # Parameterized ship type (fishing vessel types)
            },
        )

        # Call the postprocess method
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )

        # Assert that the classified activity is "fishing"
        assert activity_class == "fishing"

    def test_marine_infra_geofencing(
        self, activity_postprocessor_class: AtlasActivityPostProcessor, monkeypatch: Any
    ) -> None:
        """Test the postprocess method."""

        activity_class_details_metadata_tuples = (
            AtlasActivityLabelsTraining.FISHING,
            {"confidence": 0.9},
            {
                "most_recent_data": pd.DataFrame(
                    {
                        "sog": [5.0],
                        "rel_cog": [6.0],
                        "lat": [-11.07331190843935],
                        "lon": [126.68146729967827],
                    }
                ),
                "binned_ship_type": 2,
            },
        )
        activity_class, details = activity_postprocessor_class.postprocess(
            activity_class_details_metadata_tuples
        )
        assert activity_class == "transiting"
