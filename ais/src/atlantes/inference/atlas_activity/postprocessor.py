"""Activity postprocessor class

Do we want metrics when we are not deployed? I think yes
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from atlantes.atlas.atlas_utils import (
    AtlasActivityLabelsTraining,
    AtlasActivityLabelsWithUnknown,
    get_atlas_activity_inference_config,
    haversine_distance_radians,
)
from atlantes.datautils import NAV_NAN
from atlantes.inference.atlas_activity.utils import ports
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import (  # TODO possibly move these to a constants file rather than data_annotate_utils
    ANCHORED_MAX_SOG_METERS_PER_SECOND,
    MAX_SOG_FOR_ANCHORED_MOORED_METERS_PER_SECOND,
    MOORED_MAX_SOG_METERS_PER_SECOND,
    TRANSITING_MAX_MEAN_REL_COG_DEGREES,
    TRANSITING_MIN_HIGH_CONFIDENCE_SOG_METERS_PER_SECOND,
    TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND,
)
from atlantes.utils import (
    AIS_CATEGORIES,
    get_nav_status,
    read_geojson_and_convert_coordinates,
)
from prometheus_client import CollectorRegistry, Counter
from shapely.geometry import Point, Polygon

logger = get_logger("atlas_activity_postprocessor")

DIST_TO_COAST_THRESHOLD_METERS = 2000
STATIONARY_THRESHOLD = 0.0  # sog
DIS_THRESHOLD_METERS = 100  # Threshold for cumulative displacement
# Constants for thresholds
STRAIGHT_LINE_MAX_CUMULATIVE_COG_CHANGE_DEGREES = (
    20  # Max cumulative COG deviation to consider 'straight'
)
STRAIGHT_LINE_DISTANCE_THRESHOLD_NM = (
    20  # in NM -- probably can make this more conservative (> 20)
)
COG_VARIANCE_THRESHOLD = 100  # in degrees
N_MESSAGES_FOR_COG_VARIANCE = 200
MARINE_INFRA_THRESHOLD = 600  # meters
MARINE_INFRA_LAT, MARINE_INFRA_LON = read_geojson_and_convert_coordinates()


class AtlasActivityPostProcessor:
    """Class for postprocessing AIS trajectory activity classifications"""

    def __init__(self) -> None:
        """Initialize the confidence threshold for activity postprocessing"""
        self.ACTIVITY_CLASS_CONFIDENCE_THRESHOLD = 0.3
        self.ACTIVITY_CLASS_UNK_CONFIDENCE_THRESHOLD = 0.7  # starting value
        self.ais_categories = AIS_CATEGORIES
        self.unknown_and_fishing_category = (
            self.ais_categories[
                self.ais_categories["category_desc"].isin(["Unknown", "Fishing"])
            ]["category"]
            .unique()
            .tolist()
        )

        self.fishing_categories = (
            self.ais_categories[self.ais_categories["category_desc"] == "Fishing"][
                "category"
            ]
            .unique()
            .tolist()
        )

        self.postprocess_config = get_atlas_activity_inference_config()["postprocessor"]
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize the metrics for the postprocessor

        Always add a counter for each postprocessing rule

        This container may be instantiated with env var: PROMETHEUS_MULTIPROC_DIR.
        If set, this will write metrics to a file in that directory, which gets picked up
        by the prometheus scrape handler in the worker container.
        """

        REGISTRY = CollectorRegistry(auto_describe=True)

        self.no_postprocessing_rule_applied = Counter(
            "activity_post_processed_no_rule_applied",
            "No postprocessing rule applied",
            registry=REGISTRY,
        )
        self.transiting_post_processed_rule_applied = Counter(
            "activity_post_processed_high_mid_speed_transiting",
            "High or mid speed transiting",
            registry=REGISTRY,
        )
        self.confidence_threshold_rule_applied = Counter(
            "activity_post_processed_confidence_threshold",
            "Confidence threshold applied",
            registry=REGISTRY,
        )
        self.num_non_fishing_unknown_vessels_classified_as_fishing = Counter(
            "num_non_fishing_unknown_vessels_classified_as_fishing",
            "Non-fishing or unknown vessels classified as fishing",
            registry=REGISTRY,
        )
        self.still_anchored_based_on_nav_and_sog_rule_applied = Counter(
            "num_anchored_vessels_classifed_as_fishing",
            "Anchored vessels classified as fishing",
            registry=REGISTRY,
        )
        self.still_moored_based_on_nav_and_sog_rule_applied = Counter(
            "num_moored_vessels_classifed_as_fishing",
            "Moored vessels classified as fishing",
            registry=REGISTRY,
        )
        self.is_stationary_rule_applied_nav_unknown = Counter(
            "is_stationary_rule_applied_nav_unknown",
            "Stationary rule applied",
            registry=REGISTRY,
        )
        self.is_near_shore_threshold_applied = Counter(
            "is_near_shore_threshold_applied",
            "Near shore threshold applied",
            registry=REGISTRY,
        )
        self.is_not_fishing_or_unknown_vessel_type = Counter(
            "is_not_fishing_or_unknown_vessel_type",
            "Not fishing or unknown vessel type",
            registry=REGISTRY,
        )
        self.is_stationary_rule_applied_displacement = Counter(
            "is_stationary_rule_applied_displacement",
            "Stationary rule applied",
            registry=REGISTRY,
        )
        self.is_traveling_straight_rule_applied = Counter(
            "is_traveling_straight_rule_applied",
            "Traveling straight rule applied",
            registry=REGISTRY,
        )
        self.is_collision_avoidance_rule = Counter(
            "is_collision_avoidance_rule_applied",
            "Collision avoidance rule applied",
            registry=REGISTRY,
        )
        self.is_near_high_traffic_port_rule_applied = Counter(
            "is_near_high_traffic_port",
            "Near high traffic port rule applied",
        )
        self.removed_detections_near_infra = Counter(
            "removed_detections_near_infra",
            "Removed detections near marine infrastructure",
            registry=REGISTRY,
        )
        self.is_too_fast_for_anchored_moored_rule_applied = Counter(
            "is_too_fast_for_anchored_moored_rule_applied",
            "Too fast for anchored/moored rule applied",
            registry=REGISTRY,
        )

    def determine_postprocessed_activity_class(
        self,
        most_recent_sog: float,
        mean_rel_cog_past_5_messages: float,
        activity_class: AtlasActivityLabelsTraining,
        confidence: float,
        most_recent_nav_status: int = NAV_NAN,
        dist2coast: float = 0,
        binned_ship_type: int = 0,
        most_recent_messages: pd.DataFrame = None,
    ) -> tuple[AtlasActivityLabelsWithUnknown, str]:
        """Determines the postprocessed activity class based on the input criteria.

        N.B. The ordering of these rules is important and intended to be
        increasing in specificity. Unless you know what you are doing, don't change
        the order.
        """
        # Get the last 5 SOG messages
        if self.postprocess_config.get("apply_transiting_rule", False):
            if self.is_transiting(
                most_recent_sog, mean_rel_cog_past_5_messages, most_recent_messages
            ):
                logger.info("Transiting rule applied.")
                self.transiting_post_processed_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.TRANSITING,
                    self.transiting_post_processed_rule_applied._name,
                )

        if self.postprocess_config.get("apply_anchored_rule", False):
            if self.is_anchored(most_recent_sog, most_recent_nav_status):
                logger.info("Anchored rule applied.")
                self.still_anchored_based_on_nav_and_sog_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.ANCHORED,
                    self.still_anchored_based_on_nav_and_sog_rule_applied._name,
                )

        if self.postprocess_config.get("apply_moored_rule", False):
            if self.is_moored(most_recent_sog, most_recent_nav_status):
                logger.info("Moored rule applied.")
                self.still_anchored_based_on_nav_and_sog_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.MOORED,
                    self.still_anchored_based_on_nav_and_sog_rule_applied._name,
                )

        if self.postprocess_config.get("apply_stationary_rule_displacement", False):
            if self.is_stationary_based_on_displacement(
                most_recent_sog, most_recent_messages
            ):
                logger.info("Stationary displacement rule applied.")
                self.is_stationary_rule_applied_displacement.inc()
                return (
                    AtlasActivityLabelsWithUnknown.MOORED,
                    self.is_stationary_rule_applied_displacement._name,
                )

        if self.postprocess_config.get("apply_stationary_rule", False):
            if self.is_stationary(most_recent_messages, most_recent_nav_status):
                logger.info("Stationary rule applied.")
                self.is_stationary_rule_applied_nav_unknown.inc()
                return (
                    AtlasActivityLabelsWithUnknown.MOORED,
                    self.is_stationary_rule_applied_nav_unknown._name,
                )

        if self.postprocess_config.get("is_traveling_straight", False):
            if self.is_traveling_straight(most_recent_messages):
                logger.info("traveling straight rule applied")
                self.is_traveling_straight_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.TRANSITING,
                    self.is_traveling_straight_rule_applied._name,
                )

        if self.postprocess_config.get("collision_avoidance_rule", False):
            if self.is_collision_avoidance(most_recent_messages):
                logger.info("traveling straight rule applied")
                self.is_collision_avoidance_rule.inc()
                return (
                    AtlasActivityLabelsWithUnknown.TRANSITING,
                    self.is_collision_avoidance_rule._name,
                )

        if self.postprocess_config.get("apply_confidence_threshold_rule", False):
            if not self.is_confident_activity_class(confidence, binned_ship_type):
                logger.info("Confidence thresholding applied.")
                self.confidence_threshold_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.UNKNOWN,
                    self.confidence_threshold_rule_applied._name,
                )

        # todo: revisit this once we have the high resolution distance to coast data
        if self.postprocess_config.get("geofence_high_traffic_ports", False):
            if self.is_within_high_traffic_port(most_recent_messages, ports):
                logger.info("Geofencing high traffic areas.")
                self.is_near_high_traffic_port_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.UNKNOWN,
                    self.is_near_high_traffic_port_rule_applied._name,
                )

        if self.postprocess_config.get("apply_near_shore_rule", False):
            if self.is_near_shore(dist2coast):
                logger.info("Near shore rule applied.")
                self.is_near_shore_threshold_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.UNKNOWN,
                    self.is_near_shore_threshold_applied._name,
                )

        if self.postprocess_config.get("remove_near_marine_infra", False):
            if self.is_near_marine_infra(most_recent_messages):
                logger.info("Removing detections near marine infrastructure.")
                self.removed_detections_near_infra.inc()
                return (
                    AtlasActivityLabelsWithUnknown.TRANSITING,
                    self.removed_detections_near_infra._name,
                )

        # this rule should be last so that if it non-fishing or unknown it will still
        if self.postprocess_config.get("apply_fishing_or_unknown_rule", False):
            if not self.is_fishing_or_unknown_vessel(binned_ship_type):
                logger.info("Fishing/Unknown rule applied.")
                self.num_non_fishing_unknown_vessels_classified_as_fishing.inc()
                return (
                    AtlasActivityLabelsWithUnknown.UNKNOWN,
                    self.num_non_fishing_unknown_vessels_classified_as_fishing._name,
                )

        is_predicted_anchored_or_moored = (
            activity_class == AtlasActivityLabelsTraining.ANCHORED
            or activity_class == AtlasActivityLabelsTraining.MOORED
        )
        if (
            self.postprocess_config.get("is_too_fast_for_anchored_moored", False)
            and is_predicted_anchored_or_moored
        ):
            if self.is_too_fast_for_anchored_moored(most_recent_sog):
                logger.info("Too fast for anchored/moored rule applied.")
                self.is_too_fast_for_anchored_moored_rule_applied.inc()
                return (
                    AtlasActivityLabelsWithUnknown.UNKNOWN,
                    self.is_too_fast_for_anchored_moored_rule_applied._name,
                )

        logger.info("No rule applied, returning original activity class.")
        self.no_postprocessing_rule_applied.inc()
        return (
            AtlasActivityLabelsWithUnknown(activity_class.value),
            self.no_postprocessing_rule_applied._name,
        )

    def is_too_fast_for_anchored_moored(self, most_recent_sog: float) -> bool:
        """Checks if the vessel is too fast for anchored/moored."""
        return most_recent_sog > MAX_SOG_FOR_ANCHORED_MOORED_METERS_PER_SECOND

    def is_collision_avoidance(self, message_df: pd.DataFrame) -> bool:
        """
        Determines if a vsesel is doing collision avoidance
        """
        if message_df is None or len(message_df) < N_MESSAGES_FOR_COG_VARIANCE:
            return False  # Not enough data to perform the check

        reversed_df = message_df.iloc[::-1].reset_index(drop=True)
        reversed_df = reversed_df.head(N_MESSAGES_FOR_COG_VARIANCE)
        cog_variance = reversed_df["cog"].var()

        if cog_variance < COG_VARIANCE_THRESHOLD:
            return True
        else:
            return False

    def is_confident_activity_class(
        self, confidence: float, binned_ship_type: int
    ) -> bool:
        """
        Determines if the activity class confidence meets the threshold based on vessel type.

        todo prefer to declare these in inference config
        """
        logger.info(f"{self.fishing_categories=}")
        logger.info(f"{binned_ship_type=}")
        if self.is_fishing_vessel(binned_ship_type):
            logger.info("Fishing vessel confidence threshold applied.")
            threshold = self.ACTIVITY_CLASS_CONFIDENCE_THRESHOLD
        else:
            threshold = self.ACTIVITY_CLASS_UNK_CONFIDENCE_THRESHOLD
        return confidence >= threshold

    def is_within_high_traffic_port(
        self, message_df: pd.DataFrame, ports: dict
    ) -> bool:
        """
        Checks if the event is within any of the defined port polygons.
        """
        if message_df is None or len(message_df) < 2 or "lat" not in message_df.columns:
            return False

        last_message = message_df.iloc[-1]
        event_point = Point(last_message["lon"], last_message["lat"])

        for port_name, port_polygon_coords in ports.items():
            try:
                port_polygon = Polygon([tuple(coord) for coord in port_polygon_coords])

                if port_polygon.contains(event_point):
                    return True
            except TypeError as e:
                logger.error(f"Error processing polygon for port {port_name}: {e}")

        return False

    def is_near_marine_infra(self, message_df: pd.DataFrame) -> bool:
        """Remove detections near marine infrastructure.

        Parameters
        ----------
        pred : pd.DataFrame
            DataFrame containing the detections.

        Returns
        -------
        pd.DataFrame
            Detections with those near marine infrastructure removed.
        """
        if (
            message_df is None
            or "lat" not in message_df.columns
            or "lon" not in message_df.columns
        ):
            return False
        latest_lat = message_df["lat"].iloc[-1]
        latest_lon = message_df["lon"].iloc[-1]

        distance = haversine_distance_radians(
            latest_lat, latest_lon, MARINE_INFRA_LAT, MARINE_INFRA_LON
        )

        # Handle the case where distance might be an array
        if isinstance(distance, np.ndarray):
            distance = np.nanmin(distance)
            # or use .max() or .mean() depending on your requirements
            logger.info(f"Distance to marine infrastructure: {distance} meters")

        return float(distance) < MARINE_INFRA_THRESHOLD

    def is_stationary_based_on_displacement(
        self,
        sog: float,
        message_df: pd.DataFrame,
        time_window_minutes: int = 10,
        gap_threshold_minutes: int = 10,
    ) -> bool:
        """Checks if the vessel is stationary based on cumulative displacement over the past X minutes."""
        # 1. Drop rows where either 'lat' or 'lon' is NaN

        # 1. Check if 'lat' and 'lon' columns exist in the DataFrame
        if message_df is None or not {"lat", "lon"}.issubset(message_df.columns):
            logger.warning("Latitude or Longitude columns are missing from the data.")
            return False  # Cannot calculate displacement without lat/lon columns

        if sog > TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND:
            return False

        message_df = message_df.dropna(subset=["lat", "lon"])

        message_df["lat"] = pd.to_numeric(message_df["lat"], errors="coerce")
        message_df["lon"] = pd.to_numeric(message_df["lon"], errors="coerce")

        if message_df is None or len(message_df) < 2:
            return False  # Cannot compute displacement with less than 2 messages

        message_df["timestamp"] = pd.to_datetime(message_df["send"])

        latest_time = message_df["timestamp"].iloc[-1]
        second_last_time = message_df["timestamp"].iloc[-2]

        time_threshold = latest_time - timedelta(minutes=time_window_minutes)
        filtered_df = message_df[message_df["timestamp"] >= time_threshold]

        time_gap = latest_time - second_last_time

        # If the time gap is greater than or equal to the gap threshold, include the second-to-last message
        if time_gap >= timedelta(minutes=gap_threshold_minutes):
            # Ensure the second-to-last message is included
            second_last_message = message_df.iloc[-2].to_frame().T
            filtered_df = pd.concat(
                [filtered_df, second_last_message]
            ).drop_duplicates()

        # If there are still less than 2 messages after this, return False
        if len(filtered_df) < 2:
            return False

        # Shift lat and lon columns to create pairs of consecutive positions
        lat1 = filtered_df["lat"].shift(1)
        lon1 = filtered_df["lon"].shift(1)
        lat2 = filtered_df["lat"]
        lon2 = filtered_df["lon"]

        # Drop rows where NaN values were introduced by the shift
        valid_data = ~lat1.isna()
        lat1 = np.array(lat1[valid_data]).astype(float)
        lon1 = np.array(lon1[valid_data]).astype(float)
        lat2 = np.array(lat2[valid_data]).astype(float)
        lon2 = np.array(lon2[valid_data]).astype(float)

        # Apply vectorized haversine_distance
        displacements = haversine_distance_radians(lat1, lon1, lat2, lon2)

        # Calculate the cumulative displacement
        total_displacement = displacements.sum()

        logger.debug(
            f"Cumulative displacement: {total_displacement} meters over {len(filtered_df)} messages"
        )

        # Return whether the displacement is below the threshold
        return total_displacement < DIS_THRESHOLD_METERS

    def is_transiting(
        self, sog: float, mean_rel_cog: float, all_previous_messages: pd.DataFrame
    ) -> bool:
        """Checks if the vessel is transiting based on speed and course over ground."""

        if all_previous_messages is None or len(all_previous_messages) < 5:
            last_5_sog_messages = [sog]
        else:
            last_5_sog_messages = all_previous_messages["sog"].tail(5).tolist()
        sog_buffer = 0.5
        all_sog_in_range = all(
            sog >= TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND - sog_buffer
            for sog in last_5_sog_messages
        )
        medium_fast_and_straight = (
            sog >= TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND
            and mean_rel_cog < TRANSITING_MAX_MEAN_REL_COG_DEGREES
        ) and all_sog_in_range
        very_fast = sog >= TRANSITING_MIN_HIGH_CONFIDENCE_SOG_METERS_PER_SECOND
        logger.debug(f"{medium_fast_and_straight=}")
        logger.debug(f"{very_fast=}")
        logger.debug(f"{last_5_sog_messages=}")
        return medium_fast_and_straight or very_fast

    def is_stationary(
        self, all_previous_messages: pd.DataFrame, nav_status: int
    ) -> bool:
        """Checks if the vessel is anchored based on speed and navigation status."""
        if all_previous_messages is None or len(all_previous_messages) < 5:
            return False
        last_5_sog_messages = all_previous_messages["sog"].tail(5).tolist()

        all_sog_zero = all(sog <= STATIONARY_THRESHOLD for sog in last_5_sog_messages)
        not_defined_nav_statuses = {15, 16, NAV_NAN}  # Not_defined, NAN
        if (
            len(last_5_sog_messages) >= 5
            and all_sog_zero
            and (nav_status in not_defined_nav_statuses)
        ):
            return True
        else:
            return False

    def is_fishing_vessel(self, binned_ship_type: int) -> bool:
        """
        Determines if the vessel is a fishing vessel based on its binned_ship_type code.

        :param binned_ship_type: The AIS ship type code (num).
        :return: True if the vessel is a fishing vessel, False otherwise.
        """
        return binned_ship_type in self.fishing_categories

    def calculate_straight_travel_distance_reverse(
        self, message_df: pd.DataFrame
    ) -> float:
        """
        Calculates how far the vessel has been traveling in a straight line, starting from the last message and moving backwards.

        This function takes the first COG and checks if subsequent COGs are within a specified threshold (degrees) of the first COG.

        :param message_df: Pandas DataFrame with 'lat', 'lon', 'cog' columns.
        :param cog_threshold: The maximum allowed deviation from the first COG (in degrees).
        :return: The distance (in nautical miles) the vessel has been traveling in a straight line.
        """
        if message_df is None or len(message_df) < 2 or "cog" not in message_df.columns:
            return 0.0  # Not enough data to make a determination

        total_distance_traveled_nm = 0.0

        # Reverse the dataframe to start from the last message
        reversed_df = message_df.iloc[::-1].reset_index(drop=True)

        # Get the first COG (starting from the last message)
        first_cog = reversed_df.loc[0]["cog"] % 360  # Normalize to [0, 360)

        # Loop through each consecutive pair of points in reverse
        for i in range(1, len(reversed_df)):
            current_cog = reversed_df.loc[i]["cog"] % 360  # Normalize to [0, 360)

            # Calculate the angular difference between the current COG and the first COG
            cog_difference = abs(current_cog - first_cog)
            if cog_difference > 180:
                cog_difference = 360 - cog_difference  # Handle wrap-around

            # Check if the difference exceeds the threshold
            if cog_difference > STRAIGHT_LINE_MAX_CUMULATIVE_COG_CHANGE_DEGREES:
                logger.info("COG difference exceeds threshold. Stopping accumulation.")
                break

            # Calculate distance between the current and previous points using haversine formula
            current_lat = reversed_df.loc[i]["lat"]
            current_lon = reversed_df.loc[i]["lon"]
            previous_lat = reversed_df.loc[i - 1]["lat"]
            previous_lon = reversed_df.loc[i - 1]["lon"]
            distance_traveled_meters = haversine_distance_radians(
                current_lat, current_lon, previous_lat, previous_lon
            )

            # Convert meters to nautical miles (1 NM = 1852 meters)
            distance_traveled_nm = distance_traveled_meters / 1852

            # Accumulate the distance
            total_distance_traveled_nm += distance_traveled_nm

        return total_distance_traveled_nm

    def is_traveling_straight(self, message_df: pd.DataFrame) -> bool:
        """
        Determines if a vessel has been traveling in a straight line for more than a threshold distance,
        starting from the last message and moving backwards.

        :param dataframe: Pandas dataframe with 'lat', 'lon', 'cog' columns.
        :return: True if the vessel has been traveling straight for more than the threshold distance, otherwise False.
        """
        straight_travel_distance_nm = self.calculate_straight_travel_distance_reverse(
            message_df
        )
        logger.info(f"{straight_travel_distance_nm}")

        # Check if the distance traveled straight exceeds the threshold
        return straight_travel_distance_nm >= STRAIGHT_LINE_DISTANCE_THRESHOLD_NM

    def is_near_shore(self, dist2coast: float) -> bool:
        """Checks if the vessel is near shore based on distance to coast."""
        return dist2coast < DIST_TO_COAST_THRESHOLD_METERS

    def is_fishing_or_unknown_vessel(self, binned_ship_type: int) -> bool:
        """Checks if the vessel is fishing or unknown."""
        return binned_ship_type in self.unknown_and_fishing_category

    def is_anchored(self, sog: float, nav_status: int) -> bool:
        """Checks if the vessel is anchored based on speed and navigation status."""
        return (
            sog < ANCHORED_MAX_SOG_METERS_PER_SECOND
            and nav_status == get_nav_status("Anchored")
        )

    def is_moored(self, sog: float, nav_status: int) -> bool:
        """Checks if the vessel is moored based on speed and navigation status."""
        return sog < MOORED_MAX_SOG_METERS_PER_SECOND and nav_status == get_nav_status(
            "Moored"
        )

    def postprocess(
        self,
        activity_class_details_metadata_tuples: tuple[
            AtlasActivityLabelsTraining, dict, dict
        ],
    ) -> tuple[str, dict]:
        """Postprocess the AIS trajectory data for activity classification
        using the Atlantes system"""
        # Potentially will return Unknown if the confidence is below a certain threshold
        activity_class = activity_class_details_metadata_tuples[0]
        activity_classification_details = activity_class_details_metadata_tuples[1]
        metadata = activity_class_details_metadata_tuples[2]
        most_recent_message_df = metadata["most_recent_data"]
        if len(most_recent_message_df) > 5:
            most_recent_message_df = most_recent_message_df.iloc[-5:]
        mean_rel_cog_past_5_messages = most_recent_message_df["rel_cog"].mean()

        most_recent_sog = most_recent_message_df["sog"].iloc[-1]
        dist2coast = most_recent_message_df.get(
            "dist2coast", pd.Series([DIST_TO_COAST_THRESHOLD_METERS * 10])
        ).iloc[-1]  # 1e9 bc we want to make sure it is not near the coast
        binned_ship_type = metadata["binned_ship_type"]
        most_recent_nav_status = most_recent_message_df.get(
            "nav", pd.Series([NAV_NAN])
        ).iloc[-1]
        # sog is in meters per second 5 meters per second is 9 knots
        postprocessed_activity_class, rule_applied = (
            self.determine_postprocessed_activity_class(
                most_recent_sog,
                mean_rel_cog_past_5_messages,
                activity_class,
                activity_classification_details["confidence"],
                most_recent_nav_status,
                dist2coast,
                binned_ship_type,
                metadata["most_recent_data"],
            )
        )

        postprocessed_activity_class_name = postprocessed_activity_class.name.lower()
        original_activity_class_name = activity_class.name.lower()
        activity_classification_details["original_classification"] = (
            original_activity_class_name
        )
        activity_classification_details["postprocessed_classification"] = (
            postprocessed_activity_class_name
        )
        activity_classification_details["rule_applied"] = rule_applied
        logger.info(
            f"Original Activity Prediction: {original_activity_class_name} -> {postprocessed_activity_class_name}"
        )
        return (
            postprocessed_activity_class_name,
            activity_classification_details,
        )
