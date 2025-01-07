""" Module for online changepoint detection algorithms
"""

from datetime import datetime, timedelta

import numpy as np
from atlantes.cpd.constants import (
    BASE_SAMPLE_SIZE_SOG,
    MAX_DURATION,
    MAX_NUM_MESSAGES,
    MIN_MESSAGES,
    MIN_TIME_GAP,
    PROB_THRESHOLD,
    ChangepointOutput,
    ChangepointReasons,
)
from numpy.typing import NDArray
from scipy import stats


class TimeGapDetector:
    """
    Time Gap Detector for Online Changepoint Detection

    Parameters
    ----------
    min_time_gap : timedelta
        Minimum time gap between changepoints
    Returns
    -------
    int
    """

    def __init__(self, min_time_gap: timedelta) -> None:
        """Initializes the Time Gap Detector"""
        self.min_time_gap = min_time_gap

    def is_change_point(self, times: list[datetime]) -> bool:
        """Predicts if a changepoint has occurred"""
        if len(times) < 2:
            return False
        previous_datetime = times[-2]
        new_datetime = times[-1]
        time_gap = new_datetime - previous_datetime
        return time_gap >= self.min_time_gap


class SpeedChangeDetector:
    """
    Speed Over Ground based Cumulative Sum Detector for Online Changepoint Detection


    This detecteor is Stateless and past observations are not stored

    Parameters
    ----------
    base_sample_size : int
        Number of messages to use to form distirbution for SOG based changepoint detection
    changepoint_probability_threshold : float
        Probability threshold for changepoint detection
    Returns
    -------
    int or float
    """

    def __init__(
        self, base_sample_size: int, changepoint_probability_threshold: float
    ) -> None:
        "Initializes the Frequentist Cumulative Sum Detector"
        self.changepoint_probability_threshold = changepoint_probability_threshold
        self.base_sample_size = base_sample_size

    def _init_base_speed_distribution(
        self, base_sample: NDArray[np.float32]
    ) -> tuple[float, float]:
        """Initializes the mean and standard deviation of the data"""
        return np.mean(base_sample), np.std(base_sample)

    def _get_prob(self, new_standardized_sum: float) -> float:
        """Gets the probability the new observation is in the base distribution

        AKA the new observation is outside a probability threhsold of the base distribution

        Parameters
        ----------
        new_standardized_sum : float
            The standardized sum of the new observation

        Returns
        -------
        float
            The probability that the new observation is in the base distribution
        """
        prob_less_exreme_than_new_observation = stats.norm.cdf(
            abs(new_standardized_sum)
        )
        prob = 2 * (1 - prob_less_exreme_than_new_observation)
        return prob

    def _get_base_sample(
        self, current_observations: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Gets the base sample from the current observations"""
        return current_observations[: self.base_sample_size]

    def is_change_point(self, sogs: list[float]) -> bool:
        """Checks if a changepoint has occurred and updates the data

        Parameters
        ----------
        sogs : list[float]
            List of speeds over ground
        Returns
        -------
        bool
            True if a changepoint has occurred, False otherwise
        """
        current_num_observations = len(sogs)
        if current_num_observations < self.base_sample_size:
            return False
        current_observations = np.array(sogs)
        base_sample = self._get_base_sample(current_observations)
        # Initializes parameters at the base distirbution sample size
        current_mean, current_std = self._init_base_speed_distribution(base_sample)
        # Calculates the standardized sum of the new observation
        standardized_sum = np.sum(current_observations - current_mean) / (
            current_std * current_num_observations**0.5 + np.finfo(np.float32).eps
        )

        prob = float(self._get_prob(standardized_sum))
        return prob < self.changepoint_probability_threshold


class ChangepointDetector:
    """Changepoint Detector for Online Changepoint Detection

    Variables
    ----------
    BASE_SAMPLE_SIZE_SOG : int
        Number of messages to use to form distirbution for SOG based changepoint detection
    PROB_THRESHOLD : float
        Probability threshold for changepoint detection
    MIN_TIME_GAP : int
        Minimum time gap between changepoints
    MIN_MESSAGES : int
        Minimum number of messages between changepoints
    MAX_DURATION : timedelta
        Maximum duration of a subpath
    MAX_NUM_MESSAGES : int
        Maximum number of messages in a subpath

    """

    BASE_SAMPLE_SIZE_SOG = BASE_SAMPLE_SIZE_SOG
    PROB_THRESHOLD = PROB_THRESHOLD
    MIN_TIME_GAP = MIN_TIME_GAP
    MIN_MESSAGES = MIN_MESSAGES
    MAX_DURATION = MAX_DURATION
    MAX_NUM_MESSAGES = MAX_NUM_MESSAGES

    @classmethod
    def _get_speed_based_changepoint_detector(cls) -> SpeedChangeDetector:
        """Initializes the Speed Over Ground based Cumulative Sum Detector"""
        return SpeedChangeDetector(cls.BASE_SAMPLE_SIZE_SOG, cls.PROB_THRESHOLD)

    @classmethod
    def _get_time_based_changepoint_detector(cls) -> TimeGapDetector:
        """Initializes the Time Gap Detector"""
        return TimeGapDetector(cls.MIN_TIME_GAP)

    @classmethod
    def detect_changepoint(
        cls, sogs: list[float], times: list[datetime]
    ) -> ChangepointOutput:
        """Detects if a changepoint has occurred

        Assumes that the last element in the list is the most recent observation
        1. Checks if there is a time based gap between observations
            - If there is a time based gap, then we check if we have enough messages
            - If we have enough messages, then we return a time based changepoint
            - If not we return changepoint false with a message indicating that we don't have enough messages but found a time gap

            - If there is no time based gap, then we check if we have enough messages
                - If we don't have enough messages, we return a message indicating that we don't have enough messages
                - If we have enough messages, we check for a sog based changepoint
                    - If we find a sog based changepoint, we return a message indicating that we found a sog based changepoint
                    - If we don't find a sog based changepoint, we return a message indicating that we didn't find a changepoint
        2. Checks if Max duration or Max number of messages has been exceeded
        3. Checks if there is a sog based gap between observations


        Based on the sog based and time based changepoint detectors
        detects whether a changepoint has occured and provides a reason for the changepoint

        Parameters
        ----------
        sogs : list[float]
            List of speeds over ground assumes that the sogs are ordered in terms of time
        times : list[str]
            List of times, assumes that the times are ordered from past to present



        Returns
        -------
        ChangepointOutput
            ChangepointOutput object with the changepoint status and message
        """
        num_messages = len(sogs)
        time_based_cpd = cls._get_time_based_changepoint_detector()
        is_less_than_min_messages = num_messages < cls.MIN_MESSAGES
        # Check for time-gapbased change point first
        if time_based_cpd.is_change_point(times):
            return ChangepointOutput(
                is_changepoint=True,
                changepoint_reason=ChangepointReasons.TIME,
            )

        # Check for sog-based change point
        if is_less_than_min_messages:
            return ChangepointOutput(
                is_changepoint=False,
                changepoint_reason=ChangepointReasons.NOT_ENOUGH_MESSAGES,
            )

        if num_messages > cls.MAX_NUM_MESSAGES:
            return ChangepointOutput(
                is_changepoint=True,
                changepoint_reason=ChangepointReasons.MAX_NUM_MESSAGES_EXCEEDED,
            )
        if times[-1] - times[0] > cls.MAX_DURATION:
            return ChangepointOutput(
                is_changepoint=True,
                changepoint_reason=ChangepointReasons.MAX_DURATION_EXCEEDED,
            )

        # Initialize the sog detector only if necessary
        sog_based_cpd = cls._get_speed_based_changepoint_detector()
        if sog_based_cpd.is_change_point(np.array(sogs)):
            return ChangepointOutput(
                is_changepoint=True, changepoint_reason=ChangepointReasons.SOG
            )
        return ChangepointOutput(
            is_changepoint=False,
            changepoint_reason=ChangepointReasons.NO_CHANGEPOINT,
        )
