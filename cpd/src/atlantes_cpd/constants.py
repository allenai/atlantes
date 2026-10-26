"""Constants for the CPD module."""

from datetime import timedelta
from enum import Enum
from typing import NamedTuple

MIN_MESSAGES = 5  # Minimum number of messages between changepoints
PROB_THRESHOLD = 0.001  # Probability threshold for changepoint detection
GCP_PREFIX = "gs://"  # Google Cloud Storage prefix
MIN_TIME_GAP = timedelta(
    hours=2
)  # Minimum time to say that there is a time gap between changepoints
BASE_SAMPLE_SIZE_SOG = 5  # Number of messages to use to form distribution for SOG based changepoint detection
MAX_DURATION = timedelta(hours=24)  # Maximum duration of a subpath
MAX_NUM_MESSAGES = 500  # Maximum number of messages in a subpath


class ChangepointReasons(Enum):
    """Messages for Changepoint Detection Output

    0 - SOG: Speed Over Ground out of distribution
    1 - TIME: Time-based Changepoint
    2 - NO_CHANGEPOINT: No Changepoint Detected
    3 - NOT_ENOUGH_MESSAGES: Not Enough Messages for Changepoint Detection
    4 - MAX_DURATION_EXCEEDED: Maximum Duration Exceeded for a subpath
    5 - MAX_NUM_MESSAGES_EXCEEDED: Maximum Number of Messages Exceeded for a subpath
    """

    SOG = 0
    TIME = 1
    NO_CHANGEPOINT = 2
    NOT_ENOUGH_MESSAGES = 3
    MAX_DURATION_EXCEEDED = 4
    MAX_NUM_MESSAGES_EXCEEDED = 5


class ChangepointOutput(NamedTuple):
    """Changepoint Output for Online Changepoint Detection

    Parameters
    ----------
    changepoint : bool
        Whether a changepoint has occurred
    changepoint_reason : ChangepointReasons
        Reason for changepoint
    """

    is_changepoint: bool
    changepoint_reason: ChangepointReasons
