"""Constants for human annotation."""

import os
from datetime import datetime
from enum import Enum

GCP_BUCKET_RAW = "ais-track-data"

GCP_BUCKET_DOWNSAMPLED = "skylight-data-sky-prod-a-dayc"

COMPLETED_ANNOTATIONS_FOLDER = "annotations/completed"

PROJECTS_TO_EXCLUDE = ["playground", "examples"]
GOLD_STANDARD_PROJECTS = [
    "Joe",
    "sampled_trajectories_human_annotate14-12-2023-06-20-42_106",
    "Bradley",
    "Bradley2",
    "sampled_trajectories_human_annotate14-12-2023-06-20-42_101",
    "sampled_trajectories_human_annotate14-12-2023-06-20-42_117",
    "sampled_trajectories_human_annotate14-12-2023-06-20-42_118",
    "sampled_trajectories_human_annotate14-12-2023-06-20-42_102",
    "sampled_trajectories_human_annotate2024-04-25-20-45-28_11",
    "sampled_trajectories_human_annotate2024-04-25-20-45-28_1",
]
ACTIVITY_TYPE_LABELS = [
    "fishing",
    "transiting",
    "anchored",
    "moored",
    "drifting",
    "rendezvous",
    "other",
]

EQUIPMENT_ACTIVITY_TYPE_LABELS = ["being_transported"]
PATH_TO_LOGGING_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "logging.conf"
)


DT_STRING = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

MAP_LABELS_TO_PRIORITY = {
    "fishing": 1,
    "transiting": 2,
    "drifting": 3,
    "rendezvous": 3,
    "anchored": 4,
    "moored": 4,
    "other": 5,
    "unknown": 5,
}
REMOVED_SUBPATH_LABEL = -1


class MonthsToLoad(Enum):
    JAN = "01"
    FEB = "02"
    MAR = "03"
    APR = "04"
    MAY = "05"
    JUN = "06"
    JUL = "07"
    AUG = "08"
    SEP = "09"
    OCT = "10"
    NOV = "11"
    DEC = "12"


ACTIVITY_TYPE_IAA_THRESHOLD = 0.7
ACTIVITY_TYPE_ACCURACY_THRESHOLD = 0.85

# min messages that a track must have in the AOI to be considered "in the AOI" for sampling purposes
MIN_MESSAGES_IN_AOI = 50
VES_NAN = 9999  # NAN value for vessel type so that it has a numeric value


# Downsampling constants
MIN_MESSAGES_TO_USE_FOR_HUMAN_ANNOTATION = 100  # units=num messages in a track
RDP_THRESHOLD = 0.0005  # rdp epsilon value in  Ramer–Douglas–Peucker algorithm

# Creating GeoTiffs constants
# Boundaries do not include above 70N or below 70S because there are limited AIS messages
COORD_REF_SYS = "EPSG:4326"
WIDTH = 15000
WEST = -180
EAST = 180
NORTH = 90
SOUTH = -90
BOUNDARY_COORDS = {"WEST": WEST, "EAST": EAST, "NORTH": NORTH, "SOUTH": SOUTH}
