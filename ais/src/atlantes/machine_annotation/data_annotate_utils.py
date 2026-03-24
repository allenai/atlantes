"""Machine annotation utils for AIS track data.
#TODO: ADD README with more insight on the SME based constants
"""

from importlib import resources
from typing import Optional

import pandas as pd
from atlantes.log_utils import get_logger
from atlantes.utils import get_nav_status
from google.cloud import storage
from tqdm import tqdm

logger = get_logger(__name__)


def list_csv_files_from_bucket(bucket_name: str, directory_path: str) -> list[str]:
    """lists all csv files in a directory in a Google Cloud Storage bucket

    BEWARE: This function will cost money for large buckets copy to disk and read if you are going to do it many times
    Parameters
    ----------
    bucket_name : str
        the name of the Google Cloud Storage bucket
    directory_path : str
        the path to the directory in the Google Cloud Storage bucket

    Returns
    -------
    input_csvs : list[str]
        a list of the paths to the csv files in the Google Cloud Storage bucket

    """
    # Initialize the Google Cloud Storage client
    logger.warning(
        "This function will cost money for large buckets copy to disk and read if you are going to do it many times"
    )
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=directory_path)
    input_csvs = [
        "gs://" + bucket_name + "/" + blob.name
        for blob in tqdm(blobs, desc="Reading Files")
        if blob.name.endswith(".csv")
    ]
    return input_csvs


def write_file_to_bucket(
    output_path: str, bucket_name: str, df: pd.DataFrame, index: bool = True
) -> None:
    """Writes a dataframe to a csv file in a Google Cloud Storage bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(output_path)
    blob.upload_from_string(df.to_csv(index=index), "text/csv")


def get_ais_vessel_category(activity_descs: Optional[list[str]]) -> list[int]:
    """Gets the nav status num for an Activity TYpe"""
    with resources.path("atlantes.config", "AIS_categories.csv") as file:
        df = pd.read_csv(file)
        # Step 1: Filter the DataFrame
        filtered_df = df[df["category_desc"].isin(activity_descs)]
        # Step 2: Extract the 'num' column as a list
        num_list = filtered_df["num"].tolist()

    return num_list


ENGLISH_SPEAKING_MMSIS_CODES = [
    "232",  # United Kingdom and Ireland
    "233",  # United Kingdom
    "234",  # United Kingdom
    "235",  # United Kingdom
    "316",  # Canada
    "319",  # Cayman Islands
    "338",  # U.S.A.
    "366",  # U.S.A.
    "367",  # U.S.A.
    "368",  # U.S.A.
    "369",  # U.S.A.
    "378",  # British Virgin Islands
    "379",  # U.S Virgin Islands
    "503",  # Australia
    "512",  # New Zealand
]

# Common ways of naming a buoy on AIS generalized from anecdotal observations.
#
# These patterns are matched case-insensitively against entity names. They cover:
#   - Net identifiers: "net\d+" (NET10), "net\s+\w+" (NET 1, NET D)
#   - Chinese fishing gear suffixes: "yu\s*\d+-\d+" (MINPINGYU63036-1, ZHE DAI YU 04455-44).
#     Chinese fishing vessels use province+"YU"+registration (e.g. MINPINGYU63036).
#     A -N suffix indicates individual nets/gear, not the vessel itself.
#     Validated against VHS data: -N suffix names are GEAR 35-75% of the time
#     with near-zero FISHING classification. Names may have spaces (ZHE DAI YU 04455)
#     or be concatenated (ZHEDAIYU04455), so we allow optional whitespace between
#     "yu" and the digit block.
#   - Battery/signal indicators: "\d+%" (90%), "\d+V\d+" (8V2 = 8.2V),
#     "\d+\.\d+V" (7.7V, 8.2V). The NVN format is the second most common battery
#     reporting convention in gear names (~18% of GEAR records in VHS). The decimal
#     voltage format (e.g. [7.7V]) adds ~1,355 incremental catches at 45% GEAR
#     and only 1% FISHING.
#   - Double-dash separators: "\d+--\d+" (04001--2, 17002--41).
#     Gear IDs frequently use double-dashes between numeric fields. Validated
#     against VHS: 1,135 names not already caught by other patterns, of which
#     50.5% are GFW GEAR and only 3% FISHING.
#   - Explicit gear names: "fishing gear", "Net fish", "NetFish", "NET MARK"
#   - Numeric gear IDs: "\d{3,5} \d+ \d+" (e.g. "02333 287 82", "1638 48 90",
#     "78000 35 99"). Three-to-five-digit ID followed by space-separated numeric
#     fields (likely channel/signal and battery level without %). Observed in large
#     clusters of stationary AIS-B transponders in the East China Sea.
#
# Patterns investigated but intentionally NOT included:
#   - Standalone voltage (entire name is "6V", "7V"): too short, risk of false
#     positives on garbled AIS data or call signs.
#   - Ellipsis patterns ("..." in names like LRYY56...01): no gear concentration
#     in VHS data, spread across all vessel types. Appears to be garbled AIS data.
#   - Generic dash-number suffix ("\w+-\d+"): too broad without the "yu" prefix,
#     would match legitimate vessel names with hull numbers.
#   - Bracket voltage ("[8.0V]", "[7.1V]"): very rare in VHS data. Could revisit
#     if more examples surface.
NAME_PATTERNS_FOR_BUOYS = [
    r"net\d+",
    r"net\s+\w+",
    r"yu\s*\d+-\d+",
    r"fishing gear",
    r"\d+%",
    r"\d+V\d+",
    r"\d+\.\d+V",
    r"\d+--\d+",
    r"Net fish",
    r"NetFish",
    r"NET MARK",
    r"\d{3,5} \d+ \d+",
]


FISHING_VESSEL_CATEGORIES = get_ais_vessel_category(["Fishing"])

UNKNOWN_VESSEL_CATEGORIES = get_ais_vessel_category(["Unknown"])

KNOWN_NON_FISHING_VESSEL_CATEGORY_DESCS = [
    "Tanker",
    "Passenger",
    "Cargo",
    "Sailing",
    "Tug",
    "Search and Rescue vessel",
    "Towing",
    "Military",
    "Diving",
    "Dredging",
    "Pleasure",
    "Pilot Vessel",
]
KNOWN_NON_FISHING_VESSEL_CATEGORIES = get_ais_vessel_category(
    KNOWN_NON_FISHING_VESSEL_CATEGORY_DESCS
)

# Constants for Subpath Machine Annotation
FISHING_NAV_STATUS = get_nav_status("Fishing")
ANCHORED_NAV_STATUS = get_nav_status("Anchored")
MOORED_NAV_STATUS = get_nav_status("Moored")
UNKNOWN_NAV_STATUS = get_nav_status("Not_defined")
NOT_UNDER_COMMAND_NAV_STATUS = get_nav_status("Not_under_command")

MAX_FISHING_SOG_KNOTS = 10  # SME based, Joe and Henry
MIN_FISHING_SOG_KNOTS = 0.5  # SME based, Joe and Henry
MAX_DRIFTING_SOG_KNOTS = 4  # Based on gulf stream max
# https://oceanservice.noaa.gov/education/tutorial_currents/media/supp_cur04d.html#:~:text=The%20Gulf%20Stream%20is%20a,%2D5.55%20kilometers%20per%20hour).
MIN_TRANSITING_THRESHOLD = 5  # in knots
NEAR_TO_SHORE_DIST = 1  # in km?? ask joe
MAX_MEDIAN_MOORED_SOG = 0.1  # in knots, SME based, Joe and Henry
COG_LINEARITY_THRESHOLD = 5  # in degrees, estimate of maintaining linearity threshold


# Postprocessing constants NOT IN KNOTS
TRANSITING_MIN_HIGH_CONFIDENCE_SOG_METERS_PER_SECOND = (
    4.6  # in m/s, based on SME ~9 knots
)
TRANSITING_MIN_MED_CONFIDENCE_SOG_METERS_PER_SECOND = (
    4.1  # in m/s, based on empirical observation ~8 knots
)
TRANSITING_MAX_MEAN_REL_COG_DEGREES = 4  # in degrees
TRANSITING_MIN_MIDSPEED_SOG_METERS_PER_SECOND = 1.5  # in m/s
TRANSITING_MAX_MIDSPEED_SOG_METERS_PER_SECOND = 4.1  # in m/s
ANCHORED_MAX_SOG_METERS_PER_SECOND = 0.5
MOORED_MAX_SOG_METERS_PER_SECOND = 0.2
MAX_SOG_FOR_ANCHORED_MOORED_METERS_PER_SECOND = 0.4
