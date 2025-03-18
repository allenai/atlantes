"""This example does not work yet, please do not use.

"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from atlantes.log_utils import get_logger

logger = get_logger(__name__)

PORT = os.getenv("ATLAS_PORT", default=8000)
ATLAS_ENDPOINT = f"http://0.0.0.0:{PORT}"
TIMEOUT_SECONDS = 600

example_dir = Path(__file__).parent.parent
# Read CSV file into a DataFrame and convert to JSON string
EXAMPLE_TRACK_JSON = (
    pd.read_csv(
        example_dir
        / "test-data/test-ais-tracks/B:441667000:1629940564:1213915:464314_03.csv",
        usecols=[
            "lat",
            "lon",
            "sog",
            "cog",
            "send",
            "nav",
            "mmsi",
            "trackId",
            "dist2coast",
            "name",
            "flag_code",
            "category",
            "subpath_num",  # I should drop that from activity preprocess validate
        ],
    )
    .head(1000)
    .to_json(orient="records")
)

OUTPUT_FILENAME = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
)


def sample_request() -> None:
    """Sample request for files stored locally"""
    start = time.time()

    try:
        batch_size = int(sys.argv[1])
    except Exception:
        logger.warning("defaulting to batch size 1")
        batch_size = 1
    track = json.loads(EXAMPLE_TRACK_JSON)
    REQUEST_BODY = {
        "track_data": [
            {"track_id": f"test-{i}", "track_data": track} for i in range(batch_size)
        ],
    }
    try:
        response = requests.post(
            ATLAS_ENDPOINT + "/classify",
            json=REQUEST_BODY,
            timeout=TIMEOUT_SECONDS,
        )
        if not response.ok:
            logger.warning(
                f"Request failed with status code {response.status_code}, {response.text}"
            )
            return

        response_data = response.json()

        classifications = [
            prediction["classification"] for prediction in response_data["predictions"]
        ]
        classification_count = len(classifications)
        logger.info(f"Classification {classification_count=}, {classifications=}")

        with open(OUTPUT_FILENAME, "w") as outfile:
            json.dump(response_data, outfile)
            logger.info(f"response saved to {OUTPUT_FILENAME}")
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request failed: {e}")

    end = time.time()
    logger.info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    sample_request()
