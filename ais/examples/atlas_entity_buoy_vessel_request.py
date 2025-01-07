"""Test example endpoint
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from atlantes.log_utils import get_logger

logger = get_logger(__name__)

PORT = os.getenv("ATLAS_PORT", default=8000)
ATLAS_ENDPOINT = f"http://0.0.0.0:{PORT}/entityclassifybuoyvessel"
TIMEOUT_SECONDS = 600

# Read CSV file into a DataFrame and convert to JSON string
EXAMPLE_TRACK_JSON = pd.read_csv(Path(__file__).parent.parent /
    "test-data/test-ais-tracks/B:441667000:1629940564:1213915:464314_03.csv"
).to_json(orient="records")

OUTPUT_FILENAME = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_response_entity.json"
)


def sample_request() -> None:
    """Sample request for files stored locally"""
    start = time.time()

    REQUEST_BODY = {
        "track": json.loads(EXAMPLE_TRACK_JSON),  # Ensure proper JSON format
    }

    try:
        response = requests.post(
            ATLAS_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS
        )
        if response.ok:
            logger.info(response.json())
            with open(OUTPUT_FILENAME, "w") as outfile:
                json.dump(response.json(), outfile)
        else:
            logger.info(f"Request failed with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request failed: {e}")

    end = time.time()
    logger.info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    sample_request()
