"""Confirm that the subpath debug endpoint works as expected and we get the right data as expected and right output


# Steps to reproduce
1. Get a subpath ID from es
2. port forward the debug endpoint see pull_track_parquet_from_subpath_id.py
3. start activity service locally
3. run this script on that subpath Id
"""

import json
import os
import time
from pathlib import Path

import click
import pandas as pd
import requests
from atlantes.feedback.pull_track_parquet_from_subpath_id import \
    pull_parquet_from_event
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


ATLAS_ENDPOINT = "http://0.0.0.0:8000/classify"
TIMEOUT_SECONDS = 600

example_dir = Path(__file__).parent.parent


def transform_trackfile_to_activity_request(trackfile: str) -> dict:
    """Transform a trackfile to a request body for the activity endpoint"""
    track_data_json = pd.read_parquet(trackfile).to_json(orient="records")
    return {
        "track": json.loads(track_data_json),  # Ensure proper JSON format
    }


@click.command()
@click.option("--subpath_id", "-s", type=str)
def confirm_subpath_debug_endpoint(subpath_id: str) -> None:
    """Sample request for files stored locally"""
    start = time.time()
    output_path = pull_parquet_from_event(subpath_id, str(example_dir))
    logger.info(f"Output path: {output_path}")
    request = transform_trackfile_to_activity_request(output_path)
    # serialize the request body to a JSON string
    # logger.info(f"{REQUEST_BODY}")
    try:
        response = requests.post(ATLAS_ENDPOINT, json=request, timeout=TIMEOUT_SECONDS)
        if response.ok:
            logger.info(response.json())
            response_dict = response.json()
            output = response_dict["predictions"]
            logger.info(f"Output: {output}")
        else:
            logger.info(f"Request failed with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request failed: {e}")
    finally:
        os.remove(output_path)
        end = time.time()
        logger.info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    confirm_subpath_debug_endpoint()
