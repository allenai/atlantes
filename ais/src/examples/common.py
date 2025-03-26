"""Common functionality for Atlas request examples."""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from atlantes.log_utils import get_logger
from pandas._typing import UsecolsArgType

logger = get_logger(__name__)

TIMEOUT_SECONDS = 600


def create_request_body(
    track_data: str, batch_size: int
) -> dict[str, list[dict[str, Any]]]:
    """Create request body with specified batch size.

    Parameters
    ----------
    track_data : str
        JSON string containing track data
    batch_size : int
        Number of tracks to include in the request

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Request body containing track data
    """
    track = json.loads(track_data)
    return {
        "tracks": [
            {"track_id": f"test-{i}", "track_data": track} for i in range(batch_size)
        ],
    }


def process_response(response_data: dict[str, Any], output_filename: Path) -> None:
    """Process and log classification results.

    Parameters
    ----------
    response_data : Dict[str, Any]
        Response data from Atlas API
    output_filename : Path
        Path to save response data
    """
    classifications = [
        prediction["classification"] for prediction in response_data["predictions"]
    ]
    classification_count = len(classifications)
    logger.info(f"Classification {classification_count=}, {classifications=}")

    with open(output_filename, "w") as outfile:
        json.dump(response_data, outfile, indent=4)
        outfile.write("\n")
        logger.info(f"response saved to {output_filename}")


def load_example_track(csv_path: Path, usecols: UsecolsArgType | None = None) -> str:
    """Load example track data from CSV.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    usecols : UsecolsArgType | None
        Columns to load from CSV

    Returns
    -------
    str
        JSON string containing track data
    """
    df = pd.read_csv(csv_path, usecols=usecols)
    return df.to_json(orient="records")


def send_request(
    batch_size: int,
    atlas_endpoint: str,
    example_track_json: str,
    output_filename: Path,
) -> None:
    """Send request to Atlas API.

    Parameters
    ----------
    batch_size : int
        Number of tracks to process in the request
    atlas_endpoint : str
        Atlas API endpoint
    example_track_json : str
        JSON string containing track data
    output_filename : Path
        Path to save response data
    """
    start = time.time()

    try:
        request_body = create_request_body(example_track_json, batch_size)
        response = requests.post(
            f"{atlas_endpoint}/classify",
            json=request_body,
            timeout=TIMEOUT_SECONDS,
        )

        if not response.ok:
            logger.warning(
                f"Request failed with status code {response.status_code}, {response.text}"
            )
            return

        process_response(response.json(), output_filename)

    except requests.exceptions.RequestException as e:
        logger.exception(f"Request failed: {e}")

    end = time.time()
    logger.info(f"Elapsed time: {end - start}")
