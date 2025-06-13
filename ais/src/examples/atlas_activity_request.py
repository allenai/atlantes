"""Example endpoint for Atlas activity classification."""

import os
from pathlib import Path

import click
from atlantes.log_utils import get_logger

from examples.common import load_example_track, send_request

logger = get_logger(__name__)

PORT = os.getenv("ATLAS_PORT", default=8000)
ATLAS_ENDPOINT = f"http://0.0.0.0:{PORT}"

test_track_path = "noaa_test_track.csv"
logger.info(f"Loading example track from {test_track_path}")
EXAMPLE_TRACK_JSON = load_example_track(
    Path(test_track_path),
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
    ],
)

OUTPUT_FILENAME = Path(__file__).parent / "sample_response_activity.json"


@click.command()
@click.argument("batch_size", type=int, default=1)
def main(batch_size: int) -> None:
    """Send sample request to Atlas API for files stored locally.

    Example:
        python atlas_activity_request.py 5

    Parameters
    ----------
    batch_size : int
        Number of tracks to process in the request
    """
    send_request(
        batch_size=batch_size,
        atlas_endpoint=ATLAS_ENDPOINT,
        example_track_json=EXAMPLE_TRACK_JSON,
        output_filename=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()
