"""Test the mainmodule activity endpoint. To test locally run python3 src/main_activity.py and then run pytest tests/test_main_activity.py.
"""

import json

import pandas as pd
import pytest
import ray
import requests
from atlantes.log_utils import get_logger

logger = get_logger(__name__)

ACTIVITY_CLASSIFICATION_ENDPOINT = "http://0.0.0.0:8000/classify"


@pytest.fixture(scope="class")
def json_track_request(test_ais_df1: pd.DataFrame) -> str:
    """Build an in-memory AIS track stream."""
    test_ais_df1_inference = test_ais_df1.copy()
    test_ais_df1_inference = test_ais_df1_inference.drop(
        columns=["subpath_num", "geometry", "vessel_class"]
    )
    logger.info(test_ais_df1_inference.columns)
    json_str = test_ais_df1_inference.to_json(orient="records")
    return json_str


@ray.remote
def send_track(request_body: dict, endpoint: str) -> requests.Response:
    """Send a track to the specified endpoint."""
    classification_response = requests.post(endpoint, json=request_body, timeout=1000)
    return classification_response


class TestActivityFastApiEndpoint:
    def test_activity_endpoint_batching(self, json_track_request: str) -> None:
        """Test the activity endpoint to ensure batching works."""
        REQUEST_BODY = {
            "track": json.loads(json_track_request),  # Ensure proper JSON format
        }
        response_lst = ray.get(
            [
                send_track.remote(REQUEST_BODY, ACTIVITY_CLASSIFICATION_ENDPOINT)
                for i in range(4)
            ]
        )
        assert all([response.status_code == 200 for response in response_lst])
        response_outputs = [response_lst[i].json()["predictions"] for i in range(4)]
        logger.info(response_outputs)
        assert len(response_outputs) == 4
        assert all([len(response_outputs[i]) == 2 for i in range(4)])
        assert all([isinstance(response_outputs[i][0], str) for i in range(4)])
        assert all([isinstance(response_outputs[i][1], dict) for i in range(4)])

    def test_activity_endpoint(self, json_track_request: str) -> None:
        """Test the activity endpoint."""
        REQUEST_BODY = {
            "track": json.loads(json_track_request),  # Ensure proper JSON format
        }

        classification_response = requests.post(
            ACTIVITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        assert classification_response.status_code == 200

        logger.info(f"Response Headers: {classification_response.headers}")

        response_json = classification_response.json()
        logger.info(f"Response JSON: {response_json}")
        track_0_outputs = response_json["predictions"]
        predictions_text = track_0_outputs[0]
        full_predictions = track_0_outputs[1]
        model_name = full_predictions["model"]
        confidence = full_predictions["confidence"]
        outputs = full_predictions["outputs"]
        # should limit what we are sending back. In fact strings are probably not correct and we may want to use ints and probably just the raw array or the confidence scores.
        assert isinstance(predictions_text, str)
        assert isinstance(model_name, str)
        assert isinstance(confidence, float)
        assert len(outputs) == 4

        try:
            logger.info(f"Response JSON: {response_json}")
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}", exc_info=True)
