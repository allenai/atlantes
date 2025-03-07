"""Test the mainmodule activity endpoint. To test locally run python3 src/main_activity.py and then run pytest tests/test_main_activity.py.
"""

import json

import pandas as pd
import pytest
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


class TestActivityFastApiEndpoint:
    def test_activity_endpoint_batching(self, json_track_request: str) -> None:
        """Test the activity endpoint to ensure batching works."""
        batch_size = 4
        request_body = {
            "tracks": [json.loads(json_track_request)] * batch_size,
        }
        response = requests.post(
            ACTIVITY_CLASSIFICATION_ENDPOINT, json=request_body, timeout=1000
        )
        response.raise_for_status()
        outputs = response.json()["predictions"]
        assert len(outputs) == 4
        assert all([len(outputs[i]) == 2 for i in range(4)])
        assert all([isinstance(outputs[i][0], str) for i in range(4)])
        assert all([isinstance(outputs[i][1], dict) for i in range(4)])

    def test_activity_endpoint(self, json_track_request: str) -> None:
        """Test the activity endpoint."""
        REQUEST_BODY = {
            "tracks": [json.loads(json_track_request)],  # Ensure proper JSON format
        }

        classification_response = requests.post(
            ACTIVITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        assert classification_response.status_code == 200

        logger.info(f"Response Headers: {classification_response.headers}")

        response_json = classification_response.json()
        logger.info(f"Response JSON: {response_json}")
        track_0_outputs = response_json["predictions"][0]
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
