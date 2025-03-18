"""Test the mainmodule activity endpoint. To test locally run python3 src/main_activity.py and then run pytest tests/test_main_activity.py.
"""

import json

import pandas as pd
import pytest
import requests
from atlantes.inference.common import ATLASResponse
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
    assert json_str is not None
    return json_str


class TestActivityFastApiEndpoint:
    def test_activity_endpoint_batching(self, json_track_request: str) -> None:
        """Test the activity endpoint to ensure batching works."""
        batch_size = 4
        track_data = json.loads(json_track_request)
        request_body = {
            "track_data": [
                {"track_id": f"test-{i}", "track_data": track_data}
                for i in range(batch_size)
            ],
        }
        response = requests.post(
            ACTIVITY_CLASSIFICATION_ENDPOINT, json=request_body, timeout=1000
        )
        response.raise_for_status()
        outputs = response.json()["predictions"]
        assert len(outputs) == 4
        assert all([isinstance(output["classification"], str) for output in outputs])
        assert all([isinstance(output["details"], dict) for output in outputs])

    def test_activity_endpoint(self, json_track_request: str) -> None:
        """Test the activity endpoint."""
        track_data = json.loads(json_track_request)
        REQUEST_BODY = {
            "track_data": [{"track_id": "test", "track_data": track_data}],
        }

        classification_response = requests.post(
            ACTIVITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        assert classification_response.status_code == 200

        logger.info(f"Response Headers: {classification_response.headers}")

        response_json = classification_response.json()
        output = ATLASResponse(**response_json)
        logger.info(f"{output=}")
        details = output.predictions[0].details
        assert isinstance(details["model"], str)
        assert isinstance(details["confidence"], float)
        assert len(details["outputs"]) == 4
