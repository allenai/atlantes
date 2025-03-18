"""Testing the Entity Endpoint."""

import json

import pandas as pd
import pytest
import requests
from atlantes.inference.common import ATLASResponse
from atlantes.log_utils import get_logger

logger = get_logger(__name__)

ENTITY_CLASSIFICATION_ENDPOINT = "http://0.0.0.0:8001/classify"


@pytest.fixture(scope="class")
def json_track_request(test_ais_df1: pd.DataFrame) -> str:
    """Build an in-memory AIS track stream."""
    test_ais_df1_inference = test_ais_df1.copy().head(1000)
    json_str = test_ais_df1_inference.to_json(orient="records")
    assert json_str is not None
    return json_str


@pytest.fixture(scope="class")
def json_track_request_with_unknown_binned_ship_type_and_buoy_name(
    test_ais_df1: pd.DataFrame,
) -> str:
    """Build an in-memory AIS track stream."""
    test_ais_df1_inference = test_ais_df1.copy().head(1000)
    test_ais_df1_inference.loc[0, "category"] = 9999
    test_ais_df1_inference.loc[:, "name"] = "buoy"
    json_str = test_ais_df1_inference.to_json(orient="records")
    assert json_str is not None
    return json_str


@pytest.fixture(scope="class")
def json_track_request_with_known_binned_ship_type(test_ais_df1: pd.DataFrame) -> str:
    """Build an in-memory AIS track stream with a known ship type."""
    test_ais_df1_inference = test_ais_df1.copy().head(1000)
    test_ais_df1_inference.loc[:, "category"] = 70  # Set known ship type to 70
    json_str = test_ais_df1_inference.to_json(orient="records")
    assert json_str is not None
    return json_str


class TestEntityFastApiEndpoint:
    def test_entity_endpoint_batching(self, json_track_request: str) -> None:
        """Test the entity endpoint to ensure batching works."""
        batch_size = 4
        track_data = json.loads(json_track_request)
        request_body = {
            "track_data": [
                {"track_id": f"test-{i}", "track_data": track_data}
                for i in range(batch_size)
            ],
        }
        response = requests.post(
            ENTITY_CLASSIFICATION_ENDPOINT, json=request_body, timeout=1000
        )
        response.raise_for_status()
        outputs = response.json()["predictions"]
        assert len(outputs) == 4
        assert all([isinstance(outputs[i]["classification"], str) for i in range(4)])
        assert all([isinstance(outputs[i]["details"], dict) for i in range(4)])

    def test_entity_endpoint(self, json_track_request: str) -> None:
        """Test the entity endpoint."""
        track_data = json.loads(json_track_request)
        REQUEST_BODY = {
            "track_data": [{"track_id": "test", "track_data": track_data}],
        }

        classification_response = requests.post(
            ENTITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        classification_response.raise_for_status()

        logger.info(f"Response Headers: {classification_response.headers}")

        response_json = classification_response.json()
        output = ATLASResponse(**response_json)
        logger.info(f"{output=}")
        details = output.predictions[0].details
        assert isinstance(details["model"], str)
        assert isinstance(details["confidence"], float)
        assert len(details["outputs"]) == 2

    def test_entity_endpoint_with_known_binned_ship_type(
        self, json_track_request_with_known_binned_ship_type: str
    ) -> None:
        """Test the entity endpoint with known ship type."""
        track_data = json.loads(json_track_request_with_known_binned_ship_type)
        REQUEST_BODY = {
            "track_data": [{"track_id": "test", "track_data": track_data}],
        }

        classification_response = requests.post(
            ENTITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        assert classification_response.status_code == 200
        response_json = classification_response.json()
        assert response_json["predictions"][0]["classification"] == "vessel"
        assert (
            response_json["predictions"][0]["details"]["postprocess_rule_applied"]
            is True
        )

    def test_entity_endpoint_with_unknown_binned_ship_type_and_buoy_name(
        self, json_track_request_with_unknown_binned_ship_type_and_buoy_name: str
    ) -> None:
        """Test the entity endpoint with unknown ship type and buoy name."""
        track_data = json.loads(
            json_track_request_with_unknown_binned_ship_type_and_buoy_name
        )
        REQUEST_BODY = {
            "track_data": [{"track_id": "test", "track_data": track_data}],
        }

        classification_response = requests.post(
            ENTITY_CLASSIFICATION_ENDPOINT, json=REQUEST_BODY, timeout=1000
        )
        assert classification_response.status_code == 200
        response_json = classification_response.json()
        assert response_json["predictions"][0]["classification"] == "buoy"
        assert (
            response_json["predictions"][0]["details"]["postprocess_rule_applied"]
            is True
        )
