"""Pull in app feedback from the es
Steps (internal):

1. gcloud container clusters get-credentials sky-int-a --zone us-west1-a --project skylight-int-a-r2d2
2. kubectl config set-context --current --namespace=sky-int-a
3. kubectl port-forward -n sky-int-a service/mda-api 5000:5000
4. Export SKYLIGHT_USERNAME
5. Export SKYLIGHT_INT_PASSWORD
6. Define and run this bash function get-int-token() {
  export token=$(curl -s -k -X POST -H "Accept: application/json" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=password&client_id=mda&username=$SKYLIGHT_USERNAME&password=$SKYLIGHT_INT_PASSWORD" \
    https://sc-integration.skylight.earth/oauth/token | jq .access_token | sed 's/"//g')
}
7. Run this script

# TODO: Have a better elasticsearch setup (likely) we would want to move that to other skylight repos as they already do this well
"""

import os
from typing import Literal, NamedTuple, Optional

import click
import pandas as pd
import requests
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.schemas import FeedbackActivityLabelDataModel
from atlantes.elastic_search.elastic_search_utils import (SEARCH_HISTORY_INDEX,
                                                          SUBPATH_INDEX,
                                                          get_es_client)
from atlantes.feedback.pull_track_parquet_from_subpath_id import \
    pull_parquet_from_event
from atlantes.log_utils import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


def get_feedback_from_skylight() -> pd.DataFrame:
    """Get feedback from Skylight.
    Returns:
        pd.DataFrame: The feedback DataFrame.
    """
    feedback_url = "http://localhost:5000/api/in-app-feedback/?event_type=atlas_fishing"
    response = requests.get(
        feedback_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + str(os.environ.get("token")),
        },
        timeout=300,
    )
    if response.status_code == 200:
        feedback_dict = response.json()
    else:
        logger.info("Request failed with status code:", response.status_code)
        logger.info(response.text)
        raise ValueError("Request failed with status code:", response.status_code)
    return pd.DataFrame(feedback_dict["records"]).drop(columns=["image_url"])


FEEDBACK_VALUES = Literal["GOOD", "BAD", "UNSURE"]


def resolve_raw_feedback_to_label(
    feedback_value: FEEDBACK_VALUES, additional_context: str
) -> Optional[str]:
    """Returns an activity label from the feedback value and any additional context about the true class

    Assumes that the event_type is atlas_fishing
    """
    feedback_class_name = None
    if feedback_value == "GOOD":
        feedback_class_name = "fishing"
    elif feedback_value == "BAD":
        feedback_class_name = (
            additional_context
            if additional_context in AtlasActivityLabelsTraining.to_class_descriptions()
            else None
        )
    return feedback_class_name


def get_track_id_from_event_id(event_id: str) -> tuple[str, str, str]:
    """
    Get the track ID from the event ID.
    Parameters:
        event_id (str): The event ID.
    Returns:
        tuple[str, str, str]: The track ID, start time, and end time.
    """
    try:
        es_client = get_es_client()
        query = {
            "query": {"term": {"event_id": {"value": event_id}}},
            "_source": ["event_id", "vessels.vessel_0.track_id", "start", "end"],
        }
        response = es_client.search(
            index=SEARCH_HISTORY_INDEX, body=query
        )
        source_document = response["hits"]["hits"][0]["_source"]
        track_id = source_document["vessels"]["vessel_0"]["track_id"]
        start_time = source_document["start"]["time"]
        end_time = source_document["end"]["time"]
    except Exception as e:
        raise e
    return track_id, start_time, end_time


class SubpathInformation(NamedTuple):
    """Named tuple for subpath information.


    This is the information needed about a subpath to pull the parquet file
    and create an entry into a label file/database"""

    subpath_id: str
    subpath_start_time: str
    subpath_end_time: str
    predicted_activity_label: str
    # Additional fields for checking feedback data
    mmsi: Optional[int]
    category: Optional[int]
    outputs: Optional[dict]


class FeedbackLabelInformation(BaseModel):
    raw_paths: list[str]
    trackId: str
    event_id: str  # Add event id
    activity_type_name: str  # Feedback activity label
    send: str  # Equivalent to subpath_end_time
    predicted_activity_label: str  # Originally predicted activity label
    subpath_id: str
    subpath_start_time: str
    # Additional fields for checking feedback data
    id: str  # Feedback ID
    user_id: str  # User ID
    username: str  # Username
    mmsi: Optional[int]
    category: Optional[int]
    outputs: Optional[dict]


def get_subpath_information_from_trackId_sendtime(
    track_id: str, event_start_time: str, event_end_time: str
) -> list[SubpathInformation]:
    """
    Get the subpath ID from the track ID and send time range.
    Parameters:
        track_id (str): The track ID.
    Returns:
        str: The subpath ID.
    """
    es_client = get_es_client()
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"track_id": track_id}},
                    {"range": {"start_time": {"gte": event_start_time}}},
                    {"range": {"end_time": {"lte": event_end_time}}},
                ]
            }
        }
    }
    response = es_client.search(index=SUBPATH_INDEX, body=query)
    if (
        response["hits"]["hits"][0]["_source"]["start_time"]
        < "2024-08-01T09:51:32+00:00"
    ):
        raise Exception("Old type of prediction")
    logger.info(response)
    # Need to add something to reject if it is the old type of predicition
    subpath_information_lst = []
    for source_dict in response["hits"]["hits"]:
        source_document = source_dict["_source"]
        logger.info(f"{source_document=}")

        # Extract mmsi, outputs, and category
        mmsi = source_document.get("vessels", {}).get("vessel_0", {}).get("mmsi")
        category = source_document.get("activity_details", {}).get("params", {}).get("vessel", {}).get("category")
        outputs = source_document.get("activity_details", {}).get("outputs", [])

        subpath_info = SubpathInformation(
            subpath_id=source_document["id"],
            subpath_start_time=source_document["start_time"],
            subpath_end_time=source_document["end_time"],
            predicted_activity_label=source_document["activity_details"][
                "postprocessed_classification"
            ],
            mmsi=mmsi,
            category=category,
            outputs=outputs,
        )
        subpath_information_lst.append(subpath_info)
    return subpath_information_lst


def pull_input_data_for_event_id_and_create_feedback_label(
    event_id: str, feedback_label: str, output_dir: str, user_id: str, username: str, feedback_id: str
) -> list[dict]:
    """
    Pull the input data for an event ID and create a feedback label.
    Parameters:
        event_id (str): The event ID.
        feedback_label (str): The feedback label.
        output_dir (str): The output directory.

    Returns:
        dict: The feedback label information.
    """
    track_id, start_time, end_time = get_track_id_from_event_id(event_id)
    subpath_information_lst = get_subpath_information_from_trackId_sendtime(
        track_id, event_start_time=start_time, event_end_time=end_time
    )
    feedback_list = []
    for subpath_info in subpath_information_lst:
        output_path = pull_parquet_from_event(subpath_info.subpath_id, output_dir)
        feedback_list.append(
            FeedbackLabelInformation(
                raw_paths=[output_path],
                trackId=track_id,
                event_id=event_id,  # Add event id
                activity_type_name=feedback_label,
                send=end_time,
                predicted_activity_label=subpath_info.predicted_activity_label,
                subpath_id=subpath_info.subpath_id,
                subpath_start_time=subpath_info.subpath_start_time,
                id=feedback_id,
                user_id=user_id,
                username=username,
                mmsi=subpath_info.mmsi,
                category=subpath_info.category,
                outputs=subpath_info.outputs,
            ).model_dump()
        )
    return feedback_list


@click.command()
@click.option(
    "--output-dir",
    default="gs://ais-track-data/feedback_input_data/",
    help="Output directory for feedback input data",
)
def create_labeled_dataset_from_feedback(output_dir: str) -> None:
    # Feedback DataFrame
    feedback_df = get_feedback_from_skylight()
    feedback_df.loc[:, "feedback_activity_label"] = feedback_df[
        ["value", "additional_context"]
    ].apply(lambda x: resolve_raw_feedback_to_label(x[0], x[1]), axis=1)
    # Drop all feedback with a nan label
    feedback_df = feedback_df.dropna(subset=["feedback_activity_label"])

    logger.info(feedback_df.feedback_activity_label.value_counts())
    logger.info(f"Feedback length: {len(feedback_df)}")
    logger.info(f"Feedback columns: {feedback_df.columns}")

    if feedback_df is not None:
        if "gs://" not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
        label_list = []
        for event_id, feedback_label, user_id, username, feedback_id in zip(
            feedback_df.event_id,
            feedback_df.feedback_activity_label,
            feedback_df.user_id,
            feedback_df.username,
            feedback_df.id
        ):
            logger.info(f"Processing {event_id=} {feedback_label=}")
            try:
                feedback_label_info_dict_list = (
                    pull_input_data_for_event_id_and_create_feedback_label(
                        event_id, feedback_label, output_dir, user_id, username, feedback_id
                    )
                )
                label_list.extend(feedback_label_info_dict_list)
            except Exception as e:
                logger.info(f"Error: {e}")
                continue
        output_df = pd.DataFrame(label_list)
        try:
            # Validate the DataFrame with the updated schema
            validated_df = FeedbackActivityLabelDataModel.to_schema().validate(output_df)
            logger.info("Validation passed")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

        # Need to change this name
        output_path = (
            "gs://ais-track-data/labels/feedback/end_of_sequence/test_labels.csv"
        )
        validated_df.to_csv(
            output_path,
            index=False,
        )
        logger.info(validated_df.head())
        logger.info(f"Dataset file Saved to {output_path}")


if __name__ == "__main__":
    create_labeled_dataset_from_feedback()
