"""Script for bulk uploading feedback on human annotations to the annotation tool


Please Sanity Check the code before running it on production data

To set up Authentication, you need to set the following environment variables:
- SKYLIGHT_PROD_USERNAME : Username for the Skylight
- SKYLIGHT_PROD_PASSWORD : Password for the Skylight
- SEARCH_USERNAME : Username for the ElasticSearch
- SEARCH_PASSWORD : Password for the ElasticSearch
"""

import json
import os
from typing import Any, Callable, Literal, Optional

import requests
from atlantes.elastic_search.elastic_search_utils import (
    TRACK_ANNOTATION_INDEX, TRACK_ANNOTATION_SUMMARY_INDEX, get_es_client)
from atlantes.log_utils import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

status_literal = Literal[
    "ANNOTATOR-SUBMITTED",
    "REVIEWER-COMMENTED",
    "REVIEWER-ACCEPTED",
    "REVIEWER-REJECTED",
]


def get_trajectory_project_pairs_all_labeled_fishing() -> list[tuple[str, str]]:
    """Get all projects that are completely labeled as fishing"""
    # Query list of all trajectory_Ids with fishing
    es_client = get_es_client("production")
    query = {
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {"field": "trajectory_id", "size": 10000},
                "aggs": {
                    "project_ids": {"terms": {"field": "project_id", "size": 10000}}
                },
            }
        },
        "query": {
            "bool": {
                "must": [{"term": {"annotation.activity_type.name": "fishing"}}],
            },
        },
    }
    res = es_client.search(index=TRACK_ANNOTATION_INDEX, body=query)
    project_id_trajectory_id_pairs = []
    for record in res["aggregations"]["unique_trajectory_ids"]["buckets"]:
        trajectory_id = record["key"]
        project_id_trajectory_id_pairs.extend(
            [
                (trajectory_id, project["key"])
                for project in record["project_ids"]["buckets"]
                if project["key"] is not None
            ]
        )
    # see how many of these have no transitingpoints
    project_id_trajectory_ids_no_transiting = []
    for trajectory_id, project_id in tqdm(
        project_id_trajectory_id_pairs, desc="checking which projects have only fishing"
    ):
        query = {
            "size": 0,
            "aggs": {
                "unique_trajectory_ids": {
                    "terms": {"field": "trajectory_id", "size": 10000},
                    "aggs": {
                        "project_ids": {"terms": {"field": "project_id", "size": 10000}}
                    },
                }
            },
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trajectory_id": trajectory_id}},
                        {"term": {"project_id": project_id}},
                        {"term": {"annotation.activity_type.name": "transiting"}},
                    ],
                },
            },
        }
        res = es_client.search(index=TRACK_ANNOTATION_INDEX, body=query)
        if len(res["aggregations"]["unique_trajectory_ids"]["buckets"]) == 0:
            logger.info(f"Trajectory {trajectory_id} has only fishing")
            project_id_trajectory_ids_no_transiting.append((trajectory_id, project_id))
    project_id_trajectory_id_pairs_only_fishing = []
    for trajectory_id, project_id in project_id_trajectory_ids_no_transiting:
        query = {
            "size": 0,
            "aggs": {
                "unique_trajectory_ids": {
                    "terms": {"field": "trajectory_id", "size": 10000},
                    "aggs": {
                        "project_ids": {"terms": {"field": "project_id", "size": 10000}}
                    },
                }
            },
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trajectory_id": trajectory_id}},
                        {"term": {"project_id": project_id}},
                        {"term": {"annotation.activity_type.name": "other"}},
                    ],
                },
            },
        }
        res2 = es_client.search(index=TRACK_ANNOTATION_INDEX, body=query)
        if len(res2["aggregations"]["unique_trajectory_ids"]["buckets"]) == 0:
            project_id_trajectory_id_pairs_only_fishing.append(
                (trajectory_id, project_id)
            )

    logger.info(
        f"Number of tracks with only fishing: {len(project_id_trajectory_id_pairs_only_fishing)}"
    )
    logger.info(
        f"Tracks with only fishing and no transiting or unknown: {project_id_trajectory_id_pairs_only_fishing}"
    )
    return project_id_trajectory_id_pairs_only_fishing


def get_auth_credentials() -> dict:

    url = "https://sc-production.skylight.earth/oauth/token"
    username = os.environ.get("SKYLIGHT_PROD_USERNAME")
    password = os.environ.get("SKYLIGHT_PROD_PASSWORD")
    if username is None or password is None:
        logger.error(
            "Please set the SKYLIGHT_PROD_USERNAME and SKYLIGHT_PROD_PASSWORD as environment variables"
        )
        raise ValueError(
            "Please set the SKYLIGHT_PROD_USERNAME and SKYLIGHT_PROD_PASSWORD"
        )
    data = {
        "grant_type": "password",
        "client_id": "mda",
        "username": username,  # Replace <user> with the actual username
        "password": password,  # Replace <pass> with the actual password
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(url, headers=headers, data=data, timeout=1000)
    return response.json()


def upload_feedback(
    id: str,
    comment: Optional[str] = None,
    status: Optional[status_literal] = None,
    annotated_count: Optional[int] = None,
) -> None:
    """Upload feedback on human annotations programattically

    Example Data format: {
    id: str,
    comment: Optional[str] = None
    status: Optional[status_literal] = None
    annotated_count: Optional[int] = None}"""
    auth_token = get_auth_credentials()["access_token"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + auth_token,
    }

    request_data: dict[str, Any] = {
        "id": id,
    }
    if comment is not None:
        request_data["comment"] = comment
        logger.debug(f"Added comment: {comment}")
    if status is not None:
        request_data["status"] = status
        logger.debug(f"Added status: {status}")
    if annotated_count is not None:
        request_data["annotated_count"] = annotated_count
        logger.debug(f"Added annotated count: {annotated_count}")

    if annotated_count is None and status is None and comment is None:
        logger.warning("No data was added to the request")
        raise ValueError("No data was added to the request")

    logger.debug(f"Request data: {request_data}")
    url = "https://sc-production.skylight.earth/api/track-annotation/summary"
    response = requests.put(
        url, data=json.dumps(request_data), headers=headers, timeout=100
    )  # Convert test_data to JSON format
    logger.info(response.json())
    logger.info(response.status_code)


def get_summary_id_from_project_id_trajectory_id_pair(
    trajectory_id: str, project_id: str
) -> str:
    """Get the summary id from a project_id and trajectory_id"""
    es_client = get_es_client("production")
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"project_id": project_id}},
                    {"term": {"trajectory_id": trajectory_id}},
                ],
            }
        }
    }
    res = es_client.search(index=TRACK_ANNOTATION_SUMMARY_INDEX, body=query)
    logger.info(f"Number of summary ids: {len(res['hits']['hits'])}")
    return res["hits"]["hits"][0]["_id"]


def get_lst_of_summary_ids_from_trajectory_project_pairs(
    trajectory_project_pairs: list[tuple[str, str]]
) -> list[str]:
    """Get the summary id from a project_id and trajectory_id"""
    return [
        get_summary_id_from_project_id_trajectory_id_pair(trajectory_id, project_id)
        for trajectory_id, project_id in trajectory_project_pairs
    ]


def check_if_project_has_accepted_feedback(id: str) -> bool:
    """Check if a track in A project has accepted feedback"""
    es_client = get_es_client("production")
    accepted_status = "REVIEWER-ACCEPTED"
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"id": id}},
                    # {"term": {"status": accepted_status}},
                ],
            }
        }
    }
    res = es_client.search(index=TRACK_ANNOTATION_SUMMARY_INDEX, body=query)
    return res["hits"]["hits"][0]["_source"].get("status", None) == accepted_status


def bulk_upload_feedback_on_human_annotations(
    query_function: Callable,
    status: Optional[status_literal] = None,
    comment: Optional[str] = None,
    annotated_count: Optional[int] = None,
) -> None:
    """Bulk upload feedback on human annotations based on a query function"""
    trajectory_project_pairs = query_function()
    summary_ids = get_lst_of_summary_ids_from_trajectory_project_pairs(
        trajectory_project_pairs
    )
    # test_summary_ids = [
    #     "sampled_trajectories_human_annotate14-12-2023-06-20-42_21:B:224068000:1636174607:1586572:1002967_06_summary",
    #     "sampled_trajectories_human_annotate14-12-2023-06-20-42_45:B:701031000:1646072651:1142597:421392_05_summary",
    #     "osr_examples_prelabeled_poc:B:403705150:1570930884:2300628:1175537_05_summary",
    # ]
    logger.info(f"Number of summary ids: {len(summary_ids)}")
    for summary_id in summary_ids:
        # Check if there is already accepted feedback before uploading
        if check_if_project_has_accepted_feedback(summary_id):
            logger.info(f"Feedback already accepted for summary id: {summary_id}")
            continue
        logger.info(f"Uploading feedback for summary id: {summary_id}")
        upload_feedback(
            id=summary_id,
            status=status,
            comment=comment,
            annotated_count=annotated_count,
        )


if __name__ == "__main__":
    all_fishing_bulk_comment = "Please mark all transiting sections between fishing as transiting,\
          as well as look for anchored and moored sections, \
            if any, additionally be sure to consider if this is fishing gear"
    bulk_upload_feedback_on_human_annotations(
        get_trajectory_project_pairs_all_labeled_fishing,
        status="REVIEWER-REJECTED",
        comment=all_fishing_bulk_comment,
    )
