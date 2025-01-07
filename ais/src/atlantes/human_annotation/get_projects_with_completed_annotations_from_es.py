"""Get completed annotation projects from ES and save them to a json file in the
format to put it into airflow

SET SEARCH_USERNAME and SEARCH_PASSWORD as env variables read in the es_auth.py
file
"""

import json
import os
from typing import Tuple

from atlantes.elastic_search.elastic_search_utils import (
    TRACK_ANNOTATION_INDEX, get_es_client)
from atlantes.log_utils import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

PRODUCTION_ES_SEARCH_CLIENT = get_es_client("production")


def get_all_projects() -> list[str]:
    """Get all projects from the response"""
    # Query list of all project IDs
    project_id_agg_query = {
        "size": 0,
        "aggs": {
            "project_counts": {
                "terms": {
                    "field": "project_id",
                    "size": 100000,  # Must be bigger than the number of projects
                }
            }
        },
    }
    response = PRODUCTION_ES_SEARCH_CLIENT.search(
        index=TRACK_ANNOTATION_INDEX, body=project_id_agg_query
    )
    response = response["aggregations"]["project_counts"]["buckets"]
    return [project["key"] for project in response]


def get_trajectories_for_project(project_id: str) -> list[str]:
    """
    Returns a list of all trajectories within a given project
    """
    query = {
        "query": {"term": {"project_id": project_id}},
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {"field": "trajectory_id", "size": 100000}
            }
        },
    }
    res = PRODUCTION_ES_SEARCH_CLIENT.search(index=TRACK_ANNOTATION_INDEX, body=query)
    trajectories = [
        b["key"] for b in res["aggregations"]["unique_trajectory_ids"]["buckets"]
    ]
    return trajectories


# FIx the query here
def is_trajectory_completed(project_id: str, trajectory_id: str) -> bool:
    """
    Returns a boolean indicating whether a given trajectory is fully annotated
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"trajectory_id": trajectory_id}},
                    {"term": {"project_id": project_id}},
                    {"term": {"annotated": False}},
                ],
            }
        }
    }
    res = PRODUCTION_ES_SEARCH_CLIENT.search(index=TRACK_ANNOTATION_INDEX, body=query)
    return res["hits"]["total"]["value"] == 0


def get_num_trajectories_with_fishing_annotation() -> int:
    """
    Returns a boolean indicating whether a given trajectory contains fishing activity
    """
    query = {
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {"field": "trajectory_id", "size": 100000000}
            }
        },
        "query": {
            "bool": {
                "must": [{"term": {"annotation.activity_type.name": "fishing"}}],
            },
        },
    }

    res = PRODUCTION_ES_SEARCH_CLIENT.search(index=TRACK_ANNOTATION_INDEX, body=query)
    return len(res["aggregations"]["unique_trajectory_ids"]["buckets"])


def build_json_of_projects_containing_completely_labeled_trajectory() -> list[str]:
    """Builds a json file of completed projects from the response"""
    # Make this memoized by reading the json file if it exists
    if os.path.exists("annotation_projects_with_completed_trajectories.json"):
        with open(
            "annotation_projects_with_completed_trajectories.json", "r"
        ) as infile:
            project_json = json.load(infile)
        projects_with_completed_trajectories = project_json["projects"].split(",")
    # Query list of all project IDs
    projects = get_all_projects()
    logger.info(f"Number of projects: {len(projects)}")
    projects_with_completed_trajectories = []
    projects_with_uncompleted_trajectories = []
    for project in tqdm(
        projects, desc="checking which projects have completed trajectories"
    ):
        if project in projects_with_completed_trajectories:
            continue
        trajectories = get_trajectories_for_project(project)
        for trajectory in trajectories:
            if is_trajectory_completed(project, trajectory):
                projects_with_completed_trajectories.append(project)
                break
        else:
            projects_with_uncompleted_trajectories.append(project)

    logger.info(f"completed {len(projects_with_completed_trajectories)} projects")
    project_str = (",").join(projects_with_completed_trajectories)
    project_json = {"projects": project_str}

    with open("annotation_projects_with_completed_trajectories.json", "w") as outfile:
        json.dump(project_json, outfile)
    logger.info(f"Number of projects with completed trajectories: {len(project_json)}")
    logger.info(
        f"Number of projects with uncompleted trajectories: {len(projects_with_uncompleted_trajectories)}"
    )
    logger.info(
        f"Projects with uncompleted trajectories: {projects_with_uncompleted_trajectories}"
    )
    return projects_with_completed_trajectories


# Query number of completed annotated tracks
def how_many_tracks_are_completely_annotated() -> Tuple[list[str], list[str]]:
    """How many tracks are completely annotated"""

    project_id_agg_query = {
        "size": 0,
        "aggs": {
            "all_trajectory_ids": {
                "terms": {
                    "field": "trajectory_id",
                    "size": 100000,  # Must be bigger than the number of trajectory_ids (single months of a trackId)
                }
            }
        },
    }
    response = PRODUCTION_ES_SEARCH_CLIENT.search(
        index=TRACK_ANNOTATION_INDEX, body=project_id_agg_query
    )
    response = response["aggregations"]["all_trajectory_ids"]["buckets"]
    logger.info(f"Number of tracks: {len(response)}")
    completed_trajectories = []
    incomplete_trajectories = []
    for trajectory in tqdm(
        response, desc="checking which trajectories are completley labeled"
    ):
        trajectory_id = trajectory["key"]
        are_any_points_in_trajectory_unannotated_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trajectory_id": trajectory_id}},
                        {"term": {"annotated": False}},
                    ],
                }
            }
        }

        response = PRODUCTION_ES_SEARCH_CLIENT.search(
            index=TRACK_ANNOTATION_INDEX,
            body=are_any_points_in_trajectory_unannotated_query,
            size=1,
        )
        if response["hits"]["total"]["value"] > 0:
            logger.info(f"Trajectory {trajectory_id} has unannotated points")
            incomplete_trajectories.append(trajectory_id)
            continue
        completed_trajectories.append(trajectory_id)
    logger.info(f"Number of completed trajectories: {len(completed_trajectories)}")
    logger.info(f"Number of incomplete trajectories: {len(incomplete_trajectories)}")
    return completed_trajectories, incomplete_trajectories


def how_many_annotated_points() -> int:
    """How many annotated points are there"""
    annotated_points_query = {
        "size": 0,
        "aggs": {"annotated_points": {"filter": {"exists": {"field": "annotated"}}}},
    }
    response = PRODUCTION_ES_SEARCH_CLIENT.search(
        index=TRACK_ANNOTATION_INDEX, body=annotated_points_query
    )
    response = response["aggregations"]["annotated_points"]["doc_count"]
    logger.info(f"Number of annotated points: {response}")
    return response


def get_all_projects_completely_labeled_as_fishing() -> list:
    """Get all projects that are completely labeled as fishing"""
    # Query list of all trajectory_Ids with fishing
    query = {
        "size": 0,
        "aggs": {
            "unique_trajectory_ids": {
                "terms": {"field": "trajectory_id", "size": 100000000}
            }
        },
        "query": {
            "bool": {
                "must": [{"term": {"annotation.activity_type.name": "fishing"}}],
            },
        },
    }
    res = PRODUCTION_ES_SEARCH_CLIENT.search(index=TRACK_ANNOTATION_INDEX, body=query)
    all_fishing_tracks = [
        b["key"] for b in res["aggregations"]["unique_trajectory_ids"]["buckets"]
    ]
    # see how many of these have no transitingpoints
    only_fishing_tracks = []
    for trajectory_id in tqdm(
        all_fishing_tracks, desc="checking which projects have only fishing"
    ):
        query = {
            "size": 0,
            "aggs": {
                "unique_trajectory_ids": {
                    "terms": {"field": "trajectory_id", "size": 100000000}
                }
            },
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trajectory_id": trajectory_id}},
                        {"term": {"annotation.activity_type.name": "transiting"}},
                    ],
                },
            },
        }
        res = PRODUCTION_ES_SEARCH_CLIENT.search(
            index=TRACK_ANNOTATION_INDEX, body=query
        )
        if len(res["aggregations"]["unique_trajectory_ids"]["buckets"]) == 0:
            only_fishing_tracks.append(trajectory_id)
    logger.info(f"Number of tracks with only fishing: {len(only_fishing_tracks)}")
    return only_fishing_tracks


if __name__ == "__main__":
    build_json_of_projects_containing_completely_labeled_trajectory()
