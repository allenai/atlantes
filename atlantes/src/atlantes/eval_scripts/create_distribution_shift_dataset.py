"""Script to create a dataset containing the outputs of a model currently in integration over a given time range.

This script is intended to be used to create a dataset for the purpose of evaluating the performance of a model currently in integration.
This script will allow us to see how the classification distirbution would have shifted given a new model.

To run this script we need to able to run the subpath debug service
https://allenai.atlassian.net/wiki/spaces/IUU/pages/edit-v2/30182047746 (internal)
This docuement describes how to start the integration kubernetes cluster to access the integration service.
"""

import datetime
from functools import partial
from typing import Any, Generator

import click
import dask
import pandas as pd
from atlantes.atlas.schemas import ActivityEndOfSequenceLabelDataModel
from atlantes.elastic_search.elastic_search_utils import (SUBPATH_INDEX,
                                                          get_es_client)
from atlantes.feedback.pull_track_parquet_from_subpath_id import \
    pull_parquet_from_event
from atlantes.log_utils import get_logger
from atlantes.utils import batch
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm

logger = get_logger(__name__)


def get_subpaths_ids_and_classifications(
    evaluation_start_time: str,
    evaluation_end_time: str,
    max_number_of_results: int = 1000,
) -> pd.DataFrame:
    """
    Get subpaths and their corresponding activity classifications within a given time frame.
    Parameters:
        evaluation_start_time (str): The start time of the evaluation period.
                                        format: "2024-08-22T00:00:00+00:00"
        evaluation_end_time (str): The end time of the evaluation period
                                    format: "2024-08-23T20:00:00+00:00"
        max_number_of_results (int): The maximum number of results to return.
    Returns:
        pd.DataFrame: A DataFrame containing the subpath IDs and their corresponding activity classifications as well as the model name.
    """
    MAX_HITS = 10000
    hits: list[dict[str, Any]] = []
    last_sort_value = None
    while len(hits) < max_number_of_results:
        page_size = min(MAX_HITS, max_number_of_results - len(hits))
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"start_time": {"gte": evaluation_start_time}}},
                        {"range": {"end_time": {"lte": evaluation_end_time}}},
                        {"exists": {"field": "activity_classification"}},
                    ],
                    "must_not": [
                        {"term": {"activity_classification": "time_gap"}},
                    ],
                }
            },
            "sort": [
                {
                    "start_time": "asc"
                },  # To ensure uniqueness, you can add _id as a secondary sort
            ],
            "size": page_size,  # The number of results to return per page
        }
        if last_sort_value:
            query["search_after"] = last_sort_value
        es_client = get_es_client()
        response = es_client.search(index=SUBPATH_INDEX, body=query)
        response_hits = response["hits"]["hits"]
        if len(response_hits) > 0:
            last_sort_value = response_hits[-1]["sort"]
            logger.info(f"{last_sort_value=}")
        else:
            break
        hits.extend(response_hits)
        logger.info(f"Currently found {len(hits)} subpaths")

    logger.info(f"Found {len(hits)} subpaths")

    subpath_ids = []
    activity_classifications = []
    model_name_list = []
    classificaiton_send_times = []
    track_ids = []
    for hit in hits:
        subpath_id = hit["_id"]
        subpath_ids.append(subpath_id)
        activity_classification = hit["_source"]["activity_classification"]
        activity_classifications.append(activity_classification)
        model_name = hit["_source"]["activity_details"]["model"]
        model_name_list.append(model_name)
        send_time = hit["_source"]["end_time"]
        classificaiton_send_times.append(send_time)
        track_id = hit["_source"]["track_id"]
        track_ids.append(track_id)

    df = pd.DataFrame(
        {
            "subpath_id": subpath_ids,
            "activity_classification": activity_classifications,
            "model_name": model_name_list,
            "send": classificaiton_send_times,
            "track_id": track_ids,
        }
    )
    return df


def pull_data_and_create_label_file_record(subpath_data: dict, output_dir: str) -> dict:
    """Pull data and create label file record.

    Parameters
    ----------
    subpath_data : tuple
        Tuple containing subpath data.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        Dictionary containing label file record.
    """
    subpath_id = subpath_data["subpath_id"]
    model_name = subpath_data["model_name"]
    activity_classification = subpath_data["activity_classification"]
    send_time = subpath_data["send"]
    track_id = subpath_data["track_id"]
    logger.info(f"Processing {subpath_id=}")
    # different file for every piece of feedback
    # What if we just added a column with the event_id, feedback, event timestamp, and originally predicted class
    output_path = pull_parquet_from_event(subpath_id, output_dir)
    return {
        "raw_paths": [output_path],
        "trackId": track_id,
        "activity_type_name": activity_classification,
        "send": send_time,
        "model_name": model_name,
    }


def df_tuple_iterator(df: pd.DataFrame) -> Generator[dict, None, None]:
    """Return a tuple of the DataFrame's columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame.

    Returns
    -------
    tuple
        Tuple of the DataFrame's columns.
    """
    for data in df.itertuples(index=False):
        yield {
            "subpath_id": data.subpath_id,
            "model_name": data.model_name,
            "activity_classification": data.activity_classification,
            "send": data.send,
            "track_id": data.track_id,
        }


@click.command()
@click.option(
    "--evaluation_start_time",
    default="2024-08-22T00:00:00+00:00",
    help="Start time of the evaluation period. Format: 'YYYY-MM-DDTHH:MM:SS+00:00'",
)
@click.option(
    "--evaluation_end_time",
    default="2024-08-24T20:00:00+00:00",
    help="End time of the evaluation period. Format: 'YYYY-MM-DDTHH:MM:SS+00:00'",
)
@click.option(
    "--output_dir",
    default="gs://ais-track-data/regression_test_data/",
    help="Output directory for the data",
)
@click.option(
    "--max_number_of_results", default=10000, help="Maximum number of results to return"
)
@click.option("--chunk_size", default=10000, help="Chunk size for processing the data")
def create_distribution_shift_dataset(
    evaluation_start_time: str,
    evaluation_end_time: str,
    output_dir: str,
    max_number_of_results: int,
    chunk_size: int,
) -> None:
    DT_STRING = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logger.info(
        f"Pulling subpaths from {evaluation_start_time} to {evaluation_end_time}"
    )
    df = get_subpaths_ids_and_classifications(
        evaluation_start_time,
        evaluation_end_time,
        max_number_of_results=max_number_of_results,
    )
    logger.info(df.head())
    logger.info(df.activity_classification.value_counts())
    logger.info(df.model_name.value_counts())

    process_func = delayed(
        partial(pull_data_and_create_label_file_record, output_dir=output_dir)
    )

    data_list = [data_input for data_input in df_tuple_iterator(df)]
    label_record_list = []
    with ProgressBar():
        for data_chunk in tqdm(
            batch(data_list, chunk_size), total=len(data_list) // chunk_size
        ):
            tasks = []
            for data in data_chunk:
                tasks.append(process_func(data))
            label_record_list.extend(dask.compute(*tasks))

    output_df = pd.DataFrame(label_record_list)
    validated_df = ActivityEndOfSequenceLabelDataModel.to_schema().validate(output_df)
    validated_df.to_csv(
        f"gs://ais-track-data/labels/regression_test_data/end_of_sequence_{DT_STRING}/distribution_8-22_to_8-23_{max_number_of_results}.csv",
        index=False,
    )


if __name__ == "__main__":
    create_distribution_shift_dataset()
