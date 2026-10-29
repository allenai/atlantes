"""Grab a parquet file from an event and save it to a local directory
https://allenai.atlassian.net/wiki/spaces/IUU/pages/edit-v2/30182047746 (internal)
This docuement describes how to start the integration kubernetes cluster to access the integration service.
"""

import os
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import requests
from atlantes.utils import get_logger

logger = get_logger(__name__)


def get_classification_trajectory(
    subpath_id: str, model: str = "atlas_activity", min_positions_required: int = 7
) -> dict[str, Any]:
    """
    Retrieves the activity or entity classification trajectory for a given subpath ID.
    Ensure the kubernetes cluster is spun up and the API is running before calling this function.
    Parameters:
        base_url (str): The base URL of the API where the endpoint is hosted.
        subpath_id (str): The unique identifier for the subpath.
        model (str, optional): The model type to query. This can be 'atlas_activity' for activity classification
                               or 'atlas_entity' for entity classification. Defaults to 'atlas_activity'.
        min_positions_required (int, optional): The minimum number of position points required to perform the classification
                                                without encountering a "not-enough-positions-available-to-classify" error.
                                                Defaults to 7, but should be adjusted according to the specific requirements
                                                of the API and the data available.
    Returns:
        dict: The JSON response from the API containing the classification trajectory. If the response status code is not 200,
              raises an exception.
    Raises:
        Exception: An error occurred during the API request, including details from the HTTP response.
    Example:
        >>> subpath_id = "fa687c12-63a1-49a2-a42f-5c61eb46fe9c"
        >>> response = get_classification_trajectory(base_url, subpath_id)
        >>> print(response)
    """
    # Construct the URL
    url = "http://localhost:5102/subpaths/debug/classification/trajectory"
    params = {
        "subpath_id": subpath_id,
        "model": model,
        "min_positions_required": min_positions_required,
    }  # type: ignore

    # Make the HTTP GET request
    response = requests.get(url, params=params, timeout=300)  # type: ignore

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to retrieve data: {response.status_code} {response.text}"
        )


def convert_response_to_df(trajectory_response: dict[str, Any]) -> pd.DataFrame:
    """
    Convert the classification trajectory response to a pandas DataFrame.
    Parameters:
        trajectory_response (dict): The JSON response from the API containing the classification trajectory.
    Returns:
        pd.DataFrame: The classification trajectory as a pandas DataFrame.
    Example:
        >>> response = get_classification_trajectory(base_url, subpath_id)
        >>> df = convert_response_to_df(response)
        >>> print(df)
    """
    # Extract the trajectory data from the response
    trajectory_data = trajectory_response["trajectory"]
    logger.info(trajectory_response["subpath"])

    logger.info(trajectory_response.keys())
    df = pd.DataFrame(trajectory_data)
    return df


def pull_parquet_from_event(subpath_id: str, output_dir: str) -> str:
    """Pull a parquet file from an event and save it to a local directory
    Parameters:
        subpath_id (str): The unique identifier for the subpath.

        output_dir (str): The directory to save the parquet file.

    Returns:
        str: The path to the saved parquet file.
    """
    response = get_classification_trajectory(subpath_id)
    start_time = response["subpath"]["start_time"]
    end_time = response["subpath"]["end_time"]
    activity_classification = response["subpath"]["activity_classification"]
    # Convert the response to a DataFrame
    df = convert_response_to_df(response)
    logger.info(df.tail())
    logger.info(f" final send times {df.send.values[-5:]}")
    logger.info(df.send.values[0:5])
    trackId = df["trackId"].iloc[0]
    month = pd.to_datetime(df.iloc[0]["send"]).month
    month_str = "0" + str(month) if month < 10 else str(month)
    # Save the DataFrame to a parquet file
    name = Path(
        f"{trackId}_{month_str}_{start_time}_{end_time}_{activity_classification}.parquet"
    )
    logger.info(f"Saving parquet file to {name}")
    os.makedirs(output_dir, exist_ok=True)
    if "gs://" not in output_dir:
        output_path = Path(output_dir) / name
        output_str = str(output_path.resolve())
    else:
        output_str = str(output_dir) + str(name)
    logger.info(f"Saving parquet file to {output_str}")
    df.to_parquet(
        output_str,
        engine="pyarrow",
    )
    return output_str

@click.command()
@click.option(
    "--subpath_id",
    help="Comma-separated list of unique identifiers for the subpath.",
)
@click.option(
    "--subpath_csv",
    help="Path to a CSV file containing subpath IDs.",
)
@click.option(
    "--output_dir",
    required=True,
    help="The directory to save the parquet file",
)
def pull_parquet_from_event_cli(
    subpath_id: Optional[str],
    subpath_csv: Optional[str],
    output_dir: str
) -> None:
    """CLI entry point for pulling parquet from an event."""
    if subpath_id and subpath_csv:
        raise click.UsageError(
            "Please provide either --subpath_id or --subpath_csv, not both."
        )

    if subpath_id:
        subpath_ids = subpath_id.split(",")
    elif subpath_csv:
        subpath_ids = pd.read_csv(subpath_csv)["subpath_id"].tolist()
    else:
        raise click.UsageError(
            "Please provide either --subpath_id or --subpath_csv."
        )

    for subpath in subpath_ids:
        try:
            pull_parquet_from_event(subpath.strip(), output_dir)
        except Exception as e:
            logger.error(f"Error pulling parquet from event {subpath}: {e}")


if __name__ == "__main__":
    pull_parquet_from_event_cli()
