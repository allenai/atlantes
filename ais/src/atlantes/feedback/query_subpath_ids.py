"""
Script for querying the subpath IDs from the elastic search index and saving them to a CSV file.
"""

import click
import pandas as pd
from atlantes.elastic_search.elastic_search_utils import (SUBPATH_INDEX,
                                                          get_es_client)
from atlantes.log_utils import get_logger
from atlantes.utils import BaseRegistry

logger = get_logger(__name__)


# Query definitions
FISHING_HIGH_SOG_QUERY = {
    "size": 10000,
    "query": {
        "bool": {
            "must": [
                {"term": {"activity_classification": "fishing"}},
                {"range": {"mean_sog": {"gt": 8}}},
                {"range": {"start_time": {"gte": "2024-10-06T00:00:00Z"}}},
            ]
        }
    },
    "_source": ["id"],
}


class QueryRegistry(BaseRegistry):
    pass


QUERY_REGISTRY = QueryRegistry()
QUERY_REGISTRY.register(FISHING_HIGH_SOG_QUERY, "fishing_high_sog")
# Add more queries here as needed


def query_subpath_ids(query: dict) -> pd.DataFrame:
    """
    Query the subpath IDs from the elastic search index
    """
    es_client = get_es_client()
    response = es_client.search(index=SUBPATH_INDEX, body=query)
    return pd.DataFrame([hit["_source"]["id"] for hit in response["hits"]["hits"]])


@click.command()
@click.option(
    "--query-key",
    type=click.Choice(list(QUERY_REGISTRY.registry.keys())),
    default="fishing_high_sog",
    help="Key of the query to use from the registry",
)
@click.option(
    "--output-file",
    default="subpath_ids.csv",
    help="Output CSV file name",
)
def main(query_key: str, output_file: str) -> None:
    """
    Main function to query the subpath IDs from the elastic search index
    """
    query_dict = QUERY_REGISTRY.get(query_key)
    result = query_subpath_ids(query_dict)
    result.to_csv(output_file, index=False, header=["subpath_id"])
    logger.info(f"Saved {len(result)} subpath IDs to {output_file}")


if __name__ == "__main__":
    main()
