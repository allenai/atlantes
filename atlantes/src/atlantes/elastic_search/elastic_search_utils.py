"""Utils to access different Skylight Elasticsearch indices."""

from functools import lru_cache
from typing import Any, Optional

from atlantes.elastic_search.es_auth import SEARCH_PASSWORD, SEARCH_USERNAME
from atlantes.log_utils import get_logger

logger = get_logger(__name__)

ES_HOSTS = {
    "integration": "skylight-int-a-2.es.us-west1.gcp.cloud.es.io",
    "production": "skylight-prod-a.es.us-west1.gcp.cloud.es.io",
}
ES_PORT = 443  # for https

# Indices
TRACK_ANNOTATION_INDEX = "track_annotation_tool"
TRACK_ANNOTATION_SUMMARY_INDEX = "track_annotation_summaries"
SUBPATH_INDEX = "subpath"
SEARCH_HISTORY_INDEX = "event-history"


# Global variable to store the Elasticsearch class
Elasticsearch: Optional[type] = None


def load_elasticsearch() -> None:
    """Lazy-load the Elasticsearch class."""
    global Elasticsearch
    if Elasticsearch is None:
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            logger.error(
                "Failed to import Elasticsearch. Make sure it's installed if you need it.",
                exc_info=True,
            )
            return


@lru_cache(maxsize=2)
def get_es_client(env: str = "integration") -> Any:
    """
    Initialize and return an Elasticsearch client based on the environment.

    Parameters
    ----------
    env : str
        Environment to connect to ('integration' or 'production').

    Returns
    -------
    Elasticsearch
        Initialized Elasticsearch client.

    Raises
    ------
    ValueError
        If an invalid environment is provided.
    ImportError
        If Elasticsearch cannot be imported.
    """
    env = env.lower()
    if env not in ES_HOSTS:
        raise ValueError(f"Invalid environment: {env}")

    load_elasticsearch()
    if Elasticsearch is None:
        raise ImportError("Elasticsearch is not available")

    try:
        return Elasticsearch(
            [{"host": ES_HOSTS[env], "port": ES_PORT, "scheme": "https"}],
            http_auth=(SEARCH_USERNAME, SEARCH_PASSWORD),
            request_timeout=30,
            max_retries=2,
            retry_on_timeout=True,
        )
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}", exc_info=True)
        raise
