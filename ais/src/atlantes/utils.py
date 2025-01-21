""" Main utils file for the project.

#TODO: Clean up all the utils constants and configs so we don't ahve any circular imports in the future"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import os
from datetime import datetime
from functools import wraps
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Generator, Tuple, Union

import click
import contextily
import dask
import dask.dataframe as dd
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import pandera as pa
import pyproj
import torch
import yaml
from dask.diagnostics import ProgressBar
from geopandas import GeoDataFrame
from google.cloud import storage
from pandera.typing import DataFrame
from pydantic import BaseModel
from shapely.geometry import Point
from tqdm import tqdm

from atlantes.datautils import MMSI_FLAG_CODES
from atlantes.human_annotation.schemas import LocationDataModel
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


def find_most_recent_checkpoint(load_checkpoint_dir: str) -> str:
    """Find the most recent checkpoint in a directory.

    Parameters
    ----------
    checkpoint_dir : str
        The directory where the checkpoints are stored.

    Returns
    -------
    tuple[str,str]
        The path to the most recent checkpoint and the checkpoint name.


    Raises
    ------
    FileNotFoundError
        If no checkpoints are found in the directory.
    """
    os.makedirs(load_checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(load_checkpoint_dir)
    checkpoints = [f for f in checkpoint_dir.glob("*.pt")]
    logger.info(f"Checkpoints found: {checkpoints}")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {load_checkpoint_dir}")
    most_recent_checkpoint = max(checkpoints)
    logger.info(f"Most recent checkpoint: {str(most_recent_checkpoint)}")
    return most_recent_checkpoint.name


def load_all_metadata_indexes(
    paths_to_metadata_indexes: list[str], **kwargs: Any
) -> DataFrame:
    """Load all metadata indexes from the paths.

    Parameters
    ----------
    paths_to_metadata_indexes : list[str]
        A list of paths to the metadata indexes.

    Returns
    -------
    DataFrame
        A dataframe with all the metadata indexes.
    """
    metadata_df = pd.concat(
        [
            pd.read_parquet(
                path,
                **kwargs,
            )  # Maybe add dtype
            for path in paths_to_metadata_indexes
        ]
    )
    if "columns" in kwargs:
        if "month" not in kwargs["columns"] or "year" not in kwargs["columns"]:
            return metadata_df
    metadata_df.loc[:, "month"] = metadata_df["month"].astype("int16")
    metadata_df.loc[:, "year"] = metadata_df["year"].astype("int16")
    return metadata_df


def read_df_file_type_handler(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """Reads a file based on its extension.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    pd.DataFrame
        The read dataframe.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet"):
        # change columns keyword
        if "usecols" in kwargs:
            kwargs["columns"] = kwargs.get("usecols")
            del kwargs["usecols"]
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def batch(lst: list, n: int) -> Generator[list, None, None]:
    """
    Batch elements of a list by a given n.

    Parameters
    ----------
    lst : list
        The list of elements.
    n : int
        The batch size.

    Yields
    ------
    list
        The batched list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_config() -> dict:
    with resources.path("atlantes.config", "config.yaml") as config_path:
        with open(config_path) as file:
            config = yaml.safe_load(file)
    return config


CONFIG = load_config()["utils"]

LABEL_MAPPING = CONFIG["LABEL_MAPPING"]
ELEVATION_PATH = (
    "data/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo/GEBCO_2022_sub_ice_topo.nc"
)


def load_ais_categories_csv() -> pd.DataFrame:
    with resources.path("atlantes.config", "AIS_categories.csv") as file:
        vessel_category_df = pd.read_csv(file)
    return vessel_category_df


AIS_CATEGORIES = load_ais_categories_csv()
NUM_TO_CATEGORY_DESC = dict(
    zip(AIS_CATEGORIES["category"], AIS_CATEGORIES["category_desc"])
)
VESSEL_TYPES_BIN_DICT = AIS_CATEGORIES.set_index("num", drop=True)["category"].to_dict()


def list_parquet_files(bucket_name: str, blob_name: str, n: int) -> list[storage.Blob]:
    """List n Parquet files from a blob in a bucket.

    USE THE METADATA INDEX INSTEAD OF THIS FUNCTION TO GET THE FILENAMES"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    files = []
    file_len = 0
    for file in bucket.list_blobs(prefix=blob_name):
        if not file.name.endswith(".parquet"):
            continue
        logger.debug(file.name)
        files.append(file)
        file_len += 1
        if file_len == n:
            break

    return files


class GeoPoint(BaseModel):
    """A class to represent a point on the earth's surface"""

    lat: float
    lon: float


@pa.check_types
def load_all_dataset_points(
    paths_to_files: Union[list[str], np.ndarray]
) -> DataFrame[LocationDataModel]:
    """Load all the dataset points."""
    dfcols = ["lat", "lon", "send"]
    logger.info(f"Loading all dataset points {paths_to_files[:10]}")

    with ProgressBar():
        lazy_dfs = [
            dask.delayed(read_df_file_type_handler)(file, usecols=dfcols)
            for file in tqdm(paths_to_files)
        ]
        dfs = dask.compute(*lazy_dfs)

    return pd.concat(dfs)


def write_file_to_bucket(
    output_path: str, bucket_name: str, df: pd.DataFrame, index: bool = True
) -> None:
    """Writes a dataframe to a csv file in a Google Cloud Storage bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(output_path)
    blob.upload_from_string(df.to_csv(index=index), "text/csv")


def plot_and_save_ais_dataset(
    df: pd.DataFrame, output_folder: Path, bucket_name: str, img_dpi: int = 1000
) -> None:
    """Plot the ais dataset. high level view"""
    ax = df.plot(kind="scatter", x="lon", y="lat", s=0.00001)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    contextily.add_basemap(ax, crs=4326, source=contextily.providers.Esri.OceanBasemap)
    os.makedirs(output_folder, exist_ok=True)
    filename = str(output_folder / "ais_dataset_location_distribution.png")
    plt.savefig(filename, dpi=img_dpi)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Create a new blob and upload the file
    logger.info(f"Uploading {filename} to {bucket_name}")
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)


def export_dataset_to_gcp(
    output_dir: str,
    df: pd.DataFrame,
    label_file_name: str,
    gcp_bucket_name_ais_tracks: str,
    plot_png: bool = False,
) -> None:
    """Export the dataset to the GCP bucket."""
    # TODO: move this to a shared location
    output_folder = Path(output_dir)
    # Plot Geographic Distribution
    if plot_png:
        logger.warning(
            "Plotting ENTIRE trajectory for each file in dataset \
                       may take a long time"
        )
        all_unique_raw_paths = np.unique(np.concatenate(df.raw_paths.to_numpy()))
        all_lat_lon_points_df = load_all_dataset_points(all_unique_raw_paths)
        plot_and_save_ais_dataset(
            all_lat_lon_points_df, output_folder, bucket_name=gcp_bucket_name_ais_tracks
        )

    # Write to GCP Bucket
    blob_path = str(output_folder / f"{label_file_name}.csv")
    logger.info(f"Writing {blob_path} to {gcp_bucket_name_ais_tracks}")
    write_file_to_bucket(blob_path, gcp_bucket_name_ais_tracks, df)
    logger.info("Done")


def format_datetime(dt: Union[datetime, str]) -> str:
    """
    Formats a datetime object or a datetime string into the ISO 8601 format.

    Args:
    dt (Union[datetime, str]): The datetime object or string to format.

    Returns:
    str: The formatted datetime string.
    """
    if isinstance(dt, str):
        # Parse the string into a datetime object
        dt = datetime.fromisoformat(dt)

    # Format the datetime into the specified ISO 8601 format
    formatted_datetime = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return formatted_datetime


def get_changepoint_binary(track_df: pd.Series) -> np.ndarray:
    """gets changepoint indices from subpath column"""
    # Check that there is a subpath 0
    if track_df["subpath_num"].iloc[0] != 0:
        raise ValueError(f"First subpath is not 0 in trajectory {track_df}")
    cpoint_binary = track_df["subpath_num"].diff(-1).fillna(-1.0).reset_index(drop=True)
    if len(cpoint_binary) == 0:
        raise ValueError(f"No changepoints in trajectory {track_df}")
    subpath_idxs = cpoint_binary[cpoint_binary <= -1].index.to_numpy()
    return subpath_idxs


def get_flag_code_from_mmsi(mmsi: int) -> str:
    """Get the flag code from the MMSI number.

    Parameters
    ----------
    mmsi : str
        MMSI number.

    Returns
    -------
    str
        Flag code.
    """
    mmsi_country_code = str(mmsi)[:3]
    return MMSI_FLAG_CODES.get(mmsi_country_code, "none")


def plot_data_coords_on_map_dask(data_path: str, event_type: str = "fishing") -> None:
    """
    Must use dask to do this compute (versus pandas) because the data is too big
    and this will still take time
    TODO ensure that this is called on every training/validation loop and dynamically
    plot the data map so that it is clear (as part of CI) where the model is drawing
    annotations from

    """
    ddf = dd.read_csv(list(Path(data_path).rglob("*.csv")), assume_missing=True)
    df = pd.DataFrame.from_dict(
        {
            "latitude": ddf["lat"].values.compute(),
            "longitude": ddf["lon"].values.compute(),
            "label": ddf["label"].values.compute(),
        }
    )
    if event_type == "fishing":
        label_subset = df[df["label"] != 0]
    elif event_type == "none":
        label_subset = df[df["label"] == 0]

    label_subset["label"] = label_subset["label"].replace(LABEL_MAPPING)
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    geometry = [
        Point(xy) for xy in zip(label_subset["longitude"], label_subset["latitude"])
    ]
    gdf = GeoDataFrame(label_subset, geometry=geometry)
    gdf.plot(
        ax=world.plot(figsize=(20, 12)),
        column="label",
        marker="o",
        markersize=6,
        categorical=True,
        legend=True,
    )
    plt.savefig(f"figures/{event_type}-data-global-snapshot.jpg", bbox_inches="tight")


def load_elevation_data() -> Tuple[np.ndarray, dict]:
    """Sea floor height in meters (above mean sea level) bathymetric height

    This data file is large and this file should be loaded once and held in memory

    Returns
    -------
    Tuple[np.ndarray, dict]

    """
    elevation_layer_name = "elevation"
    dataset = nc.Dataset(ELEVATION_PATH)

    layer_data = dataset[elevation_layer_name]
    data_array = layer_data[:].data
    layer_metadata = layer_data.__dict__
    global_metadata = dataset.__dict__
    global_metadata.update(layer_metadata)

    return data_array, global_metadata


def calculate_e2e_cog(
    start_point: GeoPoint, end_point: GeoPoint
) -> Tuple[float, float]:
    """Calculate great circle distance, forward and backward azimuth

    Parameters
    ----------
    start_point : GeoPoint
    end_point : GeoPoint

    Returns
    -------
    Tuple[float, float]
        fwd_azimuth, distance in km
    """
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, _, meters = geodesic.inv(
        start_point.lon, start_point.lat, end_point.lon, end_point.lat
    )

    return meters


def infra_within_threshold(
    infrastructure: gpd.GeoDataFrame,
    point: GeoPoint,
    infra_threshold_meters: float = 1000,
) -> int:
    """determines if there is infrastructure within a threshold distance of a point

    Parameters
    ----------
    point : GeoPoint

    threshold_meters : float, optional
         , by default 1000

    Returns
    -------
    int
        number of platforms within threshold distance of point
    """
    infrastructure = infrastructure[
        (np.abs(infrastructure.lon - point.lon) < 1)
        & (np.abs(infrastructure.lat - point.lat) < 1)
    ]

    infrastructure["geopoint"] = infrastructure.apply(
        lambda x: GeoPoint(lat=x["lat"], lon=x["lon"]), axis=1
    )
    infrastructure["distance"] = infrastructure.apply(
        lambda x: calculate_e2e_cog(
            GeoPoint(lat=point.lat, lon=point.lon), x["geopoint"]
        ),
        axis=1,
    )

    return len(infrastructure[infrastructure["distance"] < infra_threshold_meters])


def print_model_statedict(model_arch: torch.nn.module, state_dict: dict) -> None:
    logger.debug("Model architecture state dict:")
    for param_tensor in model_arch.state_dict():
        logger.info(param_tensor, "\t", model_arch.state_dict()[param_tensor].size())

    logger.debug("Saved weights state dict:")
    for param_tensor in state_dict:
        logger.info(param_tensor, "\t", state_dict[param_tensor].size())


def csv_to_dict(ENTITY_IDENTITY_CSV: str) -> dict:
    """Most likely this code and the lookup code should go outside ATLAS"""
    track_dict = {}

    with open(ENTITY_IDENTITY_CSV, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip the header (optional)
        logger.info(next(csv_reader))

        for row in csv_reader:
            track_id = row[0]
            identity = int(row[1])  # Convert to integer
            track_dict[track_id] = identity

    return track_dict


def load_entity_db(entity_identity_csv: str) -> dict:
    """This is a simple dictionary based lookup for the entity type of a track_id"""
    track_dict = {}

    with open(entity_identity_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip the header (optional)
        logger.info(next(csv_reader))

        for row in csv_reader:
            track_id = row[0]
            identity = int(row[1])  # Convert to integer
            track_dict[track_id] = identity

    return track_dict


def is_directory_empty(path: str) -> bool:
    """Returns True if directory is empty, False otherwise."""
    return not bool(os.listdir(path))


def process_additional_args(args: tuple, config_dict: dict) -> dict:
    """for overriding config key value pairs (this is intended for beaker)"""
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)

            logger.info(f"Processing additional argument: {key}={value}")
            # TODO simplify this logic and the nesting
            if key in config_dict:
                # Attempt to cast to the correct type
                if value == "null":
                    logger.info(f"Updating {key} to None")
                    config_dict[key] = None
                    continue
                try:
                    logger.info(f"Attempting to cast {key} to {type(config_dict[key])}")
                    if type(config_dict[key]) is not type(None):
                        logger.info(f"Updating {key} to {value} of type {type(value)}")
                        if isinstance(config_dict[key], bool):
                            if value not in ["True", "False"]:
                                raise ValueError(
                                    f"Invalid boolean value for {key}: {value}"
                                )
                            config_dict[key] = value == "True"
                        elif isinstance(config_dict[key], list):
                            config_dict[key] = ast.literal_eval(value)
                        else:
                            config_dict[key] = type(config_dict[key])(value)
                        logger.info(f"Casting {key} to {type(config_dict[key])}")
                    else:
                        logger.info(f"Unknown type for {key} using {type(value)}")
                        config_dict[key] = value
                    logger.info(f"Updated {key} to {config_dict[key]}")
                    click.echo(f"Updated {key} to {config_dict[key]}")
                except ValueError:
                    logger.error(f"Invalid value for {key}: {value}")
                    click.echo(f"Invalid value for {key}: {value}")
            else:
                click.echo(f"Unknown key: {key}")
    return config_dict


def generate_subpath_segments(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
    """Generate subpath segments from a dataframe simulating building a real time stream

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with the subpaths.
    Yields
    ------
    pd.DataFrame
        The subpath segment.
    """
    n = df["subpath_num"].max() + 1
    for i in range(n):
        yield df[df["subpath_num"] <= i]


def filter_kwargs(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(func)
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in list(sig.parameters.keys())
        }
        logger.info(f"Filtered kwargs: {filtered_kwargs}")
        logger.info(f"Original kwargs: {kwargs}")
        return func(*args, **filtered_kwargs)

    return wrapper


def get_vessel_type_name(
    vessel_type_num: int, num_to_category_desc: dict[Any, Any]
) -> str:
    """
    Get the vessel type name (category_desc) based on the vessel type number (num).

    """
    return num_to_category_desc.get(vessel_type_num, "Unknown")


def get_nav_status(activity_desc: str) -> int:
    """Gets the nav status num for an Activity TYpe"""
    with resources.path("atlantes.config", "nav_categories.csv") as file:
        nav_df = pd.read_csv(file)
    nav_status_num = nav_df[nav_df["desc"] == activity_desc]["num"].values[0]
    return nav_status_num


def get_commit_hash() -> str:
    """Get the commit hash of the current git repository."""
    try:
        # Attempt to get GIT_COMMIT_HASH from the environment; only catch KeyError
        return os.environ["GIT_COMMIT_HASH"]
    except KeyError:
        try:
            # If GIT_COMMIT_HASH is not set, try to fetch using git/gitpython
            import git

            repo = git.Repo(search_parent_directories=True)
            return repo.git.rev_parse(repo.head.commit.hexsha, short=7)
        except ImportError:
            # Log any errors related to importing the git library
            logger.error("Error importing git", exc_info=True)
            return "no_commit_hash"
        except Exception as e:
            # Log any errors related to accessing the git repo
            logger.error(f"Error getting commit hash from git: {e}", exc_info=True)
            return "no_commit_hash"


def floats_differ_less_than(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Helper function for comparing floats."""
    return abs(a - b) < tolerance


def read_geojson_and_convert_coordinates() -> Tuple[np.ndarray, np.ndarray]:
    """reads the geojson file and converts the coordinates to numpy arrays"""
    if Path("/ais").exists():
        marine_file = Path(
            "/ais/src/atlantes/data/latest_marine_infrastructure.geojson"
        )
    else:
        # Assume local environment, set relative path
        marine_file = Path("src/atlantes/data/latest_marine_infrastructure.geojson")

    with open(marine_file, "r") as f:
        geojson_data = json.load(f)

    # Initialize empty arrays for latitudes and longitudes
    latitudes = []
    longitudes = []

    # Iterate through the features and extract coordinates
    for feature in geojson_data["features"]:
        coordinates = feature["geometry"]["coordinates"]
        longitude, latitude = coordinates  # GeoJSON uses [longitude, latitude] order

        # Append coordinates to the respective arrays
        latitudes.append(latitude)
        longitudes.append(longitude)

    # Convert lists to numpy arrays
    latitudes_array = np.array(latitudes)
    longitudes_array = np.array(longitudes)

    return latitudes_array, longitudes_array


class BaseRegistry:
    """Base class for registries"""

    def __init__(self) -> None:
        self.registry: dict[str, Any] = {}

    def register(self, obj: Any, name: str) -> None:
        """Register an item in the registry"""
        if name in self.registry:
            raise ValueError(f"Item {name} already registered")
        self.registry[name] = obj

    def get(self, name: str) -> Any:
        """Get an item from the registry"""
        if name not in self.registry:
            raise ValueError(f"Item {name} not registered")
        return self.registry[name]

    def keys(self) -> list[str]:
        """Get the keys of the registry"""
        return list(self.registry.keys())
