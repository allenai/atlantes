"""Utility functions for ATLAS-activity and ATLAS-entity



#TODO atlas_utils supports datasets and nets
# Eval utils supports evals
# Training utils supports training
# inference utils should be in general utils?

TODO: We may want to consider recomputing sog and cog when they are unknown if is possible from the position of the vessel.

TODO: Make different tables for different dataset kinds of errors for activity logging
"""

import io
from enum import Enum
from importlib import resources
from typing import Optional, Tuple

import cartopy.crs as ccrs
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
import yaml
from atlantes.atlas.schemas import (EntityClassLabelDataModel,
                                    TrajectoryLengthsDataModel)
from atlantes.datautils import (KNOTS_TO_MPS, MAX_VESSEL_SPEED, UNKNOWN_COG,
                                UNKNOWN_SOG)
from atlantes.log_utils import get_logger
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from pandera.typing import DataFrame
from PIL import Image
from sklearn.metrics import f1_score
from suncalc import get_position
from torch import Tensor

logger = get_logger(__name__)

ATLAS_CONFIG_MODULE = "atlantes.atlas.config"
COLORBLIND_FRIENDLY_COLORS = [
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#F0E442",
    "#CC79A7",
]

# Use the list of colors directly with seaborn
sns.set_palette(sns.color_palette(COLORBLIND_FRIENDLY_COLORS))

# If you need to use the ListedColormap elsewhere in your code
COLORBLIND_FRIENDLY_COLOR_MAP = ListedColormap(COLORBLIND_FRIENDLY_COLORS)
MAX_SOG = 10


class LabelEnum(Enum):
    """Base class for the label Enums"""

    @classmethod
    def to_name_label_dict(cls) -> dict:
        """Convert the Enum to a dictionary"""
        return {k.name.lower(): k.value for k in cls}

    @classmethod
    def to_label_name_dict(cls) -> dict:
        """Convert the Enum to a dictionary"""
        return {k.value: k.name.lower() for k in cls}

    @classmethod
    def to_class_descriptions(cls) -> list[str]:
        """Return the class descriptions"""
        return [k.name.lower() for k in cls]


class AtlasEntityLabelsTraining(LabelEnum):
    """The type of entity (eg vessel, buoy, etc.)."""

    VESSEL = 0
    BUOY = 1


class AtlasEntityLabelsTrainingWithUnknown(LabelEnum):
    """The type of entity (eg vessel, buoy, etc.)."""

    VESSEL = 0
    BUOY = 1
    UNKNOWN = -100


class AtlasActivityLabelsTraining(LabelEnum):
    """The type of activity (eg transiting, fishing, anchored, etc.)."""

    FISHING = 0
    ANCHORED = 1
    MOORED = 2
    TRANSITING = 3


class AtlasActivityLabelsWithUnknown(LabelEnum):
    """The type of activity (eg transiting, fishing, anchored, etc.)."""

    FISHING = 0
    ANCHORED = 1
    MOORED = 2
    TRANSITING = 3
    UNKNOWN = -100


class AtlasEntityVesselTypeLabelClass(LabelEnum):
    """The type of vessel (e.g., fishing, cargo, tanker, etc.).

    These do not correspond to AIS type numbers
      but rather the distirbution of the vessel types in the training dataset,
      with the caveat of having fishing vessels first.
      This enables training on different number of classes dynamically.
    """

    FISHING = 0
    CARGO = 1
    TANKER = 2
    PLEASURE = 3
    SAILING = 4
    TOWING = 5
    PASSENGER = 6
    TUG = 7
    DREDGING = 8
    DIVING = 9
    MILITARY = 10
    HIGH_SPEED = 11
    PILOT_VESSEL = 12
    SEARCH_AND_RESCUE_VESSEL = 13
    INDUSTRIAL = 14
    LAW_ENFORCEMENT = 15
    SPARE = 16
    MEDICAL_TRANSPORT = 17
    NONCOMBATANT = 18


# use file type handler here
@pa.check_types
def read_trajectory_lengths_file(
    trajectory_lengths_file: str,
) -> DataFrame[TrajectoryLengthsDataModel]:
    """Reads the trajectory lengths file"""
    if trajectory_lengths_file.endswith(".parquet"):
        df = pd.read_parquet(trajectory_lengths_file)
    else:
        df = pd.read_csv(trajectory_lengths_file)
    return df


@pa.check_types
def read_entity_label_csv(
    enity_label_file: str,
) -> DataFrame[EntityClassLabelDataModel]:
    """Read in the entity label file"""
    return pd.read_csv(enity_label_file).set_index("Path", drop=True)


def read_trajectory_lengths_file_dask(
    trajectory_lengths_file: str,
) -> DataFrame[TrajectoryLengthsDataModel]:
    """Reads the trajectory lengths file"""
    return dd.read_csv(
        trajectory_lengths_file,
        usecols=["Path", "Length"],
        dtype={"Path": str},
    ).compute()


def read_entity_label_csv_dask(
    enity_label_file: str,
) -> DataFrame[EntityClassLabelDataModel]:
    """Read in the entity label file"""
    return dd.read_csv(enity_label_file).compute().set_index("Path", drop=True)


def load_experimental_yaml_config(config_path: str, key: Optional[str] = None) -> dict:
    """loads the experimental configuration file for the model"""
    with resources.path(ATLAS_CONFIG_MODULE, config_path) as experimental_config_path:
        with open(experimental_config_path, "r") as file:
            config = yaml.safe_load(file)
        if key:
            config = config[key]
    return config


def get_experimental_config_buoy(key: str = "train") -> dict:
    """loads the experimental configuration file for the model"""
    BUOY_CONFIG_FILE = "buoy_experimental_config.yaml"
    return load_experimental_yaml_config(BUOY_CONFIG_FILE, key)


def get_experimental_config_vessel_pretrain(key: str = "train") -> dict:
    """loads the experimental configuration file for the model"""
    VESSEL_PRETRAIN_CONFIG_FILE = "atlas_vessel_type_pretrain.yaml"
    return load_experimental_yaml_config(VESSEL_PRETRAIN_CONFIG_FILE, key)


def get_experimental_config_atlas(key: str = "train") -> dict:
    """loads the experimental configuration file for the model"""
    ATLAS_CONFIG_FILE = "atlas_experimental_config.yaml"
    return load_experimental_yaml_config(ATLAS_CONFIG_FILE, key)


def get_experiment_config_atlas_activity_real_time(key: str = "train") -> dict:
    """loads the experimental configuration file for the model"""
    ATLAS_CONFIG_FILE = "atlas_activity_real_time_config.yaml"
    return load_experimental_yaml_config(ATLAS_CONFIG_FILE, key)


def get_atlas_activity_inference_config() -> dict:
    """loads the inference configuration file for the activity model"""
    ACTIVITY_INFERENCE_CONFIG_FILE = "atlas_activity_inference_config.yaml"
    return load_experimental_yaml_config(ACTIVITY_INFERENCE_CONFIG_FILE)


def get_atlas_entity_inference_config() -> dict:
    """loads the inference configuration file for the entity model"""
    ENTITY_INFERENCE_CONFIG_FILE = "atlas_entity_inference_config.yaml"
    config = load_experimental_yaml_config(ENTITY_INFERENCE_CONFIG_FILE)
    return config


def remove_module_from_state_dict(state_dict: dict) -> dict:
    """removes ".module" from the keys in a state dict so that the model
    can be loaded on a single CPU/GPU with or without data parallelism

    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
        new_state_dict[name] = v
    return new_state_dict


ACTIVITY_DICT_CLASSNAMES_TO_LABELS_W_UNKNOWN = (
    AtlasActivityLabelsWithUnknown.to_name_label_dict()
)

NOT_UNDERWAY_LABELS = [
    ACTIVITY_DICT_CLASSNAMES_TO_LABELS_W_UNKNOWN["anchored"],
    ACTIVITY_DICT_CLASSNAMES_TO_LABELS_W_UNKNOWN["moored"],
]

TRANSITING_LABELS = [
    ACTIVITY_DICT_CLASSNAMES_TO_LABELS_W_UNKNOWN["transiting"],
]


def generate_colors_for_activity_classes(n: int) -> list[str]:
    if n > 10:
        raise ValueError(
            "The number of classes must be less than or equal to 10 because we are using tab10 color map"
        )
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n)]
    return colors


ACTIVITY_COLORS = generate_colors_for_activity_classes(len(AtlasActivityLabelsTraining))
ACTIVITY_CLASS_CMAP = ListedColormap(ACTIVITY_COLORS)
if len(ACTIVITY_COLORS) != len(AtlasActivityLabelsTraining):
    raise ValueError("The number of colors must equal the number of activity labels")

TABLE_COLUMN_NAMES_ATLAS_ACTIVITY = [
    "id",
    "trackId",
    "ShipType",
    "name",
    "file_location",
    "image",
    "Overall F1",
    "Fishing F1",
    "predicted activity sequence",
    "predicted activity seq top score",
    "ground_truth_activity_sequence",
]

# TODO: make seperate columns for each task remove subapth num for activity
# Second TODO: once the subpath_num has been removed from atlas-e, can make only one version of this list.
ATLAS_ACTIVITY_COLUMNS_WITH_META = [
    "lat",
    "lon",
    "send",
    "sog",
    "cog",
    "nav",
    "category",
    "flag_code",
    "trackId",
    "dist2coast",
    "name",
    "mmsi",
]  # columns to load in from a trackfile for ATLAS Dataset

ATLAS_COLUMNS_WITH_META = [
    "lat",
    "lon",
    "send",
    "sog",
    "cog",
    "nav",
    "subpath_num",
    "category",
    "flag_code",
    "trackId",
    "dist2coast",
    "name",
    "mmsi",
]  # columns to load in from a trackfile for ATLAS Dataset

FIG_SIZE_IN_INCHES = 4
FIG_DPI = 150
FIG_SIZE_IN_INCHES_SUBPATH = 8
FIG_DPI_SUBPATH = 150
SOG_COLOR_MAP = "viridis"
MARKER_SIZE = 15
SCATTER_POINT_TRANSPARENCY = 0.7


def haversine_distance(
    lat_diff: np.ndarray, lon_diff: np.ndarray, lat1: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Calculate the haversine distance between two arrays of points in meters

    Parameters
    ----------
    lat_diff : np.ndarray
        The difference in latitude between the two arrays of coordinates
    lon_diff : np.ndarray
        The difference in longitude between the two arrays of coordinates
    lat1 : np.ndarray
        The latitude of the first array of coordinates
    lat2 : np.ndarray
        The latitude of the second array of coordinates

    Returns
    -------
    np.ndarray
        The haversine distance between the two arrays of coordinates in meters"""

    distances = (
        6371
        * 2
        * np.arcsin(
            np.sqrt(
                np.sin(lat_diff / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(lon_diff / 2) ** 2
            )
        )
        / 1000
    )
    return distances


def haversine_distance_radians(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate the haversine distance between two arrays of points in meters.

    Parameters
    ----------
    lat1 : np.ndarray
        The latitude of the first array of coordinates (in degrees).
    lon1 : np.ndarray
        The longitude of the first array of coordinates (in degrees).
    lat2 : np.ndarray
        The latitude of the second array of coordinates (in degrees).
    lon2 : np.ndarray
        The longitude of the second array of coordinates (in degrees).

    Returns
    -------
    np.ndarray
        The haversine distance between the two arrays of coordinates in meters.
    """

    # Earth's radius in meters
    R = 6371000  # meters

    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Differences in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    distances = R * c
    return distances


def preprocess_trackfile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the trackfile of AIS data for Atlas models.

    This function does the following:
    - Drops rows with unknown values for sog and cog
    - Removes rows with obvious incorrect lat/lon
    - Removes rows with obvious incorrect sog
    - Converts sog to meters/second

    Note: This function is used for both the ENTITY and ACTIVITY models.

    Parameters
    ----------
    df : pd.DataFrame
        The trackfile to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed trackfile.
    """

    df = df[~(df["sog"] == UNKNOWN_SOG)]

    df = df[~(df["cog"] == UNKNOWN_COG)]

    # Drop rows with cog outside of 0-360 Potential Idea
    # df = df[~((df["cog"] < 0) | (df["cog"] > 360))]

    # remove nans
    df.dropna(inplace=True)

    # remove rows with obvious incorrect lat/lon
    df = df[~((df["lat"] == 0) & (df["lon"] == 0))]

    # remove rows with obvious incorrect sog we should think about clipping this
    df = df[df["sog"] < MAX_VESSEL_SPEED]

    # convert sog to meters/second
    df["sog"] = (
        df["sog"] * KNOTS_TO_MPS
    )  # distances are meters, time is in seconds -> convert sog to meters/second

    return df


def label_seq_by_subpath_activity(
    subpath_idxs: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Label the points in a trajectory by subpath

    This function makes a sequence of labels for every point given subpath wide labels and subpath indices

    Parameters
    ----------
    subpath_idxs : np.ndarray
        The subpath indices of the trajectory
    labels : np.ndarray
        The labels of the trajectory

    Returns
    -------
    np.ndarray
        The labels for each point in the trajectory
    """

    shifted_subpath_idxs = np.roll(subpath_idxs, 1)
    shifted_subpath_idxs[0] = -1
    subpath_lengths = subpath_idxs - shifted_subpath_idxs
    pointwise_labels = np.repeat(labels, subpath_lengths)
    return pointwise_labels


def normalize_longitudes(longitudes: np.ndarray) -> np.ndarray:
    return (longitudes + 180) % 360 - 180


def check_span_antimeridian(longitudes: np.ndarray) -> bool:
    # Normalize longitudes
    longitudes = normalize_longitudes(longitudes)

    # Find minimum and maximum longitude
    min_longitude = np.min(longitudes)
    max_longitude = np.max(longitudes)

    # Calculate the direct range
    direct_range = max_longitude - min_longitude

    # Check if the direct range is greater than 180 degrees
    if direct_range > 180:
        return True
    else:
        return False


def calculate_normalized_mean_longitude(longitudes: np.ndarray) -> float:
    """Calculate the mean longitude of a trajectory normalized to be within -180 to 180 degrees

    If the longitudes span the antimerridian we use this to calculate the mean longitude
    Parameters
    ----------
    longitudes : np.ndarray
        The longitudes of the trajectory

    Returns
    -------
    float
        The mean longitude of the trajectory normalized to be within -180 to 180 degrees
    """

    # Convert longitudes from degrees to radians
    longitudes_rad = np.radians(longitudes)

    # Calculate mean of cosines and sines of the longitudes
    mean_cos = np.mean(np.cos(longitudes_rad))
    mean_sin = np.mean(np.sin(longitudes_rad))

    # Calculate the arctangent of the average sin and cos, handling two quadrants (atan2)
    mean_longitude_rad = np.arctan2(mean_sin, mean_cos)

    # Convert the mean longitude from radians back to degrees
    mean_longitude_deg = np.degrees(mean_longitude_rad)

    # Normalize the result to be within -180 to 180 degrees
    mean_longitude_deg = (mean_longitude_deg + 180) % 360 - 180

    return mean_longitude_deg


def analyze_longitudes(longitudes: np.ndarray) -> Tuple[float, float, float, float]:
    """Analyze the longitudes of a trajectory to determine the range and center longitude

    Parameters
    ----------
    longitudes : np.ndarray
        The longitudes of the trajectory

    Returns
    -------
    Tuple[float, float, float, float]
        The minimum, maximum, range, and center longitudes of the trajectory
    """
    # Normalize longitudes to be within the range -180 to 180
    longitudes = (longitudes + 180) % 360 - 180

    # Determine if all longitudes are positive or negative
    all_positive = np.all(longitudes >= 0)
    all_negative = np.all(longitudes < 0)

    if all_positive or all_negative:
        # Scenario 1 & 2: All positive or all negative
        min_lon = np.min(longitudes)
        max_lon = np.max(longitudes)
        lon_range = max_lon - min_lon
        center_lon = (min_lon + max_lon) / 2

        return min_lon, max_lon, lon_range, center_lon
    else:
        # Scenario 3: Spans the antimeridian
        positive_longitudes = longitudes[longitudes >= 0]
        negative_longitudes = longitudes[longitudes < 0]
        min_positive = (
            np.min(positive_longitudes) if positive_longitudes.size > 0 else None
        )
        max_negative = (
            np.max(negative_longitudes) if negative_longitudes.size > 0 else None
        )
        # Calculate the range taking into account the antimeridian crossing
        direct_distance = min_positive - max_negative
        wrap_around_distance = 360 - direct_distance

        # The true range across the antimeridian
        lon_range = min(direct_distance, wrap_around_distance)

        center_lon = calculate_normalized_mean_longitude(longitudes)

        return min_positive, max_negative, lon_range, center_lon


def plot_trajectory(
    ax: plt.Axes,
    longitude: np.ndarray,
    latitude: np.ndarray,
    title: str,
    color_type: np.ndarray,
    color_map: ListedColormap,
    **kwargs: dict,
) -> plt.Axes:
    # Need to add a clearer name to these functions I should have reviewed this closer
    min_lon, max_lon, lon_range, normalized_mean_longitude = analyze_longitudes(
        longitude
    )

    if check_span_antimeridian(longitude):
        logger.info("spans antimeridian")
        central_longitude = normalized_mean_longitude
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        central_longitude = np.mean(longitude)
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
    ax.projection = projection
    ax.coastlines()

    ax.projection = projection
    ax.coastlines()

    # Normalize the labels for coloring

    if "label_to_class_name_dict" in kwargs:
        label_to_class_name_dict = kwargs["label_to_class_name_dict"]
        min_label = min(label_to_class_name_dict.keys())
        max_label = max(label_to_class_name_dict.keys()) + 1
        normalize = Normalize(vmin=min_label, vmax=max_label)
    else:
        normalize = Normalize(vmin=0, vmax=MAX_SOG)

    scatter = ax.scatter(
        longitude,
        latitude,
        c=color_type,
        cmap=color_map,
        norm=normalize,
        s=MARKER_SIZE,
        alpha=SCATTER_POINT_TRANSPARENCY,
        transform=ccrs.PlateCarree(),
        marker=".",
    )
    last_latitude = [latitude[-1]]
    last_longitude = [longitude[-1]]
    ax.scatter(
        last_longitude,
        last_latitude,
        color="pink",
        transform=ccrs.PlateCarree(),
        s=30,
        marker="*",
        edgecolor="pink",
        linewidth=2,
        zorder=200000000000,
        alpha=1,
    )

    ax.gridlines(draw_labels=True)
    lat_range = np.ptp(latitude)
    max_range = max(lon_range, lat_range) / 2 + 1

    extent = [
        max(central_longitude - max_range, -179.999),
        min(central_longitude + max_range, 179.999),
        max(np.min(latitude) - max_range, -89.999),
        min(np.max(latitude) + max_range, 89.999),
    ]
    # we need to put bounds on this
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add the legends and labels
    if "Activity" not in title:
        colorbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.2)
        colorbar.set_label(
            "Speed over ground (meters/second)"
        )  # confirm units with Henry

    else:  # legend is specific to plot type

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=color,
                markersize=6,
            )
            for label, color in zip(
                label_to_class_name_dict.values(), COLORBLIND_FRIENDLY_COLORS
            )
        ]

        ax.legend(
            handles=legend_handles,
            title="Activities",
            loc="upper left",
            fontsize="small",
        )
    ax.set_title(title)

    return ax


def log_val_predictions_subpath_activity(
    trajectories: np.ndarray,
    seq_lengths: list[int],
    subpath_idxs_lst: list[np.ndarray],
    entity_names: list[str],
    trackIds: list[str],
    file_locations: list[str],
    binned_ship_types: list[str],
    log_labels: np.ndarray,
    probs: np.ndarray,
    log_preds: np.ndarray,
    traj_table: wandb.Table,
    log_counter: int,
    label_to_class_name_dict: dict,
) -> wandb.Table:
    """Log val predictions as images with subpath labeling to a wandb table

    Table has the following columns: table_column_names = [
            "id",
            "trackId",
            "ShipType",
            "name",
            "file_location",
            "image",
            "Overall F1",
            "Fishing F1",
            "predicted activity sequence"
             "pred_activity_seq_top score",
            "ground_truth_activity_sequence",
        ]

    Parameters
    ----------
    trajectories : np.ndarray
        The trajectories of the entities
    seq_lengths : list[int]
        The lengths of the trajectories
    subpath_idxs_lst : list[np.ndarray]
        The subpath indices of the trajectories
    entity_names : list[str]
        The names of the entities
    trackIds : list[str]
        The trackIds of the entities
    file_locations : list[str]
        The file locations of the entities
    log_labels : np.ndarray
        The ground truth labels
    outputs : Tensor
        The outputs of the model
    log_preds : np.ndarray
        The predictions of the model
    traj_table : wandb.Table
        The table to populate with correct predictions
    log_counter : int
        The number of times the model has been evaluated
    label_to_class_name_dict : dict
        The human readable class names
    """

    # adding ids based on the order of the images
    sns.set_theme(style="whitegrid")
    log_scores = []
    _id = 0
    for (
        img,
        seq_length,
        subpath_idxs,
        entity_name,
        trackId,
        file_path,
        binned_ship_type,
        label,
        prediction,
        probs,
    ) in zip(
        trajectories,
        seq_lengths,
        subpath_idxs_lst,
        entity_names,
        trackIds,
        file_locations,
        binned_ship_types,
        log_labels,
        log_preds,
        probs,
    ):

        # Convert tensor to numpy if it's a tensor and you want to use NumPy's functions
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()  # Ensure it's on CPU and convert to NumPy
        else:
            img_np = img

        longitude = img_np[:seq_length, 1]
        latitude = img_np[:seq_length, 0]

        # obtain confidence scores for all classes
        log_scores.append(probs)
        # Want single Image with 2 subplots

        # Create figure with two subplots showing gt/pred side by side.
        fig, (ax_gt, ax_pred) = plt.subplots(
            1,
            2,
            figsize=(FIG_SIZE_IN_INCHES_SUBPATH * 2, FIG_SIZE_IN_INCHES_SUBPATH),
            dpi=FIG_DPI,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        label_color_type = label_seq_by_subpath_activity(subpath_idxs, label)
        pred_color_type = label_seq_by_subpath_activity(subpath_idxs, prediction)
        color_map = COLORBLIND_FRIENDLY_COLOR_MAP
        ax_gt = plot_trajectory(
            ax_gt,
            longitude,
            latitude,
            "Human Annotated Activity Classifications",
            label_color_type,
            color_map,
            label_to_class_name_dict=label_to_class_name_dict,
        )
        ax_pred = plot_trajectory(
            ax_pred,
            longitude,
            latitude,
            "Predicted Activity Classifications",
            pred_color_type,
            color_map,
            label_to_class_name_dict=label_to_class_name_dict,
        )

        fig.tight_layout()
        plt.tight_layout()

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)
        plt.close()

        overall_f1 = f1_score(label, prediction, average="macro")
        # Use the fishing class as the positive class and everything else as other
        # Change prediction to a binary vector where 1 is fishing and 0 is other
        class_name_to_label_dict = {v: k for k, v in label_to_class_name_dict.items()}

        fishing_label = class_name_to_label_dict["fishing"]
        fishing_label_array = np.where(label == fishing_label, 1, 0)
        fishing_pred_array = np.where(prediction == fishing_label, 1, 0)
        fishing_f1 = f1_score(
            fishing_label_array, fishing_pred_array, zero_division=np.nan
        )
        img_id = str(_id) + "_" + str(log_counter)
        # STORE THE FOLLowing as single strings no problem
        pred_activitiy_seq = str(
            [label_to_class_name_dict[i.item()] for i in prediction]
        )

        pred_activity_scores = str(list(log_scores[_id].max(axis=1)))
        gt_activity_seq = str([label_to_class_name_dict[i.item()] for i in label])

        traj_table.add_data(
            img_id,
            trackId,
            binned_ship_type,
            entity_name,
            file_path,
            wandb.Image(image, "caption"),
            overall_f1,
            fishing_f1,
            pred_activitiy_seq,
            pred_activity_scores,
            gt_activity_seq,
        )

    return traj_table


def log_val_predictions_entity_type(
    trajectories: np.ndarray,
    seq_lengths: list[int],
    entity_names: list[str],
    trackIds: list[str],
    file_locations: list[str],
    log_labels: np.ndarray,
    outputs: Tensor,
    log_preds: np.ndarray,
    correct_outputs: wandb.Table,
    incorrect_outputs: wandb.Table,
    log_counter: int,
    human_readable_class_names: dict,
) -> Tuple[wandb.Table, wandb.Table]:
    """Populate the a table with images and their gt/predictions

    Note that while most of the functianality in this function matches
    log_val_predictions_activity, there are some unique features and for that reason
    is a seaparate function

    Parameters
    ----------
    trajectories : np.ndarray
        The trajectories of the entities
    seq_lengths : list[int]
        The lengths of the trajectories
    entity_names : list[str]
        The names of the entities
    trackIds : list[str]
        The trackIds of the entities
    file_locations : list[str]
        The file locations of the entities
    log_labels : np.ndarray
        The ground truth labels
    outputs : Tensor
        The outputs of the model
    log_preds : np.ndarray
        The predictions of the model
    correct_outputs : wandb.Table
        The table to populate with correct predictions
    incorrect_outputs : wandb.Table
        The table to populate with incorrect predictions
    log_counter : int
        The number of times the model has been evaluated
    human_readable_class_names : dict
        The human readable class names

    Returns
    -------
    correct_outputs : wandb.Table
        The table to populate with correct predictions
    incorrect_outputs : wandb.Table
        The table to populate with incorrect predictions


    """
    # obtain confidence scores for all classes
    sns.set_theme(style="whitegrid")
    log_scores = F.softmax(outputs.data, dim=1)

    _id = 0
    for (
        img,
        seq_length,
        entity_name,
        trackId,
        file_path,
        label,
        prediction,
        scores,
    ) in zip(
        trajectories,
        seq_lengths,
        entity_names,
        trackIds,
        file_locations,
        log_labels,
        log_preds,
        log_scores,
    ):

        # Create figure with two subplots showing gt/pred side by side.
        fig, ax_gt = plt.subplots(
            1,
            1,
            figsize=(FIG_SIZE_IN_INCHES_SUBPATH, FIG_SIZE_IN_INCHES_SUBPATH),
            dpi=FIG_DPI,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        latitude = img[:seq_length, 0].numpy()
        longitude = img[:seq_length, 1].numpy()
        sog = img[:seq_length, 2].numpy()

        ax_gt = plot_trajectory(ax_gt, longitude, latitude, "", sog, SOG_COLOR_MAP)

        # Convert matplotlib figure to in-memory file
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)

        img_id = str(_id) + "_" + str(log_counter)
        if prediction.item() != label.item():
            incorrect_outputs.add_data(
                img_id,
                trackId,
                entity_name,
                file_path,
                wandb.Image(image, "caption"),
                human_readable_class_names[prediction.item()],
                human_readable_class_names[label.item()],
                *scores,
            )
        elif prediction.item() == label.item():
            correct_outputs.add_data(
                img_id,
                trackId,
                entity_name,
                file_path,
                wandb.Image(image, "caption"),
                human_readable_class_names[prediction.item()],
                human_readable_class_names[label.item()],
                *scores,
            )
        plt.clf()
        plt.close()
    return correct_outputs, incorrect_outputs


def log_val_predictions_end_of_sequence_task(
    img: np.ndarray,
    file_locations: list[str],
    send_time: pd.Timestamp,
    label: int,
    pred: int,
    probs: float,
    track_length: int,
    output_table: wandb.Table,
    log_counter: int,
    human_readable_class_names: dict,
) -> Tuple[wandb.Table, wandb.Table]:
    """Populate the a table with images and their gt/predictio"""
    # obtain confidence scores for all classes
    sns.set_theme(style="whitegrid")
    _id = 0

    # Create figure with two subplots showing gt/pred side by side.
    fig, ax_gt = plt.subplots(
        1,
        1,
        figsize=(FIG_SIZE_IN_INCHES_SUBPATH, FIG_SIZE_IN_INCHES_SUBPATH),
        dpi=FIG_DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    latitude = img[:track_length, 0]
    longitude = img[:track_length, 1]
    sog = img[:track_length, 2]
    ax_gt = plot_trajectory(
        ax_gt, longitude[:-1], latitude[:-1], "", sog[:-1], SOG_COLOR_MAP
    )
    # Convert matplotlib figure to in-memory file
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    image = Image.open(buf)
    img_id = str(_id) + "_" + str(log_counter)
    output_table.add_data(
        img_id,
        str(file_locations),
        send_time.strftime("%Y-%m-%d %H:%M:%S"),
        wandb.Image(image, "caption"),
        human_readable_class_names[label],
        human_readable_class_names[pred],
        str(probs),
    )

    plt.clf()
    plt.close()
    return output_table


def compute_solar_altitude(
    latitudes: np.ndarray, longitudes: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """Calculate the solar altitude using suncalc (https://pypi.org/project/suncalc/).

    Args:
        latitudes (np.ndarray): Array of latitudes of the locations.
        longitudes (np.ndarray): Array of longitudes of the locations.
        timestamps (np.ndarray): Array of UTC timestamps to evaluate.

    Returns:
        np.ndarray: Array of solar altitudes in radians (-1.5708 to 1.5708).
    """
    res = pd.DataFrame(get_position(timestamps, longitudes, latitudes))
    altitudes = res["altitude"].to_numpy()

    return altitudes
