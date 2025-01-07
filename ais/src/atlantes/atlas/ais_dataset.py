""" AIS Trajectory Dataset

This module contains the dataset class for AIS Trajectory Entity and Activity Classification

# TODO: make min subpath messages for online configurable and remove from init and decide what to do for online
# TODO expose human readbale metadata for ship type i.e fishing , unknown, not fishing
# TODO: # add tests for get item and online mode as well as testing inference error catching
# TODO: Add tests for inference on empty data
# Remove unneeded statefulness from the class
"""

import ast
import logging
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Any, Literal, Optional, Union, get_args

import numpy as np
import pandas as pd
import pandera as pa
import torch
from atlantes.atlas.atlas_utils import (ATLAS_ACTIVITY_COLUMNS_WITH_META,
                                        ATLAS_COLUMNS_WITH_META,
                                        AtlasActivityLabelsTraining,
                                        AtlasEntityLabelsTraining,
                                        compute_solar_altitude,
                                        haversine_distance,
                                        preprocess_trackfile,
                                        read_trajectory_lengths_file)
from atlantes.atlas.augmentations import \
    augmentation_registry  # Probably should make this into a basic class so its better documented
from atlantes.atlas.schemas import (ActivityEndOfSequenceLabelDataModel,
                                    EntityClassLabelDataModel,
                                    TrackfileDataModelTrain)
from atlantes.datautils import DATE_FORMAT
from atlantes.log_utils import get_logger
from atlantes.utils import VESSEL_TYPES_BIN_DICT
from pandera.typing import DataFrame
from torch.utils.data import Dataset

logger = get_logger(__name__)
logging.getLogger("gcsfs").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

NUM_ROWS_TO_VALIDATE_FOR_LABEL_DF = 5
logger.info("Setting warnings to ignore FutureWarnings")
warnings.filterwarnings("ignore", category=FutureWarning)


MODETYPES = Literal["train", "eval", "online"]


class AISTrajectoryBaseDataset(Dataset):
    """Base Dataset for AIS Trajectory Model"""

    def __init__(
        self,
        dataset_config: dict,
        mode: MODETYPES = "train",
    ):
        """Dataset for AIS Trajectory Classification

        Both the Dataset Classes for Activity and Entity Classification inherit from this class

        Parameters
        ----------
        dataset_config : dict
            Configuration for the dataset,
        mode:
            mode of dataset, by default "train" but can be "eval" or "online"

        Raises
        ------
        ValueError
            If there are no trajectories in the dataset
        ValueError
            If there are no trajectories that meet the minimum number of AIS messages
        ValueError
            If kernel size is not odd

        """
        self.min_ais_messages = dataset_config["MIN_AIS_MESSAGES"]
        self.max_trajectory_length = dataset_config["MAX_TRAJECTORY_LENGTH"]
        self.n_total_trajectories = dataset_config.get("N_TOTAL_TRAJECTORIES")
        self.trajectory_lengths_file = dataset_config.get("TRAJECTORY_LENGTHS_FILE")
        self._mode = mode
        self.dataset_config = dataset_config
        self.cpe_kernel_size = dataset_config["CPE_KERNEL_SIZE"]
        self._validate_kernel_size()
        self.augmentation_config = dataset_config.get("AUGMENTATION", [])
        self.input_feature_keys: list = []
        self.norm_config: dict = {}

    def _validate_kernel_size(self) -> None:
        """Validates the CPE kernel size."""
        if self.cpe_kernel_size is None:
            raise ValueError("CPE kernel size cannot be None")
        if self.cpe_kernel_size % 2 == 0:
            logger.error(f"CPE kernel size is {self.cpe_kernel_size}")
            raise ValueError("CPE kernel size must be odd")

    def _validate_normalization_config(self) -> None:
        """Validates the feature normalization configuration."""
        if not self.norm_config:
            raise ValueError(
                "Missing FEATURE_NORMALIZATION_CONFIG_ACTIVITY in the dataset configuration."
            )
        if not self.input_feature_keys:
            raise ValueError(
                "Missing MODEL_INPUT_COLUMNS_ACTIVITY in the dataset configuration."
            )
        if not all(
            feat in list(self.norm_config.keys()) for feat in self.input_feature_keys
        ):
            raise ValueError(
                "Missing columns in FEATURE_NORMALIZATION_CONFIG_ACTIVITY."
            )

    @property
    def is_train(self) -> bool:
        return self._mode == "train"

    @property
    def is_eval(self) -> bool:
        return self._mode == "eval"

    @property
    def is_online(self) -> bool:
        return self._mode == "online"

    @property
    def mode(self) -> str:
        """Getter for the mode attribute."""
        return self._mode

    @mode.setter
    def mode(self, value: MODETYPES) -> None:
        """Setter for the mode attribute with validation."""
        if value not in get_args(MODETYPES):
            raise ValueError(f"Mode must be one of {MODETYPES}.")
        logger.info(f"Setting mode to {value}")
        self._mode = value

    def apply_augmentations(
        self, trajectory: DataFrame[TrackfileDataModelTrain]
    ) -> DataFrame[TrackfileDataModelTrain]:
        """Applies augmentations to the trajectory data

        Augmentations are applied in the order they are defined in the configuration.

        Parameters
        ----------
        trajectory : DataFrame[TrackfileDataModelTrain]
            The trajectory data to augment

        Returns
        -------
        DataFrame[TrackfileDataModelTrain]
            The augmented trajectory data
        """
        augmentations = self.augmentation_config["augmentations"]
        applicable_augmentations = [aug for aug in augmentations if aug["apply"]]
        if not applicable_augmentations:
            return trajectory

        mode: Literal["compose", "random_choice"] = self.augmentation_config.get(
            "mode", "compose"
        )
        if mode == "compose":
            for augmentation in applicable_augmentations:
                trajectory = self._apply_single_augmentation(trajectory, augmentation)

        elif mode == "random_choice":
            random_augmentation = np.random.choice(applicable_augmentations)
            trajectory = self._apply_single_augmentation(
                trajectory, random_augmentation
            )
        else:
            raise ValueError(f"Invalid augmentation mode: {mode}")
        return trajectory

    def _apply_single_augmentation(
        self, trajectory: DataFrame[TrackfileDataModelTrain], augmentation: dict
    ) -> DataFrame[TrackfileDataModelTrain]:
        """Applies a single augmentation to the trajectory data


        If the augmentation fails, the original trajectory data is returned.
        Parameters
        ----------
        trajectory : DataFrame[TrackfileDataModelTrain]
            The trajectory data to augment
        augmentation : dict
            The augmentation to apply

        Returns
        -------
        DataFrame[TrackfileDataModelTrain]
            The augmented trajectory data
        """
        name = augmentation["name"]
        try:
            augmentation_func = augmentation_registry.get_augmentation(name)
        except ValueError as e:
            logger.error(
                f"Error getting augmentation {name}: {e} aug not applied", exc_info=True
            )
            return trajectory
        params = augmentation.get("params", {})
        try:
            output_trajectory = augmentation_func(trajectory, **params)
        except Exception as e:
            logger.error(
                f"Error applying augmentation {name}: {e} aug not applied",
                exc_info=True,
            )
            output_trajectory = trajectory
        return output_trajectory

    def _filter_short_trajectories(self) -> np.ndarray:
        """Filter out trajectories that would be too short for kernel and signal

        Returns
        -------
        np.ndarray
            Array of paths to trajectories that are long enough"""
        df_lengths = read_trajectory_lengths_file(
            self.trajectory_lengths_file,
        )
        # Filter out trajectories that are too short
        df_long_enough = df_lengths[df_lengths["Length"] >= self.min_ais_messages]
        logger.info(f"Found {len(df_long_enough)} long enoughtracks ")

        return df_long_enough["Path"].to_numpy(str)

    @abstractmethod
    def _initialize_target_classes(self) -> None:
        """initializes target classes, method should be changed for different tasks"""
        pass

    def __len__(self) -> int:
        """returns length of dataset

        Returns
        -------
        int

        """

        return len(self.trajectories)

    def _get_spatial_intervals(
        self, df: pd.DataFrame, kernel_size: int
    ) -> torch.tensor:
        """returns tensor of delta distance in meters

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of trajectory
        kernel_size : int
            kernel size for continuous point embedding

        Returns
        -------
        torch.tensor
            tensor of delta distance in meters with shape (len(df), kernel_size)"""
        latitudes = np.radians(df["lat"].values)
        longitudes = np.radians(df["lon"].values)
        # pad lat and long by .5 kernel size rounded down
        latitudes_pad = np.pad(
            latitudes, (int(kernel_size / 2), int(kernel_size / 2)), mode="edge"
        )
        longitudes_pad = np.pad(
            longitudes, (int(kernel_size / 2), int(kernel_size / 2)), mode="edge"
        )
        # Make an array of lat and lons that has the kernel around each point as a row
        latitudes_kernel = np.lib.stride_tricks.sliding_window_view(
            latitudes_pad, kernel_size
        )
        longitudes_kernel = np.lib.stride_tricks.sliding_window_view(
            longitudes_pad, kernel_size
        )
        # get the diffs betwen each point in the kernel and the center
        lat_diff = latitudes_kernel - latitudes[:, np.newaxis]
        lon_diff = longitudes_kernel - longitudes[:, np.newaxis]
        # Apply Haversine formula to calculate distances
        distances = haversine_distance(
            lat_diff, lon_diff, latitudes[:, np.newaxis], latitudes_kernel
        )  # in meters
        return distances

    def _get_time_intervals(self, df: pd.DataFrame, kernel_size: int) -> torch.tensor:
        """gets a tensor of time intervals in seconds between each message within a kernel

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of trajectory
        kernel_size : int
            kernel size for continuous point embedding

        Returns
        -------
        torch.tensor
            tensor of time intervals in seconds with shape (len(df), kernel_size)
        """
        times = pd.to_datetime(
            df["send"], format=self.dataset_config.get("DATE_FORMAT", DATE_FORMAT)
        )

        # Convert times_df to Unix timestamp
        timestamps = times.apply(datetime.timestamp).to_numpy()

        # Timestamps padded
        timestamps_pad = np.pad(
            timestamps, (int(kernel_size / 2), int(kernel_size / 2)), mode="edge"
        )
        # Create kernel view of timestamps
        timestamps_kernel = np.lib.stride_tricks.sliding_window_view(
            timestamps_pad, kernel_size
        )
        # Calculate time deltas for each point within the kernel
        time_deltas = timestamps_kernel - np.expand_dims(timestamps, -1)
        return time_deltas

    @pa.check_types
    def _preprocess(
        self, df: DataFrame[TrackfileDataModelTrain]
    ) -> DataFrame[TrackfileDataModelTrain]:
        """Pre-process AIS data for use in Modeling

        Validates the dataframe using the TrackfileDataModelTrain schema
        This function is implemented in utils so there is a single source of truth for the preprocessing done on the data

        Parameters
        ----------
        df : DataFrame[TrackfileDataModelTrain]
            dataframe of AIS trajectory


        Returns
        -------
        df: DataFrame[TrackfileDataModelTrain]
            preprocessed dataframe
        """
        return preprocess_trackfile(df)

    def _calculate_relative_cog(self, cogs: pd.Series) -> np.ndarray:
        """Calculates the relative change in course over ground

        Parameters
        ----------
        df : pd.Series
            dataframe of trajectory

        Returns
        -------
        np.ndarray
            relative change in course over ground
        """

        def dif_between_angles(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
            """calculates the unsigned difference between two angles
            that lie between 0 and 360 degrees"""
            return 180 - np.abs(180 - (np.abs(a1 - a2)))

        rel_cog = np.roll(
            dif_between_angles(cogs.shift(-1).fillna(0).to_numpy(), cogs.to_numpy()), 1
        )
        rel_cog[0] = 0
        # rel_cog - np.log(1+ rel_cog)
        return rel_cog

    def _add_amount_of_light_feature(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """Adds the amount of light feature to the trajectory"""
        trajectory.insert(
            loc=2,  # this index is not relied on.
            column="amount_of_light",
            value=compute_solar_altitude(
                trajectory["lat"],
                trajectory["lon"],
                trajectory["send"],
            ),
        )
        return trajectory

    def _add_rel_cog_feature(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        trajectory.insert(
            loc=2,  # this index is not relied on.
            column="rel_cog",
            value=self._calculate_relative_cog(trajectory["cog"]),
        )
        return trajectory

    def _normalize_features(
        self, df: pd.DataFrame, feature_config: dict
    ) -> pd.DataFrame:
        """
        Normalize features in a pandas DataFrame and convert it to a PyTorch tensor.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to normalize
        feature_config : dict
            A dictionary with the following structure:
            {
                "feature_1": {"sqrt": False, "log": False, "z-scale": True},
                "feature_2": {"sqrt": True, "log": False, "z-scale": False},
                ...
            }
            Each key is a feature name, and the corresponding value is a dictionary with three boolean keys:
            "sqrt", "log", and "z-scale". If "sqrt" is True, the square root transformation is applied to the feature.
            If "log" is True, the natural logarithm transformation is applied to the feature. If "z-scale" is True,
            the feature is standardized using the z-score.

        Returns
        -------
        pd.DataFrame
            The normalized DataFrame.

        Raises
        ------
        ValueError
            If a square root transformation is attempted on negative values.
        """
        # Make a copy of the DataFrame to avoid modifying the original
        df_normalized = df.copy()

        # Apply transformations and standardization to each feature
        for feature, config in feature_config.items():
            # Apply square root transformation if specified
            if config["sqrt"]:
                if np.any(df_normalized[feature] < 0):
                    raise ValueError(
                        f"Cannot apply square root transformation to negative values in feature {feature}."
                    )
                df_normalized.loc[:, feature] = np.sqrt(
                    df_normalized[feature].to_numpy()
                )
            # Apply log transformation if specified
            elif config["log"]:
                if np.any(df_normalized[feature] < 0):
                    raise ValueError(
                        f"Cannot apply log transformation to negative values in feature {feature}."
                    )
                df_normalized.loc[:, feature] = np.log1p(
                    df_normalized[feature].to_numpy()
                )
            elif config["z-scale"]:
                # Standardize the feature
                mean = df_normalized[feature].mean()
                std = df_normalized[feature].std()
                df_normalized.loc[:, feature] = df_normalized[feature].apply(
                    lambda x: (x - mean) / std
                )
        return df_normalized

    def _load_track_parquet(
        self, trajectory_path: str, usecols: list[str] = ATLAS_COLUMNS_WITH_META
    ) -> pd.DataFrame:
        """Loads a track parquet from a path
        Parameters
        ----------
        trajectory_path : str
            path to trajectory csv

        Returns
        -------
        pd.DataFrame
            dataframe of trajectory
        """
        return pd.read_parquet(
            trajectory_path,
            engine="pyarrow",
            columns=usecols,
        )

    def _load_track_csv(
        self, trajectory_path: str, usecols: list[str] = ATLAS_COLUMNS_WITH_META
    ) -> pd.DataFrame:
        """Loads a track csv from a path
        Parameters
        ----------
        trajectory_path : str
            path to trajectory csv

        Returns
        -------
        pd.DataFrame
            dataframe of trajectory
        """
        return pd.read_csv(
            trajectory_path,
            usecols=usecols,
            parse_dates=["send"],
            date_format=self.dataset_config.get("DATE_FORMAT", DATE_FORMAT),
        )

    def _load_trackfile(
        self, trajectory_path: str, usecols: list[str] = ATLAS_COLUMNS_WITH_META
    ) -> pd.DataFrame:
        """Loads a track file from a path
        Parameters
        ----------
        trajectory_path : str
            path to trajectory csv

        Returns
        -------
        pd.DataFrame
            dataframe of trajectory
        """
        if trajectory_path.endswith(".parquet"):
            return self._load_track_parquet(trajectory_path, usecols)
        elif trajectory_path.endswith(".csv"):
            return self._load_track_csv(trajectory_path, usecols)
        else:
            raise NotImplementedError(
                f"Trajectory file type {trajectory_path} not supported"
            )

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """gets item from dataset"""
        pass


class ActivityDatasetEndOfSequence(AISTrajectoryBaseDataset):

    def __init__(
        self,
        dataset_config: dict,
        activity_label_file_paths: Optional[list[str]] = None,
        mode: MODETYPES = "train",
        in_memory_data: Optional[list[pd.DataFrame]] = None,
        online_file_paths: Optional[list[str]] = None,
        label_enum: Any = AtlasActivityLabelsTraining,
    ) -> None:
        super().__init__(
            mode=mode,
            dataset_config=dataset_config,
        )
        """Dataset for AIS Trajectory Subpath Activity Classification

        If online, we can use the in_memory_data or the online_file_paths
        In memory data is a list of dataframes, online_file_paths is a list of file paths
        We use file paths for lazy loading for batch inference on unlabeled data
        During Training the
        Parameters
        ----------
        dataset_config : dict
            Configuration for the dataset
        activity_label_file_paths : list[str]
            List of paths to csv files with subpath labels
        mode: str, optional
            mode of dataset, by default "train" but can be "eval" or "online" does not
            reinitialize the dataset
        in_memory_data: pd.DataFrame, optional
            if provided, will use this dataframe instead of reading from disk
            in order to use the same class during training and inference
        online_file_paths: list[str], optional
            List of file paths to use in online mode for simulating batches where we have no labels

        """
        # Type depends on if online or not
        self.label_enum = label_enum
        self.max_context_length = self.max_trajectory_length
        self.min_context_length = self.min_ais_messages
        self.trajectories: Union[list[str], list[pd.DataFrame]]
        self.norm_config = self.dataset_config["FEATURE_NORMALIZATION_CONFIG_ACTIVITY"]
        self.input_feature_keys = self.dataset_config["MODEL_INPUT_COLUMNS_ACTIVITY"]
        self.activity_label_file_paths = activity_label_file_paths
        self.in_memory_data = in_memory_data
        self.online_file_paths = online_file_paths
        self._validate_normalization_config()

        if self.is_online:
            self._init_online_mode()
        elif self.is_train:
            self._init_train_mode()
        else:
            self._init_eval_mode()

    def _init_eval_mode(self) -> None:
        """Initializes the dataset for eval mode.

        Not clear if this mode should exist or what it should look like
        we care about the data source adn whether we want to have labels or not"""
        raise NotImplementedError("Eval mode not implemented")

    def _init_online_mode(self) -> None:
        """Initializes the dataset for online mode."""
        if self.in_memory_data is not None and self.online_file_paths is not None:
            raise ValueError(
                "We can only use one of in_memory_data or online_file_paths at a time"
            )
        if self.in_memory_data is None and self.online_file_paths is not None:
            self.trajectories = self.online_file_paths
            self.use_paths = True
            return
        self.trajectories = (
            self.in_memory_data
            if isinstance(self.in_memory_data, list)
            else [self.in_memory_data]
        )
        self.use_paths = False
        return

    def _init_train_mode(self) -> None:
        """Initializes the dataset for train mode."""
        self._validate_args_for_train()
        # Indexed by File Name not Track ID because track ID is split on month
        activity_labels_df = self._load_target_classes()

        logger.info(f"Filtered to {len(activity_labels_df)} tracks")
        num_samples_to_use = (
            self.n_total_trajectories if self.n_total_trajectories is not None else 0
        )

        if len(activity_labels_df) > num_samples_to_use:
            activity_labels_df = activity_labels_df.sample(
                n=num_samples_to_use, random_state=self.dataset_config["RANDOM_STATE"]
            )
        else:
            # SHuffle the data
            activity_labels_df = activity_labels_df.sample(
                frac=1, random_state=self.dataset_config["RANDOM_STATE"]
            )
        if self.dataset_config["USE_WEKA"]:
            # add logging confirmation
            logger.info("Using Weka")
            activity_labels_df["raw_paths"] = activity_labels_df["raw_paths"].apply(
                lambda x: [path.replace("gs://ais-track-data", "/data") for path in x]
            )
            logger.info(activity_labels_df["raw_paths"].head())

        # Init targets for each sample
        self.class_descriptions: dict[int, str] = self.label_enum.to_label_name_dict()

        name_to_label_dict = self.label_enum.to_name_label_dict()
        logger.info(activity_labels_df["activity_type_name"].value_counts())
        self.activity_class_targets = activity_labels_df.activity_type_name.apply(
            lambda x: name_to_label_dict[x]
        ).to_numpy()
        self.track_ids = activity_labels_df["trackId"].to_numpy()

        self.track_id_groups = pd.factorize(activity_labels_df["trackId"])[0]

        self.trajectory_send_time_pairs = activity_labels_df[
            ["raw_paths", "send"]
        ].values.tolist()
        self.trajectories = activity_labels_df["raw_paths"].to_numpy()
        self.dataset_membership = activity_labels_df["dataset_name"].to_numpy()
        self.random_state_numpy = np.random.RandomState(
            self.dataset_config["RANDOM_STATE"]
        )

    def _validate_args_for_train(self) -> None:
        """Validates the arguments for the dataset when not in online mode."""
        if not (
            self.cpe_kernel_size is not None
            and self.activity_label_file_paths is not None
            and self.n_total_trajectories is not None
            and self.max_trajectory_length is not None
        ):
            raise ValueError(
                "Must provide cpe_kernel_size, activity_label_file_paths, n_total_trajectories, and max_trajectory_length"
            )
        if self.in_memory_data is not None:
            raise ValueError("Cannot use in_memory_data in train mode")
        if self.online_file_paths is not None:
            raise ValueError("Cannot use online_file_paths in train mode")

    def __len__(self) -> int:
        """returns length of dataset"""
        return len(self.trajectories)

    def _load_activity_label_file(self, path: Optional[str]) -> pd.DataFrame:
        """Loads activity label file from a path and converts the raw_paths column to a list of strings"""
        logger.info(f"_load_activity_label_file: {path}")
        if path is None:
            raise ValueError("Path to activity label file is required")
        return pd.read_csv(
            path,
            converters={"raw_paths": ast.literal_eval},
        )

    def _get_dataset_name_list(self, list_of_paths: list[str]) -> list[str]:
        """gets the dataset name from the path"""
        path_component_idx = -1

        dataset_names = [
            path.split("/")[path_component_idx].split(".")[0] for path in list_of_paths
        ]
        # ensure names are unique otherwise use more of the path
        while len(set(dataset_names)) != len(dataset_names):
            path_component_idx -= 1
            dataset_names = [
                "/".join(path.split("/")[path_component_idx:]) for path in list_of_paths
            ]
        return dataset_names

    @pa.check_types(head=NUM_ROWS_TO_VALIDATE_FOR_LABEL_DF)
    def _load_target_classes(
        self,
    ) -> DataFrame[ActivityEndOfSequenceLabelDataModel]:
        """initializes target classes for subpath activity classification"""
        if self.activity_label_file_paths is None:
            raise ValueError("No activity label file path provided")
        activity_labels_df_lst = [
            self._load_activity_label_file(path)
            for path in self.activity_label_file_paths
        ]

        dataset_names = self._get_dataset_name_list(self.activity_label_file_paths)

        for i, df in enumerate(activity_labels_df_lst):
            df["dataset_name"] = dataset_names[i]
            activity_labels_df_lst[i] = df

        activity_labels_df = pd.concat(activity_labels_df_lst).dropna(axis=1, how="any")

        if "Unnamed: 0" in activity_labels_df.columns:
            activity_labels_df.drop("Unnamed: 0", axis=1, inplace=True)
        logger.info(activity_labels_df.head())

        # Filter for labels for the given task
        activity_labels_df = activity_labels_df[
            activity_labels_df.activity_type_name.isin(
                self.label_enum.to_label_name_dict().values()
            )
        ]
        return activity_labels_df

    def filter_out_trackids(self, track_ids: list[str]) -> None:
        """Filters out examples with a given trackId

        Updates the dataset class in place to remove examples with the given trackId


        This allows us to filter duplicate examples from other training sets,
        ensure validation and training sets are disjoint etc.

        Parameters
        ----------
        track_ids : list[str]
            list of trackIds to filter out
        """
        logger.info(f"Filtering out {len(track_ids)} trackIds")
        track_id_series = pd.Series(self.track_ids)
        track_id_filter_idxs = track_id_series[
            ~track_id_series.isin(track_ids)
        ].index.values
        self.trajectories = self.trajectories[track_id_filter_idxs]
        self.activity_class_targets = self.activity_class_targets[track_id_filter_idxs]
        logger.info(
            f"Activit Claass post filter {pd.Series(self.activity_class_targets).value_counts()}"
        )
        self.trajectory_send_time_pairs = [
            self.trajectory_send_time_pairs[i] for i in track_id_filter_idxs
        ]
        self.track_ids = self.track_ids[track_id_filter_idxs]

    def _load_track_context(
        self, trajectory_paths: list[str], send_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Loads a track context from a path
        Parameters
        ----------
        trajectory_path : str
            path to trajectory csv

        Returns
        -------
        pd.DataFrame
            dataframe of trajectory
        """

        trajectory_df = pd.concat(
            [
                self._load_trackfile(path, usecols=ATLAS_ACTIVITY_COLUMNS_WITH_META)
                for path in trajectory_paths
            ]
        )
        trajectory_df.loc[:, "send"] = pd.to_datetime(
            trajectory_df["send"],
            format=self.dataset_config.get("DATE_FORMAT", DATE_FORMAT),
        )

        if not trajectory_df["send"].is_monotonic_increasing:
            trajectory_df = trajectory_df.sort_values("send")
        # Convert send times to pd.Timestamp
        return trajectory_df[trajectory_df["send"] <= send_time].tail(
            self.max_context_length
        )

    def _pick_random_context_segment(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """Picks a random segment of the trajectory"""
        start_idx = np.random.randint(0, len(trajectory) - self.min_context_length)
        return trajectory.iloc[start_idx : start_idx + self.max_context_length]

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:
        """gets item from dataset for activity classification

        If there is not sufficient context, the item will have the key "enough_context" set to False and will return early
        Then, the collate function will ignore that sample and not pass it to the model
        This is only true during training, during inference we will still return the sample prediction
        """
        if self.is_online:
            if self.use_paths:
                trajectory_path = self.trajectories[idx]
                trajectory_paths = trajectory_path
                trajectory = self._load_trackfile(trajectory_path)
            else:
                trajectory = self.trajectories[idx]
                trajectory_paths = (
                    None  # perhaps add a way to find the file for monitoring
                )
        else:
            trajectory_paths, send_time = self.trajectory_send_time_pairs[idx]
            trajectory = self._load_track_context(trajectory_paths, send_time)

        dataset_membership_name = (
            self.dataset_membership[idx] if not self.is_online else "online"
        )
        # Data Frame Valiadtion and format coercion is performed in the preprocess function
        processed_trajectory = self._preprocess(trajectory)

        track_length = len(processed_trajectory)

        clipped_trajectory = processed_trajectory.iloc[-self.max_context_length :]

        if self.is_train:
            clipped_trajectory = self.apply_augmentations(clipped_trajectory)

        is_there_enough_context = track_length >= self.min_ais_messages
        if not is_there_enough_context and (not self.is_online):
            return {"enough_context": is_there_enough_context}

        if self.is_online and not is_there_enough_context:
            raise ValueError("Not enough context for online inference")

        if "amount_of_light" in self.dataset_config["MODEL_INPUT_COLUMNS_ACTIVITY"]:
            clipped_trajectory = self._add_amount_of_light_feature(clipped_trajectory)

        if "rel_cog" in self.dataset_config["MODEL_INPUT_COLUMNS_ACTIVITY"]:
            clipped_trajectory = self._add_rel_cog_feature(clipped_trajectory)

        # Form Model input tensors
        time_intervals = self._get_time_intervals(
            clipped_trajectory, self.cpe_kernel_size
        )
        # THis can occur joinly and be sped up
        spatial_intervals = self._get_spatial_intervals(
            clipped_trajectory, self.cpe_kernel_size
        )

        spatiotemporal_intervals = np.stack(
            (time_intervals, spatial_intervals), axis=2
        ).astype(np.float32)

        assert spatiotemporal_intervals.shape == (
            len(clipped_trajectory),
            self.cpe_kernel_size,
            2,
        )
        most_recent_data = clipped_trajectory.iloc[-500:]

        selected_columns = clipped_trajectory[
            self.dataset_config["MODEL_INPUT_COLUMNS_ACTIVITY"]
        ]
        selected_columns = self._normalize_features(
            selected_columns,
            self.dataset_config["FEATURE_NORMALIZATION_CONFIG_ACTIVITY"],
        )

        traj_array = selected_columns.to_numpy().astype(np.float32)

        binned_ship_type = VESSEL_TYPES_BIN_DICT[
            clipped_trajectory["category"].iloc[0].astype(np.float32)
        ]

        if not self.is_online:
            activity_label = self.activity_class_targets[idx].astype(np.int64)
        else:
            activity_label = np.array([]).astype(np.int64)
            send_time = None
        return {
            "metadata": {
                "flag_code": clipped_trajectory["flag_code"].iloc[0],
                "binned_ship_type": binned_ship_type,
                "entity_name": str(
                    clipped_trajectory["name"].iloc[0]
                ),  # in case entity name is an int
                "trackId": clipped_trajectory["trackId"].iloc[0],
                "file_location": trajectory_paths,
                "send_time": send_time,
                "track_length": track_length,
                "most_recent_data": most_recent_data,
                "dataset_membership": dataset_membership_name,
            },
            "inputs": {
                "traj_array": traj_array,
                "spatiotemporal_intervals": spatiotemporal_intervals,
            },
            "activity_label": activity_label,
            "enough_context": is_there_enough_context,
        }


# TODO: RENAME buoy vessel labels


class AISTrajectoryEntityDataset(AISTrajectoryBaseDataset):
    """Dataset for AIS Entity Classification

    Parameters
    ----------
    AISTrajectoryBaseDataset : [type]
        AiS Trajectory Base Dataset
    """

    def __init__(
        self,
        dataset_config: dict,
        entity_labels_path: Optional[str] = None,
        mode: MODETYPES = "train",
        in_memory_data: Optional[pd.DataFrame] = None,
        label_enum: Any = AtlasEntityLabelsTraining,
    ) -> None:
        super().__init__(
            mode=mode,
            dataset_config=dataset_config,
        )
        """Dataset for AIS Buoy Vessel Classification

        Parameters
        ----------
        entity_labels_path : str
            Path to csv file with buoy vessel labels
        in_memory_data: Optional[pd.DataFrame] = None,
            New argument, if provided, will use this dataframe instead of reading from disk
            in order to use the same class during training and inference

        Raises
        ------
        ValueError
            If there are no buoy or vessel labels in the dataset

        Returns
        -------
        None
        """
        self.label_enum = label_enum
        self.entity_labels_path = entity_labels_path
        if not all(
            x in list(dataset_config["FEATURE_NORMALIZATION_CONFIG_ENTITY"].keys())
            for x in dataset_config["MODEL_INPUT_COLUMNS_ENTITY"]
        ):
            raise ValueError(
                f"Missing columns in {dataset_config['FEATURE_NORMALIZATION_CONFIG_ENTITY'].keys()} \
                {dataset_config['MODEL_INPUT_COLUMNS_ENTITY']}"
            )
        if self.is_online:
            self._init_online_mode(in_memory_data)
        elif self.is_train:
            self._init_train_mode()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _init_online_mode(self, in_memory_data: pd.DataFrame) -> None:
        """Initializes the dataset for online mode."""
        self.trajectories = in_memory_data

    def _init_train_mode(self) -> None:
        """Initializes the dataset for train mode."""
        missing_args = []
        if self.cpe_kernel_size is None:
            missing_args.append("cpe_kernel_size")
        if self.entity_labels_path is None:
            missing_args.append("entity_labels_path")
        if self.n_total_trajectories is None:
            missing_args.append("n_total_trajectories")
        if self.max_trajectory_length is None:
            missing_args.append("max_trajectory_length")

        if missing_args:
            raise ValueError(
                f"Missing arguments for train mode: {', '.join(missing_args)}"
            )
        self.trajectory_lengths_file = self.dataset_config["TRAJECTORY_LENGTHS_FILE"]
        default_num_classes = len(self.label_enum.to_class_descriptions())
        self.class_descriptions = [
            self.label_enum.to_label_name_dict()[i]
            for i in range(self.dataset_config.get("NUM_CLASSES", default_num_classes))
        ]
        logger.info(f"{self.class_descriptions=}")
        self.entity_class_df = self._initialize_target_classes()
        self.entity_class_df = self.entity_class_df[
            self.entity_class_df.index.isin(self._filter_short_trajectories())
        ]
        if self.dataset_config["USE_WEKA"]:
            logger.info("Using Weka")
            path_series = pd.Series(self.entity_class_df.index.values)
            self.entity_class_df.loc[:, "Path"] = path_series.apply(
                lambda x: x.replace("gs://ais-track-data", "/data")
            ).values
            if "NUM_CLASSES" in self.dataset_config:
                self.entity_class_df = self.entity_class_df[
                    self.entity_class_df.entity_class_label.isin(
                        self.class_descriptions[: self.dataset_config["NUM_CLASSES"]]
                    )
                ]
            self.entity_class_df = self.entity_class_df.set_index("Path", drop=True)
            logger.info(self.entity_class_df.head())
        self.trajectories = self.entity_class_df.index.to_numpy()
        self.targets = self.entity_class_df.entity_class_label.to_numpy()

    @pa.check_types
    def _initialize_target_classes(self) -> DataFrame[EntityClassLabelDataModel]:
        """Initializes target classes for the dataset"""
        self.entity_labels = pd.read_csv(self.entity_labels_path)
        self.entity_labels["entity_class_label"] = self.entity_labels[
            "entity_class_label"
        ].map(self.label_enum.to_name_label_dict())
        # in buoy vessel data we say unknown type is negative, so we need to filter for that
        valid_entity_labels = self.entity_labels.loc[
            self.entity_labels["entity_class_label"] >= 0
        ].set_index("Path")
        logger.info(valid_entity_labels.head())
        # log the column names
        return valid_entity_labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """gets item from dataset for entity classification"""
        if self.is_online:
            trajectory = self.trajectories[idx]
            trajectory_path = None
        else:
            trajectory_path = self.trajectories[idx]
            trajectory = self._load_trackfile(trajectory_path)

        trajectory = self._preprocess(trajectory)
        # this sets the ship type to be based on the ais_categories.csv file category column
        binned_ship_type = VESSEL_TYPES_BIN_DICT[trajectory.category.iloc[0]]
        if not self.is_online:
            if len(trajectory) < self.min_ais_messages:
                raise ValueError(
                    f"Filtering is improper for {trajectory_path} \
                    with length {len(trajectory)} \
                    need {self.min_ais_messages} messages min"
                )
            trajectory = trajectory.iloc[: self.max_trajectory_length]
        if self.is_train:
            trajectory = self.apply_augmentations(trajectory)

        if "amount_of_light" in self.dataset_config["MODEL_INPUT_COLUMNS_ENTITY"]:
            trajectory = self._add_amount_of_light_feature(trajectory)

        if "rel_cog" in self.dataset_config["MODEL_INPUT_COLUMNS_ENTITY"]:
            trajectory = self._add_rel_cog_feature(trajectory)
        selected_columns = trajectory[self.dataset_config["MODEL_INPUT_COLUMNS_ENTITY"]]
        selected_columns = self._normalize_features(
            selected_columns, self.dataset_config["FEATURE_NORMALIZATION_CONFIG_ENTITY"]
        )
        traj_tensor = torch.as_tensor(selected_columns.to_numpy(), dtype=torch.float32)

        time_intervals = self._get_time_intervals(trajectory, self.cpe_kernel_size)
        spatial_intervals = self._get_spatial_intervals(
            trajectory, self.cpe_kernel_size
        )

        spatiotemporal_intervals = torch.as_tensor(
            np.stack((time_intervals, spatial_intervals), axis=2, dtype=np.float32)
        )

        assert spatiotemporal_intervals.shape == (
            len(trajectory),
            self.cpe_kernel_size,
            2,
        )
        entity_class_targets = torch.as_tensor(
            self.targets[idx] if not self.is_online else np.array([])
        )
        return {
            "metadata": {
                "flag_code": trajectory.flag_code.iloc[0],
                "binned_ship_type": binned_ship_type,  # Needs to be changed to binned AIS category
                "ais_type": trajectory.category.iloc[0],
                "entity_name": str(
                    trajectory["name"].iloc[0]
                ),  # in case entity name is an int
                "trackId": trajectory.trackId.iloc[0],
                "file_location": trajectory_path,
                "mmsi": str(trajectory["mmsi"].iloc[0]),
                "track_length": len(trajectory),
            },
            "inputs": {
                "traj_tensor": traj_tensor,
                "spatiotemporal_intervals": spatiotemporal_intervals,
            },
            "targets": {
                "class_id": entity_class_targets,
            },
        }
