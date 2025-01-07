""""Module for running offline batch inference on the ATLAS activity model"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from atlantes.atlas.ais_dataset import ActivityDatasetEndOfSequence
from atlantes.atlas.atlas_utils import (AtlasActivityLabelsTraining,
                                        read_trajectory_lengths_file)
from atlantes.atlas.training_utils import rank_zero_only
from atlantes.datautils import ALL_META_DATA_INDEX_PATHS
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import \
    AtlasActivityPostProcessor
from atlantes.log_utils import get_logger
from atlantes.utils import load_all_metadata_indexes
from torch.utils.data import DataLoader

# TODO: Allow outputs across month Boundaries by getting intitial inputs across trackIds
logger = get_logger(__name__)

MIN_CONTEXT_LENGTH = 1000


def get_activity_batch_inferencee_config() -> dict:
    """Get the configuration for batch inference"""
    config_path = (
        Path(__file__).parent / "config" / "activity_batch_inference_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class OutputWriter:
    """Class for writing output to a file"""

    postprocessor = AtlasActivityPostProcessor()

    @classmethod
    def extract_info_from_output(
        cls, outputs: list[tuple[AtlasActivityLabelsTraining, dict, dict]]
    ) -> list[dict]:
        output_records = []
        for output in outputs:
            activity_class, details, metadata = output
            postprocessed_output = cls.postprocessor.postprocess(output)
            postprocessed_class, _ = postprocessed_output
            # DO We want to postprocess as well here? Yes I think so
            confidence = details["confidence"]
            anchor_score = details["outputs"][
                AtlasActivityLabelsTraining.ANCHORED.value
            ]
            transiting_score = details["outputs"][
                AtlasActivityLabelsTraining.TRANSITING.value
            ]
            moored_score = details["outputs"][AtlasActivityLabelsTraining.MOORED.value]
            fishing_score = details["outputs"][
                AtlasActivityLabelsTraining.FISHING.value
            ]

            most_recent_data = metadata["most_recent_data"]
            ais_type = most_recent_data["category"].iloc[-1]
            binned_ais_category = metadata["binned_ship_type"]
            nav_status = most_recent_data["nav"].iloc[-1]
            lat = most_recent_data["lat"].iloc[-1]
            lon = most_recent_data["lon"].iloc[-1]
            sog = most_recent_data["sog"].iloc[-1]
            track_id = most_recent_data["trackId"].iloc[-1]
            mmsi = most_recent_data["mmsi"].iloc[-1]
            dist2coast = most_recent_data["dist2coast"].iloc[-1]
            send_time = most_recent_data["send"].iloc[-1]
            record = {
                "activity_class": activity_class.name.lower(),
                "confidence": confidence,
                "file_location": metadata["file_location"],
                "anchor_score": anchor_score,
                "transiting_score": transiting_score,
                "moored_score": moored_score,
                "fishing_score": fishing_score,
                "ais_type": ais_type,
                "binned_ais_category": binned_ais_category,
                "nav_status": nav_status,
                "lat": lat,
                "lon": lon,
                "sog": sog,
                "dist2coast": dist2coast,
                "send_time": send_time,
                "track_id": track_id,
                "mmsi": mmsi,
                "postprocessed_class": postprocessed_class,
            }
            output_records.append(record)
        return output_records

    @classmethod
    def write_output_to_file(cls, outputs: list, output_dir: str) -> None:
        """Write output to a file"""
        extracted_info = cls.extract_info_from_output(outputs)
        os.makedirs(output_dir, exist_ok=True)
        output_df = pd.DataFrame(extracted_info)
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_file_name = f"inference_output_{current_time}.csv"
        output_df.to_csv(os.path.join(output_dir, output_file_name), index=False)


def filter_short_trajectories(track_lengths_file: str) -> pd.Series:
    """Filter out short trajectories"""
    track_lengths_df = read_trajectory_lengths_file(track_lengths_file)
    track_lengths_df = track_lengths_df[track_lengths_df["Length"] > MIN_CONTEXT_LENGTH]
    return track_lengths_df.Path


def basic_collate_fn(batch: list) -> list:
    return batch


def dump_config(config: dict, output_dir: str) -> None:
    """Dump the config to a file"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def batch_inference(
    rank: int,
    world_size: int,
    batch_inference_config: dict,
    file_paths: list[str],
) -> None:
    """Run batch inference"""
    logger.info(f"Running batch inference on rank {rank}")
    offline_inference_config = batch_inference_config["offline_inference"]
    NUM_OUTPUTS_TO_WRITE_AT_ONCE = offline_inference_config[
        "NUM_OUTPUTS_TO_WRITE_AT_ONCE"
    ]
    # Save the config to the output directory
    output_dir = offline_inference_config["OUTPUT_DIR"]
    rank_zero_only(dump_config, rank)(batch_inference_config, output_dir)
    # Split up files across all ranks
    rank_file_paths = np.array_split(file_paths, world_size)[rank].tolist()
    data_config = batch_inference_config["data"]
    # This should probably be eval mode
    dataset = ActivityDatasetEndOfSequence(
        data_config, online_file_paths=rank_file_paths, mode="online"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=offline_inference_config["BATCH_SIZE"],
        collate_fn=basic_collate_fn,
        num_workers=offline_inference_config["N_WORKERS"],
        multiprocessing_context="forkserver",
        shuffle=False,
    )
    model = AtlasActivityModel(device=rank, inference_config=batch_inference_config)
    model_id = model.MODEL_ID
    output_dir = output_dir + f"/{model_id}"
    outputs: list[Tuple[AtlasActivityLabelsTraining, dict, dict]] = []
    num_samples_processed = 0
    for batch in dataloader:
        try:
            batch_output = model.run_inference(batch)
        except Exception as e:
            logger.error(f"Error in batch inference: {e}")
            continue
        else:
            num_samples_processed += len(batch_output)

        outputs.extend(batch_output)
        if len(outputs) > NUM_OUTPUTS_TO_WRITE_AT_ONCE:
            logger.info(f"Writing {len(outputs)} outputs to file")
            # should be written to a beaker results dataset
            OutputWriter.write_output_to_file(outputs, output_dir)
            outputs = []


def main() -> None:
    """Main function for running batch inference"""
    config = get_activity_batch_inferencee_config()
    offline_inference_config = config["offline_inference"]
    world_size = offline_inference_config["N_GPUS"]
    num_years = offline_inference_config["NUM_YEARS"]
    # Create seperate config
    if num_years == 1:
        metadata_index_paths = ALL_META_DATA_INDEX_PATHS[:1]
    elif num_years == 2:
        metadata_index_paths = ALL_META_DATA_INDEX_PATHS
    else:
        raise ValueError("Only 1 or 2 years of data are supported")
    metadata_df = load_all_metadata_indexes(
        metadata_index_paths,
        columns=["Path"],
    )
    # Trajectory lengths filtering
    filtered_track_lengths = filter_short_trajectories(
        "gs://ais-track-data/labels/trajectory_lengths/trajectory_lengths_2022_2023_0.parquet"
    )
    list_of_files = metadata_df[
        metadata_df["Path"].isin(filtered_track_lengths.to_numpy())
    ]["Path"].values
    logger.info(list_of_files[:5])
    logger.info(f"Number of files: {len(list_of_files)}")
    np.random.shuffle(list_of_files)
    file_paths = list_of_files
    logger.info(f"Running batch inference with world size: {world_size}")
    if world_size <= 1:
        rank = 0
        batch_inference(
            rank,
            world_size,
            config,
            file_paths,
        )
    else:
        mp.spawn(
            batch_inference,
            args=(
                world_size,
                config,
                file_paths,
            ),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
