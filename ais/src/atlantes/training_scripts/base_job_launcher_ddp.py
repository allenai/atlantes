"""# TODO: Use this for all training scripts


"""

import logging
import os
import warnings
from typing import Callable

import torch
import torch.multiprocessing as mp
from atlantes.log_utils import get_logger
from atlantes.utils import find_most_recent_checkpoint, process_additional_args

logger = get_logger(__name__)


def launch_ddp_job(
    init_and_launch_trainer: Callable,
    data_config: dict,
    experimental_config_train: dict,
    experimental_config_model: dict,
    additional_args: tuple[str],
) -> None:
    """Main function to launch a single GPU or a single node DDP job.

    Parameters
    ----------
    init_and_launch_trainer : Callable
        Trainer set up and launch function must take the following arguments:
        rank: int,
        world_size: int,
        data_config: dict,
        experimental_config_train: dict,
        experimental_config_model: dict

    data_config : dict
        Data config
    experimental_config_train : dict
        Experimental config train
    experimental_config_model : dict
        Experimental config model

    Returns
    -------
    None
    """
    logger.setLevel(logging.INFO)
    warnings.simplefilter(action="ignore", category=FutureWarning)
    logger.info(f"{additional_args=}")
    config_path = os.environ.get("EXPERIMENTAL_CONFIG_PATH")
    if config_path:
        import yaml

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        data_config = config["data"]
        experimental_config_train = config["train"]
        experimental_config_model = config["hyperparameters"]
    experimental_config_train = process_additional_args(
        additional_args, experimental_config_train
    )
    experimental_config_model = process_additional_args(
        additional_args, experimental_config_model
    )
    data_config = process_additional_args(additional_args, data_config)

    if experimental_config_train["MODEL_SAVE_DIR"] is not None:
        try:
            checkpoint = find_most_recent_checkpoint(
                experimental_config_train["MODEL_SAVE_DIR"]
            )
        except FileNotFoundError:
            logger.info(
                f"No checkpoint found in {experimental_config_train['MODEL_SAVE_DIR']}"
            )
        else:
            logger.info(f"Updating to : {checkpoint} to resume training")
            experimental_config_train["MODEL_LOAD_DIR"] = experimental_config_train[
                "MODEL_SAVE_DIR"
            ]
            experimental_config_train["MODEL_CHECKPOINT"] = checkpoint
            # No longer use features
            experimental_config_train["USE_FEATURES_ONLY"] = False

    logger.info(f"{data_config=}")
    logger.info(f"{experimental_config_train=}")
    logger.info(f"{experimental_config_model=}")
    logger.info(experimental_config_train["NUM_GPUS"])
    world_size = experimental_config_train["NUM_GPUS"]
    debug_mode = experimental_config_train["DEBUG_MODE"]
    logger.info(
        f"Cuda support: {torch.cuda.is_available()}: {torch.cuda.device_count()} devices"
    )

    if debug_mode or world_size <= 1:
        rank = 0
        init_and_launch_trainer(
            rank,
            world_size,
            data_config,
            experimental_config_train,
            experimental_config_model,
        )

    else:
        mp.spawn(
            init_and_launch_trainer,
            args=(
                world_size,
                data_config,
                experimental_config_train,
                experimental_config_model,
            ),
            nprocs=world_size,
        )
