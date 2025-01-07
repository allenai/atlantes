"""Script to launch the training of the atlas activity model.
"""

import click
from atlantes.atlas.activity_trainer import AtlasActivityRealTimeTrainer
from atlantes.atlas.atlas_utils import \
    get_experiment_config_atlas_activity_real_time
from atlantes.log_utils import get_logger
from atlantes.training_scripts.base_job_launcher_ddp import launch_ddp_job

logger = get_logger(__name__)

DATA_CONFIG = get_experiment_config_atlas_activity_real_time("data")
EXPERIMENTAL_CONFIG_TRAIN = get_experiment_config_atlas_activity_real_time("train")
EXPERIMENTAL_CONFIG_MODEL = get_experiment_config_atlas_activity_real_time(
    "hyperparameters"
)


def init_and_launch_activity_trainer_real_time(
    rank: int,
    world_size: int,
    data_config: dict,
    experimental_config_train: dict,
    experimental_config_model: dict,
) -> None:
    """Initialize and launch the activity trainer."""
    trainer = AtlasActivityRealTimeTrainer(
        data_config=data_config,
        experimental_config_train=experimental_config_train,
        experimental_config_model=experimental_config_model,
    )
    # Launch training job
    if trainer.debug_mode:
        trainer.setup_and_train(rank=0, world_size=1)
    else:
        trainer.setup_and_train(rank, world_size)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--data-config",
    default=DATA_CONFIG,
    help="Data config",
    type=dict,
)
@click.option(
    "--experimental-config-train",
    default=EXPERIMENTAL_CONFIG_TRAIN,
    help="Experimental config train",
    type=dict,
)
@click.option(
    "--experimental-config-model",
    default=EXPERIMENTAL_CONFIG_MODEL,
    help="Experimental config model",
    type=dict,
)
@click.argument("additional_args", nargs=-1)
def launch_train_atlas_activity(
    data_config: dict,
    experimental_config_train: dict,
    experimental_config_model: dict,
    additional_args: tuple[str],
) -> None:
    """Main function to launch the training of the atlas activity model for real time prediction

    This function handles launching activity training on optionally multiple gpus as defined
    in the train config. To change the experimental configuration update the .yaml files in atlantes/atlas/config
    We enable passing in config dicts to make testing easier

    Parameters
    ----------
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
    launch_ddp_job(
        init_and_launch_activity_trainer_real_time,
        data_config,
        experimental_config_train,
        experimental_config_model,
        additional_args,
    )


if __name__ == "__main__":
    launch_train_atlas_activity()
