"""Python CLI to launch a Beaker experiment.

Supports passing in a yaml file and specifying the task, name, and other parameters."""

from pathlib import Path
from uuid import uuid4

import click
from atlantes.log_utils import get_logger
from beaker import (Beaker, Dataset, Experiment, ExperimentSpec, ImageSource,
                    TaskSpec)

logger = get_logger(__name__)

beaker = Beaker.from_env()


# Constants
DEFAULT_WORKSPACE = "ai2/earth-systems"
DEFAULT_BUDGET = "ai2/skylight"
DEFAULT_CLUSTER = ["ai2/jupiter-cirrascale-2"]
DEFAULT_PRIORITY = "high"
PATH_TO_TRAINING_SCRIPT = (
    "src/atlantes/training_scripts/launch_train_atlas_activity_real_time.py"
)
CONFIG_MOUNT_PATH = "/experimental_config"
CONFIG_FILENAME = "atlas_activity_real_time_config.yaml"
FILE_LOCATION = f"{CONFIG_MOUNT_PATH}/{CONFIG_FILENAME}"
DEFAULT_GPU_COUNT = 8
DEFAULT_SHARED_MEMORY = "256 GiB"


def push_image_to_beaker(image_name: str, workspace: str) -> ImageSource:
    """Push an image to beaker.

    Parameters
    ----------
    image_name : str
        Name of the image
    workspace : str
        Beaker workspace to use

    Returns
    -------
    ImageSource
        Beaker ImageSource object
    """
    image = beaker.image.create(image_name, image_name, workspace=workspace)
    image_source = ImageSource(beaker=image.full_name)
    return image_source


def upload_config_as_beaker_dataset(config_path_obj: Path) -> Dataset:
    """Upload a config file as a Beaker dataset.

    Parameters
    ----------
    config_path_obj : Path
        Path to the config file

    Returns
    -------
    Dataset
        Beaker Dataset object
    """
    name = config_path_obj.stem + "-" + str(uuid4())[:8]
    dataset = beaker.dataset.create(
        name,
        config_path_obj,
        description="Config file",
        strip_paths=True,
    )
    logger.info(f"Uploaded config file as dataset: {dataset.id}")
    return dataset


def create_task_spec(
    image: str,
    config_dataset: Dataset,
    training_script_path: str,
    config_mount_path: str,
    file_location: str,
    gpu_count: int,
    shared_memory: str,
    cluster: list,
    priority: str,
    task_name: str,
) -> TaskSpec:
    """Create a Beaker task spec.

    Parameters
    ----------
    image : str
        Beaker image source.
    config_dataset : Dataset
        Beaker dataset containing the config file.
    training_script_path : str
        Path to the training script.
    config_mount_path : str
        Path where the config dataset will be mounted.
    file_location : str
        Path to the config file within the container.
    gpu_count : int
        Number of GPUs to allocate.
    shared_memory : str
        Amount of shared memory.
    cluster : list
        List of clusters to run on.
    priority : str
        Priority of the task.
    task_name : str
        Name of the task.

    Returns
    -------
    TaskSpec
        Beaker task spec.
    """
    task_spec = TaskSpec(
        name=task_name,
        image=image,
        command=["python", training_script_path],
        env_vars=[
            {"name": "WANDB_API_KEY", "secret": "WANDB_API_KEY"},
            {
                "name": "GOOGLE_APPLICATION_CREDENTIALS",
                "secret": "GCP_CREDENTIALS_PATH",
            },
            {"name": "EXPERIMENTAL_CONFIG_PATH", "value": file_location},
        ],
        datasets=[
            {"mountPath": "/data", "source": {"weka": "skylight-default"}},
            {
                "mountPath": "/etc/credentials/credentials.json",
                "source": {"secret": "GCP_CREDENTIALS"},
            },
            {"mountPath": config_mount_path, "source": {"beaker": config_dataset.id}},
        ],
        result={"path": "/models"},
        resources={"gpuCount": gpu_count, "sharedMemory": shared_memory},
        context={"priority": priority, "preemptible": True},
        constraints={"cluster": cluster},
    )
    return task_spec


# Make all the configs environment variables and or set defaults
def create_experiment_from_dataset(
    dataset: Dataset,
    image_name: str,
    experiment_name: str,
    training_script_path: str,
    config_mount_path: str,
    file_location: str,
    gpu_count: int,
    shared_memory: str,
    cluster: list,
    priority: str,
    workspace: str,
    budget: str,
    task_name: str,
) -> Experiment:
    """Create a Beaker experiment from a dataset.

    Parameters
    ----------
    dataset : Dataset
        Beaker dataset for config
    image_name : str
        Name of the image
    experiment_name : str
        Name of the experiment
    training_script_path : str
        Path to the training script
    config_mount_path : str
        Path where the config dataset will be mounted
    file_location : str
        Path to the config file within the container
    gpu_count : int
        Number of GPUs to allocate
    shared_memory : str
        Amount of shared memory
    cluster : list
        List of clusters to run on
    priority : str
        Priority of the task
    workspace : str
        Beaker workspace to use
    budget : str
        Beaker budget to use
    task_name : str
        Name of the task

    Returns
    -------
    Experiment
        Beaker experiment
    """
    spec = ExperimentSpec(
        budget=budget,
        tasks=[
            create_task_spec(
                image_name,
                dataset,
                training_script_path=training_script_path,
                config_mount_path=config_mount_path,
                file_location=file_location,
                gpu_count=gpu_count,
                shared_memory=shared_memory,
                cluster=cluster,
                priority=priority,
                task_name=task_name,
            )
        ],
    )
    experiment = beaker.experiment.create(
        spec,
        workspace=workspace,
        name=experiment_name,
    )
    logger.info(f"Created experiment: {experiment.id}")
    return experiment


@click.command()
@click.option("--config-path", type=str, required=True, help="Path to the config file.")
@click.option("--image-name", type=str, required=True, help="Name of the image.")
@click.option(
    "--is-local",
    is_flag=True,
    default=False,
    help="Flag to indicate whether to upload the image.",
)
@click.option(
    "--experiment-name",
    type=str,
    default="test_experiment",
    help="Name of the experiment.",
)
@click.option(
    "--workspace",
    type=str,
    default=DEFAULT_WORKSPACE,
    help="Beaker workspace to use.",
)
@click.option(
    "--budget",
    type=str,
    default=DEFAULT_BUDGET,
    help="Beaker budget to use.",
)
@click.option(
    "-t",
    "--training-script-path",
    type=str,
    default=PATH_TO_TRAINING_SCRIPT,
    help="Path to the training script.",
)
@click.option(
    "--task-name",
    type=str,
    default="training",
    help="Name of the task.",
)
def main(
    config_path: str,
    image_name: str,
    is_local: bool,
    experiment_name: str,
    workspace: str,
    budget: str,
    training_script_path: str,
    task_name: str,
) -> None:
    """Main function to launch a Beaker experiment.

    Parameters
    ----------
    config_path : str
        Path to the config file.
    image_name : str
        Name of the image.
    is_local : bool
        Flag to indicate whether to upload the image.
    experiment_name : str
        Name of the experiment.
    workspace : str
        Beaker workspace to use.
    budget : str
        Beaker budget to use.
    training_script_path : str
        Path to the training script.
    task_name : str
        Name of the task.
    """
    if is_local:
        image = push_image_to_beaker(image_name, workspace)
    else:
        image = ImageSource(beaker=image_name)
    config_path_obj = Path(config_path)
    dataset = upload_config_as_beaker_dataset(config_path_obj)
    CONFIG_MOUNT_PATH = "/experimental_config"
    FILE_LOCATION = f"{CONFIG_MOUNT_PATH}/{config_path_obj.name}"
    create_experiment_from_dataset(
        dataset,
        image,
        experiment_name,
        training_script_path=training_script_path,
        config_mount_path=CONFIG_MOUNT_PATH,
        file_location=FILE_LOCATION,
        gpu_count=DEFAULT_GPU_COUNT,
        shared_memory=DEFAULT_SHARED_MEMORY,
        cluster=DEFAULT_CLUSTER,
        priority=DEFAULT_PRIORITY,
        workspace=workspace,
        budget=budget,
        task_name=task_name,
    )


if __name__ == "__main__":
    main()
