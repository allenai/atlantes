"""Test to launch atlas entity training and overfit on a single small batch


This test should be run manually as need, but will be skipped in CI unless we see the need to provisin a gpu runner and start doing testing there"""

import os
from pathlib import Path

import pytest
import torch
import yaml
from atlantes.training_scripts.launch_train_atlas_entity import launch_train_atlas_entity
from atlantes.log_utils import get_logger
from click.testing import CliRunner

logger = get_logger(__name__)


def gpu_available() -> bool:
    """Check if a gpu is available."""
    return torch.cuda.is_available()


@pytest.fixture
def test_data_config() -> dict:
    """Test data config."""
    data_config_path = os.path.join(
        Path(__file__).parent.parent,
        "test-data",
        "test-experimental-configs",
        "test_data_config.yaml",
    )

    with open(data_config_path, "r") as file:
        config = yaml.safe_load(file)["dataset"]
    return config


@pytest.fixture
def test_experimental_config_train() -> dict:
    """Test experimental config train."""
    experimental_config_train_path = os.path.join(
        Path(__file__).parent.parent,
        "test-data",
        "test-experimental-configs",
        "test_buoy_experimental_config.yaml",
    )

    with open(experimental_config_train_path, "r") as file:
        config = yaml.safe_load(file)["train"]
    return config


@pytest.fixture
def test_experimental_config_model() -> dict:
    """Test experimental config model."""
    experimental_config_model_path = os.path.join(
        Path(__file__).parent.parent,
        "test-data",
        "test-experimental-configs",
        "test_buoy_experimental_config.yaml",
    )
    with open(experimental_config_model_path, "r") as file:
        config = yaml.safe_load(file)["hyperparameters"]
    return config


@pytest.mark.skipif(not gpu_available(), reason="No GPU available, Run manually")
def test_launch_train_atlas_entity_single_gpu(
    test_data_config: dict,
    test_experimental_config_train: dict,
    test_experimental_config_model: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test Launch the training of the atlas entity model"""
    test_experimental_config_model["NUM_GPUS"] = 1
    runner = CliRunner()
    commands = [
        "--data-config",
        test_data_config,
        "--experimental-config-train",
        test_experimental_config_train,
        "--experimental-config-model",
        test_experimental_config_model,
    ]
    result = runner.invoke(launch_train_atlas_entity, commands)
    captured = capsys.readouterr()
    logger.info(captured.out)
    logger.info(captured.err)
    logger.info(result.output)
    assert result.exit_code == 0, f"Output: {result.output}, Error: {result.exception}"


@pytest.mark.skipif(not gpu_available(), reason="No GPU available, Run manually")
def test_launch_train_atlas_entity_multi_gpu(
    test_data_config: dict,
    test_experimental_config_train: dict,
    test_experimental_config_model: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """Launch the training of the atlas entity model."""
    runner = CliRunner()
    commands = [
        "--data-config",
        test_data_config,
        "--experimental-config-train",
        test_experimental_config_train,
        "--experimental-config-model",
        test_experimental_config_model,
    ]
    result = runner.invoke(launch_train_atlas_entity, commands)
    logger.info(result.output)
    assert result.exit_code == 0, f"Output: {result.output}, Error: {result.exception}"
