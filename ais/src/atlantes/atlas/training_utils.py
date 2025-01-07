"""Utility Functions for AIS Model Training"""

import math
import os
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data.distributed
import wandb
from atlantes.log_utils import get_logger
from numpy.random import Generator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

logger = get_logger(__name__)


MAIN_DEVICE_RANK = 0

# add an only call function on a given rank decorator

OPTIMIZER_CLASSES = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

SCHEDULER_CLASSES = {
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "none": None,
}


def rank_zero_only(fn: Callable, rank: int) -> Callable:
    """A function and decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank == 0 or rank is None:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


class RandomSegment(object):
    """Randomly Segments an ais sequence into a smaller sequence fo given length"""

    def __init__(self, segment_length: int, random_state: int) -> None:
        """Initialize the RandomSegment class

        Parameters
        ----------
        segment_length : int
            The length of the segment to extract
        """
        self.segment_length = segment_length
        self.rng = Generator(np.random.PCG64(random_state))
        self.indices_dict: dict = {}

    def _add_indices(self, trajectory_df: pd.DataFrame, start_idx: int) -> None:
        """Add the indices to the indices_dict

        Parameters
        ----------
        trajectory_df : pd.DataFrame
            The trajectory to segment
        """
        # get the length of the sequence
        trackid = trajectory_df["trackId"].iloc[0]
        if trackid not in self.indices_dict.keys():
            self.indices_dict[trajectory_df["trackId"].iloc[0]] = set([start_idx])
        self.indices_dict[trackid].add(start_idx)

    def __call__(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """Call the RandomSegment class

        Parameters
        ----------
        sample : dict
            The sample to segment

        Returns
        -------
        dict
            The segmented sample
        """
        # get the length of the sequence
        sequence_length = trajectory_df.shape[0]
        if sequence_length <= self.segment_length:
            return trajectory_df
        # randomly choose a start index
        start_index = self.rng.integers(
            low=0, high=sequence_length - self.segment_length
        )
        self._add_indices(trajectory_df, start_index)
        return trajectory_df.iloc[start_index : start_index + self.segment_length]


def unpack_metadata(
    metadata_list: list,
) -> Tuple[list[str], list[str], list[str], list[str]]:
    """
    Unpacks the metadata from a list of dictionaries.

    Parameters
    ----------
        metadata_list (list[Parameters]): A list of dictionaries containing metadata.

    Returns
    -------
        Tuple[list[str], list[int], list[str], list[str]]: A tuple containing lists of unpacked metadata.
    """
    entity_names = [metadata["entity_name"] for metadata in metadata_list]
    trackIds = [metadata["trackId"] for metadata in metadata_list]
    file_locations = [metadata["file_location"] for metadata in metadata_list]
    binned_ship_types_metadata = [metadata["binned_ship_type"] for metadata in metadata_list]

    return entity_names, trackIds, file_locations, binned_ship_types_metadata


def setup_distributed(rank: int, world_size: int) -> None:
    """Setup function for distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """Cleanup function for distributed training"""
    dist.destroy_process_group()


def warmup(current_step: int, warmup_steps: int) -> float:
    """Warmup function for the learning rate scheduler"""
    # make a linear increase as we approach warmup steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def get_binary_weighted_sampler(
    targets: np.ndarray, indices: np.ndarray, total_samples: int
) -> WeightedRandomSampler:
    """Make a binary weighted sampler from a set of indices"""
    class_probs_train = 1.0 / np.bincount(targets[indices])
    class_sampling_weights_train = class_probs_train[targets[indices]]
    return WeightedRandomSampler(
        class_sampling_weights_train, total_samples, replacement=True
    )


def get_cross_entropy_class_weights(
    targets: torch.Tensor, num_classes: int, padding_value: float
) -> torch.Tensor:
    """Get the class weights for the cross entropy loss"""
    nonzero_indices = torch.nonzero(targets != padding_value)
    weights = torch.ones(
        num_classes,
        dtype=torch.float32,
        device=targets.device,
    )
    labels, counts = targets[nonzero_indices].unique(return_counts=True)
    weights_val = 1.0 / (counts + torch.finfo(torch.float32).eps)
    weights_val = weights_val / torch.min(weights_val)
    weights[labels] = weights_val
    return weights


def create_ddp_model(
    model: nn.Module, rank: int, world_size: int, find_unused_params: bool = False
) -> nn.Module:
    """Create a distributed data parallel model"""
    # move moodel to GPU with id rank
    model = model.to(rank)
    # create distributed data parallel model
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=find_unused_params,
    )
    return model


def move_model_to_gpu(
    model: nn.Module, rank: int, world_size: int, device: torch.device
) -> nn.Module:
    """Move the model to the GPU, and transform to DDP if necessary"""
    if world_size > 1:
        model = create_ddp_model(model, rank, world_size)
    else:
        model = model.to(device)
    return model


def save_model_wandb_artifact(
    model: nn.Module, date_string: str, filename: str, dir: str
) -> None:
    """Save the model weights as a wandb artifact

    Parameters
    ----------
    model : nn.Module
        The model to save
    date_string : str
        The date string to use for the artifact name
    dir : str
        The directory to save the model weights to

    Returns
    -------
    None
        Saves the model weights to the specified directory"""

    file_name = f"{filename}{date_string}.pt"
    torch.save(model.state_dict(), os.path.join(dir, file_name))
    artifact = wandb.Artifact("model_weights", type="model")
    artifact.add_file(os.path.join(dir, file_name))
    wandb.log_artifact(artifact)


class DistributedWeightedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: Dataset,
        targets: np.ndarray,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        shuffle: bool = True,
    ) -> None:
        """Initialize the sampler"""
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.targets = targets
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        )  # How many samples per replica
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement

    def calculate_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the weights for each sample in the dataset"""
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
        )
        weight = 1.0 / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self) -> Any:
        """Iterate over the dataset"""
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample for the current replica and gpu in the given epoch
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # select only the wanted targets for this subsample
        targets = torch.tensor(self.targets)[indices]
        assert len(targets) == self.num_samples
        # randomly sample this subset, producing balanced classes
        weights = self.calculate_weights(targets)
        subsample_balanced_idxs = torch.multinomial(
            weights, self.num_samples, self.replacement
        )
        # now map these target idxs back to the original dataset index...
        dataset_idxs = torch.tensor(indices)[subsample_balanced_idxs]
        return iter(dataset_idxs.tolist())

    def __len__(self) -> int:
        """The number of samples in the current epoch"""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler"""
        self.epoch = epoch
