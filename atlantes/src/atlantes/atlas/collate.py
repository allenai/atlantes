""" Module for the collate functions for the atlas dataset
"""

from typing import Any, NamedTuple, Optional

import torch
from atlantes.log_utils import get_logger
from torch import Tensor
from torch.nn import functional as F

logger = get_logger(__name__)


class ActivityDatasetEndOfSequenceCollatedDataOutput(NamedTuple):
    """Named Tuple for the collated data for the activity model

    Parameters
    ----------
    input_tensor : Tensor
        Tensor of shape (batch_size, max_seq_length, features) containing the features
    spatiotemporal_interval_tensor : Tensor
        Tensor of shape (batch_size, max_seq_length, kernel_size, 2) containing the time and spatial intervals
    activity_labels: Tensor
        Tensor of shape (batch_size, max_num_subpaths) containing the subpath labels
    padding_mask: Tensor
        Tensor of shape (batch_size, max_num_messages) containing the attention mask for the message padding
    binned_ship_type_tensor : Tensor
        Tensor of shape (batch_size, ) containing the ship types
    metadata : list
        list of dictionaries containing the metadata for each sample
    seq_lengths : list
        list of the sequence lengths for each sample
    """

    input_tensor: Tensor
    spatiotemporal_interval_tensor: Tensor
    activity_labels: Tensor
    padding_mask: Tensor
    binned_ship_type_tensor: Tensor
    metadata: list
    seq_lengths: list


class EntityDatasetCollatedDataOutput(NamedTuple):
    """Named Tuple for the collated data for the entity model

    Parameters
    ----------
    input tensors: Tensor
        Tensor of shape (batch_size, max_seq_length, features) å
    spatiotemporal_intervals: Tensor
        Tensor of shape (batch_size, max_seq_length, kernel_size, 2) containing the time and spatial intervals
    targets: Tensor
        Tensor of shape (batch_size, ) containing the class ålabels
    padding_mask: Tensor
        Tensor of shape  (batch_size, max_num_messages) containing the attention mask for the message padding
    binned_ship_types: Tensor
        Tensor of shape (batch_size, ) containing the ship types
    metadata: list
        list of dictionaries containing the metadata for each sample
    seq_lengths: list
        list of the sequence lengths for each sample"""

    input_tensor: Tensor
    spatiotemporal_interval_tensor: Tensor
    entity_class_targets: Tensor
    padding_mask: Tensor
    binned_ship_type_tensor: Tensor
    metadata: list
    seq_lengths: list


class BaseAISCollater:
    """Base AIS Collater class for padding inputs"""

    def __init__(self, use_prepadding: bool) -> None:
        """Initialize the collater for the real time activity prediction

        Parameters
        ----------
        use_prepadding : bool
            Whether to use prepadding or not
        """
        self.use_prepadding = use_prepadding

    def prepad_inputs(
        self, seq: Tensor, spatial_temporal_intervals: Tensor, pad_length: int
    ) -> tuple:
        """Prepad the inputs to the same length"""
        padded_seq = F.pad(
            seq,
            (0, 0, pad_length, 0),
        )
        padded_spatial_temporal_intervals = F.pad(
            spatial_temporal_intervals,
            (0, 0, 0, 0, pad_length, 0),
        )
        return padded_seq, padded_spatial_temporal_intervals

    def postpad_inputs(
        self, seq: Tensor, spatial_temporal_intervals: Tensor, pad_length: int
    ) -> tuple:
        """Postpad the inputs to the same length"""
        padded_seq = F.pad(
            seq,
            (0, 0, 0, pad_length),
        )
        padded_spatial_temporal_intervals = F.pad(
            spatial_temporal_intervals,
            (0, 0, 0, 0, 0, pad_length),
        )
        return padded_seq, padded_spatial_temporal_intervals

    def pad_inputs(
        self, seq: Tensor, spatial_temporal_intervals: Tensor, pad_length: int
    ) -> tuple:
        """Pad the inputs to the same length"""
        if self.use_prepadding:
            return self.prepad_inputs(seq, spatial_temporal_intervals, pad_length)
        else:
            return self.postpad_inputs(seq, spatial_temporal_intervals, pad_length)

    def build_padding_mask(
        self, seq_lengths: list, max_seq_length: int
    ) -> Tensor:
        """Build the padding mask for the inputs"""
        if self.use_prepadding:
            padding_mask = torch.tensor(
                [
                    [0.0] * (max_seq_length - seq_length) + [1.0] * seq_length
                    for seq_length in seq_lengths
                ],
                dtype=torch.bool,
            )
        else:
            padding_mask = torch.tensor(
                [
                    [1.0] * seq_length + [0.0] * (max_seq_length - seq_length)
                    for seq_length in seq_lengths
                ],
                dtype=torch.bool,
            )
        return padding_mask

    def __call__(self, batch: list) -> Any:
        """Method Executed by dataloader"""
        raise NotImplementedError("Method not implemented")


class RealTimeActivityCollater(BaseAISCollater):
    """Custom collate function for Activity Classification"""

    def __init__(self, use_prepadding: bool) -> None:
        """Initialize the collater for the real time activity prediction

        Parameters
        ----------
        use_prepadding : bool
            Whether to use prepadding or not
        """
        super().__init__(use_prepadding)

    def __call__(
        self,
        batch: list,
    ) -> Optional[ActivityDatasetEndOfSequenceCollatedDataOutput]:
        """Custom collate function for AIS to pad AIS trajectories to the same length
        for end of sequence activity prediction

        Parameters
        ----------
        batch : list
            list of samples from the EndOfSequenceActivityPredictionDataset

        Returns
        -------
        ActivityDatasetEndOfSequenceCollatedDataOutput
            Named Tuple containing the collated data for the activity model
        """
        # Remove samples that do not have enough context or are bad etc
        # TODO: add a debug log for this
        batch = [data for data in batch if data["enough_context"]]
        if len(batch) == 0:
            logger.warning("No samples with enough context in batch")
            return None
        seq_lengths = [x["inputs"]["traj_array"].shape[0] for x in batch]
        max_seq_length = max(seq_lengths)
        padded_data = []
        spatiotemporal_intervals = []
        targets = []
        binned_ship_types = []
        metadata = []
        for j, x in enumerate(batch):
            seq = torch.as_tensor(x["inputs"]["traj_array"].copy())  # Tensor
            spatiotemporal_interval = torch.as_tensor(
                x["inputs"]["spatiotemporal_intervals"].copy()
            )  # Tensor
            binned_ship_type = torch.as_tensor(x["metadata"]["binned_ship_type"])  # Tensor
            target = torch.as_tensor(x["activity_label"].copy())  # Tensor
            pad_length = max_seq_length - seq_lengths[j]
            padded_seq, padded_spatiotemporal_interval = self.pad_inputs(
                seq, spatiotemporal_interval, pad_length
            )
            padded_data.append(padded_seq)
            spatiotemporal_intervals.append(padded_spatiotemporal_interval)
            targets.append(target)
            binned_ship_types.append(binned_ship_type)
            metadata.append(x["metadata"])

        input_tensor = torch.stack(padded_data)
        spatiotemporal_interval_tensor = torch.stack(spatiotemporal_intervals)
        # padding mask hack for pytorch bug
        padding_mask = self.build_padding_mask(seq_lengths, max_seq_length)
        binned_ship_type_tensor = torch.tensor(binned_ship_types)

        target_tensor = torch.stack(targets)
        return ActivityDatasetEndOfSequenceCollatedDataOutput(
            input_tensor=input_tensor,
            spatiotemporal_interval_tensor=spatiotemporal_interval_tensor,
            activity_labels=target_tensor,
            padding_mask=padding_mask,
            binned_ship_type_tensor=binned_ship_type_tensor,
            metadata=metadata,
            seq_lengths=seq_lengths,
        )


def ais_collate_entity_class_with_subpaths(
    batch: list,
) -> EntityDatasetCollatedDataOutput:
    """Custom collate function for AIS to pad AIS trajectories to the same length

    Parameters
    ----------
    batch : list
        list of samples from the AISTrajectorySubpathsDataset

    Returns
    -------
    EntityDatasetCollatedDataOutput
        Named Tuple containing the collated data for the entity model
    """
    seq_lengths = [x["inputs"]["traj_tensor"].shape[0] for x in batch]
    max_seq_length = max(seq_lengths)
    padded_data = []
    spatiotemporal_intervals = []
    targets = []
    binned_ship_types = []
    metadata = []
    for j, x in enumerate(batch):
        seq = x["inputs"]["traj_tensor"]  # Tensor
        spatiotemporal_interval = x["inputs"]["spatiotemporal_intervals"]  # Tensor
        binned_ship_type = x["metadata"]["binned_ship_type"]  # Tensor
        target = x["targets"]["class_id"]  # Tensor
        pad_length = (
            max_seq_length - seq_lengths[j] + 1
        )  # Always pad 1 extra so that we cna bypass tis bug https://github.com/pytorch/pytorch/issues/107084
        padded_seq = F.pad(
            seq,
            (
                0,
                0,
                0,
                pad_length,
            ),
        )
        padded_data.append(padded_seq)
        padded_spatiotemporal_interval = F.pad(
            spatiotemporal_interval,
            (
                0,
                0,
                0,
                0,
                0,
                pad_length,
            ),
        )
        spatiotemporal_intervals.append(padded_spatiotemporal_interval)
        targets.append(target)
        binned_ship_types.append(binned_ship_type)
        metadata.append(x["metadata"])
    # Create the input tensors
    input_tensor = torch.stack(padded_data)
    spatiotemporal_interval_tensor = torch.stack(spatiotemporal_intervals)

    padding_mask = torch.tensor(
        [
            [1.0] * seq_length + [0.0] * (1 + max_seq_length - seq_length)
            for seq_length in seq_lengths
        ]
    )

    binned_ship_type_tensor = torch.tensor(binned_ship_types)
    # Stack the target tensors
    target_tensor = torch.stack(targets)

    return EntityDatasetCollatedDataOutput(
        input_tensor=input_tensor,
        spatiotemporal_interval_tensor=spatiotemporal_interval_tensor,
        padding_mask=padding_mask,
        entity_class_targets=target_tensor,
        binned_ship_type_tensor=binned_ship_type_tensor,
        metadata=metadata,
        seq_lengths=seq_lengths,
    )
