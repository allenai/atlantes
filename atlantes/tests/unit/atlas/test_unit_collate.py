"""Unit Tests for AIS Collation"""

import numpy as np
import pytest
import torch
from atlantes.atlas.collate import (
    ActivityDatasetEndOfSequenceCollatedDataOutput, BaseAISCollater,
    RealTimeActivityCollater)


@pytest.fixture
def seq_inputs() -> torch.Tensor:
    """Input Sequence Data with shape T x F"""
    input_length_dim = 3
    num_features_dim = 4

    return torch.rand_like(torch.empty(input_length_dim, num_features_dim))


@pytest.fixture
def spatial_temporal_intervals_inputs() -> torch.Tensor:
    """Input Spatial Temporal Intervals Data with shape T x Window X 2"""
    input_length_dim = 3
    windows_dim = 7
    return torch.rand_like(torch.empty(input_length_dim, windows_dim, 2))


class TestBaseAISCollater:
    def test_init(self) -> None:
        """Test the initialization of the BaseAISCollater class."""
        use_prepadding = True
        collater = BaseAISCollater(use_prepadding=use_prepadding)
        assert collater.use_prepadding is use_prepadding

    def test_prepad_inputs(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the prepad_inputs method of the BaseAISCollater class."""
        collater = BaseAISCollater(use_prepadding=True)
        pad_length = 2
        padded_seq, padded_spatial_temporal_intervals = collater.prepad_inputs(
            seq_inputs, spatial_temporal_intervals_inputs, pad_length
        )
        assert padded_seq[:pad_length].sum() == 0
        assert padded_spatial_temporal_intervals[:pad_length].sum() == 0

    def test_postpad_inputs(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the postpad_inputs method of the BaseAISCollater class."""
        collater = BaseAISCollater(use_prepadding=False)
        pad_length = 2
        padded_seq, padded_spatial_temporal_intervals = collater.postpad_inputs(
            seq_inputs, spatial_temporal_intervals_inputs, pad_length
        )
        assert padded_seq[-pad_length:].sum() == 0
        assert padded_spatial_temporal_intervals[-pad_length:].sum() == 0

    def test_pad_inputs_prepad(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the pad_inputs method of the BaseAISCollater class."""
        pad_length = 2
        collater = BaseAISCollater(use_prepadding=True)
        expected_prepad_seq, expected_prepad_spatial_temporal_intervals = (
            collater.prepad_inputs(
                seq_inputs, spatial_temporal_intervals_inputs, pad_length
            )
        )
        padded_seq, padded_spatial_temporal_intervals = collater.pad_inputs(
            seq_inputs, spatial_temporal_intervals_inputs, pad_length
        )
        assert torch.equal(padded_seq, expected_prepad_seq)
        assert torch.equal(
            padded_spatial_temporal_intervals,
            expected_prepad_spatial_temporal_intervals,
        )

    def test_pad_inputs_postpad(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the pad_inputs method of the BaseAISCollater class."""
        pad_length = 2
        collater = BaseAISCollater(use_prepadding=False)
        expected_postpad_seq, expected_postpad_spatial_temporal_intervals = (
            collater.postpad_inputs(
                seq_inputs, spatial_temporal_intervals_inputs, pad_length
            )
        )
        padded_seq, padded_spatial_temporal_intervals = collater.pad_inputs(
            seq_inputs, spatial_temporal_intervals_inputs, pad_length
        )
        assert torch.equal(padded_seq, expected_postpad_seq)
        assert torch.equal(
            padded_spatial_temporal_intervals,
            expected_postpad_spatial_temporal_intervals,
        )

    def test_build_padding_mask_prepad(self) -> None:
        """Test the build_padding_mask method of the BaseAISCollater class."""
        expected_mask = torch.tensor(
            [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        )
        collater = BaseAISCollater(use_prepadding=True)
        seq_lengths = [3, 4, 5]
        max_seq_length = 5
        padding_mask = collater.build_padding_mask(seq_lengths, max_seq_length)
        assert torch.equal(padding_mask, expected_mask)

    def test_build_padding_mask_postpad(self) -> None:
        """Test the build_padding_mask method of the BaseAISCollater class."""
        expected_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
        )
        collater = BaseAISCollater(use_prepadding=False)
        seq_lengths = [3, 4, 5]
        max_seq_length = 5
        padding_mask = collater.build_padding_mask(seq_lengths, max_seq_length)
        assert torch.equal(padding_mask, expected_mask)


class TestRealTimeActivityCollater:
    def test_init(self) -> None:
        """Test the initialization of the RealTimeActivityCollater class."""
        collater = RealTimeActivityCollater(use_prepadding=True)
        assert collater.use_prepadding is True

    def test_call(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the __call__ method of the RealTimeActivityCollater class."""

        collater = RealTimeActivityCollater(use_prepadding=True)
        batch = [
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": True,
            },
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": True,
            },
        ]
        output = collater(batch)
        assert isinstance(output, ActivityDatasetEndOfSequenceCollatedDataOutput)

    def test_call_not_enough_context(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the __call__ method of the RealTimeActivityCollater class. without enough context"""
        collater = RealTimeActivityCollater(use_prepadding=True)
        batch = [
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": False,
            },
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": False,
            },
        ]
        output = collater(batch)
        assert output is None

    def test_not_enough_context_removed(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the __call__ method of the RealTimeActivityCollater class."""
        collater = RealTimeActivityCollater(use_prepadding=True)
        batch = [
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": False,
            },
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": True,
            },
        ]
        output = collater(batch)
        assert isinstance(output, ActivityDatasetEndOfSequenceCollatedDataOutput)
        assert output.input_tensor.shape[0] == 1
        assert output.spatiotemporal_interval_tensor.shape[0] == 1
        assert output.activity_labels.shape[0] == 1
        assert output.padding_mask.shape[0] == 1
        assert output.binned_ship_type_tensor.shape[0] == 1
        assert len(output.metadata) == 1
        assert len(output.seq_lengths) == 1

    def test_data_is_batched(
        self, seq_inputs: torch.Tensor, spatial_temporal_intervals_inputs: torch.Tensor
    ) -> None:
        """Test the __call__ method of the RealTimeActivityCollater class."""
        collater = RealTimeActivityCollater(use_prepadding=True)
        batch = [
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": True,
            },
            {
                "inputs": {
                    "traj_array": seq_inputs.numpy(),
                    "spatiotemporal_intervals": spatial_temporal_intervals_inputs.numpy(),
                },
                "activity_label": np.array([0]),
                "metadata": {"binned_ship_type": 1},
                "enough_context": True,
            },
        ]

        output = collater(batch)
        assert isinstance(output, ActivityDatasetEndOfSequenceCollatedDataOutput)
        assert output.input_tensor.shape[0] == 2
        assert output.spatiotemporal_interval_tensor.shape[0] == 2
        assert output.activity_labels.shape[0] == 2
        assert output.padding_mask.shape[0] == 2
        assert output.binned_ship_type_tensor.shape[0] == 2
        assert len(output.metadata) == 2
        assert len(output.seq_lengths) == 2
