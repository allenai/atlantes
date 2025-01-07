"""Sublayers for ATLAS model

This module contains the sublayers for the ATLAS model. The squeeze attention layer is adapted from the Trajformer paper
and the CPE layer is adapted from the Trajformer paper.
"""

import torch
import torch.nn as nn
from atlantes.atlas.atlas_utils import Tensor
from atlantes.log_utils import get_logger
from torch.nn import functional as F

logger = get_logger(__name__)


class CPE(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        dropout: float,
    ) -> None:
        """Class to create continuous point embeddings

        Parameters
        ----------
        channels: int
            Number of channels in the input
        kernel_size: int
            Size of the kernel
        stride: int
            Stride of the kernel
        groups: int
            Number of groups for the attention
        dropout: float
            Dropout probability

        Returns
        -------
        Tensor
            Output tensor
        """
        super(CPE, self).__init__()
        assert (
            channels % groups == 0
        ), f"dim {channels} should be divided by num_heads {groups}."
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.groups = groups

        self.dim = channels
        self.num_heads = groups
        head_dim = channels // groups
        self.scale = head_dim**-0.5
        # TODO: Remove this unneded 2 and just use 1 but only when we train a new model
        self.kv_func = nn.Linear(self.dim, self.dim * 2, bias=True)

        self.attn_drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.groups),
        )

    def build_sliding_window_padding_mask(
        self, padding_mask: Tensor, batch_size: Tensor
    ) -> Tensor:
        """Build a padding mask for the sliding window spaito temporal kernel"""

        sliding_window_padding_mask = (
            F.unfold(
                padding_mask.unsqueeze(1).unsqueeze(-1).float(),
                kernel_size=(self.kernel_size, 1),
                padding=((self.kernel_size - 1) // 2, 0),
                stride=self.stride,
            )
            .reshape(batch_size, 1, self.kernel_size, -1)
            .squeeze(1)
        )  # b, k, t
        # THis may not be needed
        padding_mask = padding_mask.unsqueeze(-1) * sliding_window_padding_mask.permute(
            0, 2, 1
        )
        return padding_mask.unsqueeze(-1)  # B, T, K, 1 # last dim is for broadcasting

    def spatial_temporal_kernel_forward(
        self, dis: torch.Tensor, padding_mask: Tensor
    ) -> torch.Tensor:
        """Perform the forward pass of the spatial temporal kernel"""
        b, _, _, _ = dis.shape  # b,t, kernel_size, 2
        spatio_temporal_kernel = self.mlp(dis)

        padding_mask = self.build_sliding_window_padding_mask(padding_mask, b)

        mask_value = -1e9
        spatio_temporal_kernel = spatio_temporal_kernel.masked_fill(
            padding_mask == 0, mask_value
        )

        # bt, k, groups
        spatio_temporal_kernel = spatio_temporal_kernel.reshape(
            -1, self.kernel_size, self.groups
        )
        spatio_temporal_kernel = spatio_temporal_kernel.permute(0, 2, 1).unsqueeze(
            2
        )  # bt, groups, 1, k # So all points are flat here we have no batch dimension
        spatio_temporal_kernel = spatio_temporal_kernel.softmax(-1)
        spatio_temporal_kernel = self.attn_drop(spatio_temporal_kernel)

        return spatio_temporal_kernel

    def get_sliding_window_features(self, input_tensor: Tensor) -> Tensor:
        """Get sliding window features"""

        # Extract Continuous kernels by projecting each spatio-temporal delta into mlp dimension then back into number of groups
        b, c, t = input_tensor.shape
        k = F.unfold(
            input_tensor.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=((self.kernel_size - 1) // 2, 0),
            stride=self.stride,
        ).reshape(
            b, c, self.kernel_size, -1
        )  # b, c, k, t

        k = k.permute(0, 3, 2, 1).reshape(-1, self.kernel_size, c)  # bt, k, c

        kv = (
            self.kv_func(k)
            .reshape(  # What is this 2 doing it basically undoes the upward projection and just drops half of the features
                b * t, self.kernel_size, 2, self.num_heads, c // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
        )
        # TODO: K is disgarded here need to fix this code
        k, v = kv[0], kv[1]  # bt, groups, k, c
        return v

    def forward(
        self, x: torch.Tensor, dis: torch.Tensor, padding_mask: Tensor
    ) -> torch.Tensor:
        """Forward method for CPE

        Parameters
        ----------
        x: Tensor
            Input tensor
        dis: Tensor
            Distance tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        # x: [b, c, t]
        # dis: [b, t, kernel_size, 2]

        b, c, t = x.shape
        spatio_temporal_kernel = self.spatial_temporal_kernel_forward(dis, padding_mask)

        sliding_window_features = self.get_sliding_window_features(x)

        # Combine features and continuous kernels dot product is a sum of the product of the windowed features
        output_feature_seq_with_spatio_temporal = (
            spatio_temporal_kernel @ sliding_window_features
        ).reshape(b, c, t)
        return output_feature_seq_with_spatio_temporal


class CNN1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_residual: bool,
        channel_dim_only_layer_norm: bool,
        use_layer_norm: bool,
    ) -> None:
        """CNN Block of 2 convolutions"""
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size, stride, padding)
        self.activation = nn.GELU()

        # TEMPORARILY LEAVING THIS HERE for entity backwards compatibility
        # Maybe a dict would be more configurable
        if not use_layer_norm:
            logger.debug("Registering batch norms")
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None
        self.use_residual = use_residual
        self.channel_dim_only_layer_norm = channel_dim_only_layer_norm
        self.use_layer_norm = use_layer_norm

    def set_padding_mask(self, padding_mask: Tensor) -> None:
        """Set the padding mask for the CNN Block"""
        self.padding_mask = padding_mask

    def reset_padding_mask(self) -> None:
        """Reset the padding mask for the CNN Block"""
        self.padding_mask = None

    def mask_padding_for_convolution(
        self, padding_mask: Tensor, input: Tensor
    ) -> Tensor:
        """Mask padding from the convolutional layer"""
        padding_mask = padding_mask.unsqueeze(1)
        return padding_mask * input

    def layer_norm(self, x: Tensor) -> Tensor:
        """Layer normalization for the CNN Block"""
        if self.channel_dim_only_layer_norm:
            _, C, _ = x.shape
            x = x.permute(0, 2, 1)
            x = F.layer_norm(x, (C,))
            x = x.permute(0, 2, 1)
        else:
            normalized_shape = x.shape[1:]
            x = F.layer_norm(x, normalized_shape)
        return x

    def apply_norm1(self, x: Tensor) -> Tensor:
        """Apply normalization 1 for the CNN Block"""
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.norm1(x)
        return x

    def apply_norm2(self, x: Tensor) -> Tensor:
        """Apply normalization 2 for the CNN Block"""
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.norm2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the CNN Block"""
        x = self.mask_padding_for_convolution(self.padding_mask, x)

        if self.use_residual:
            residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.apply_norm1(x)
        x = self.mask_padding_for_convolution(self.padding_mask, x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.apply_norm2(x)
        x = self.mask_padding_for_convolution(self.padding_mask, x)
        if self.use_residual:
            return x + residual
        return x
