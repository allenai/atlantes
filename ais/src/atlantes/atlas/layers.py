"""Module Class for Continual Positional Encoding (CPE) and Squeeze Transformer"""

import torch.nn as nn
from atlantes.atlas.atlas_utils import Tensor
from atlantes.atlas.sublayers import CPE, CNN1DBlock
from atlantes.log_utils import get_logger

logger = get_logger(__name__)


class CPE_Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        kernel_size: int,
        stride: int,
        groups: int,
        cpe_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the CPE Block"""
        super(CPE_Block, self).__init__()

        blocks = [
            CPE(in_features, kernel_size, stride, groups, dropout)
            for i in range(cpe_layers)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, dis: Tensor, padding_mask: Tensor) -> Tensor:
        """Forward pass of the CPE Block"""
        residual = x
        # TODO: make this happen in init so we stop looping heree
        for block in self.blocks:
            x = block(x, dis, padding_mask)
        return x + residual


class Local1DEmbeddingCNN(nn.Module):
    def __init__(
        self,
        num_layers: int,
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
        """Initialize the CNN1D Local Feature Embedding Block"""
        super().__init__()
        self.num_middle_layers = num_layers - 2
        self.in_block = CNN1DBlock(
            in_channels,
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride,
            padding,
            use_residual=False,
            channel_dim_only_layer_norm=channel_dim_only_layer_norm,
            use_layer_norm=use_layer_norm,
        )
        self.mid_block = nn.Sequential(
            *[
                CNN1DBlock(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    padding,
                    use_residual=use_residual,
                    channel_dim_only_layer_norm=channel_dim_only_layer_norm,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(self.num_middle_layers)
            ]
        )
        self.out_block = CNN1DBlock(
            hidden_dim,
            hidden_dim,
            out_channels,
            kernel_size,
            stride,
            padding,
            use_residual=False,
            channel_dim_only_layer_norm=channel_dim_only_layer_norm,
            use_layer_norm=use_layer_norm,
        )
        # THink about adding maxpooling or strided convolutions
        self.activation = nn.GELU()
        self.out_hidden = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            self.activation,
            nn.Linear(out_channels, out_channels),
        )

    def set_padding_mask_for_all_blocks(self, padding_mask: Tensor) -> None:
        """Set the padding mask for all blocks"""
        self.in_block.set_padding_mask(padding_mask)
        for block in self.mid_block:
            block.set_padding_mask(padding_mask)
        self.out_block.set_padding_mask(padding_mask)

    def reset_padding_mask_for_all_blocks(self) -> None:
        """Reset the padding mask for all blocks"""
        self.in_block.reset_padding_mask()
        for block in self.mid_block:
            block.reset_padding_mask()
        self.out_block.reset_padding_mask()

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """Forward pass of the CNN1D Local Feature Embedding Block"""
        # Saving mask statefully to not mess with the nn sequential which requires single input output
        # Passing as a tuple means for many layer blocks we would need to propagate the mask through
        # Alternative implementation is x, _ = self.in_block((x, padding_mask)) where each block always returns both
        self.set_padding_mask_for_all_blocks(padding_mask)
        x = self.in_block(x)
        x = self.mid_block(x)
        x = self.out_block(x)
        x = self.out_hidden(x.permute(0, 2, 1))
        self.padding_mask = None
        self.reset_padding_mask_for_all_blocks()
        return x  # expected shape B x T x C
