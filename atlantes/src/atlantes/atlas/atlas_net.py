"""Module for ATLAS class

ATLAS is a transformer based model.
ATLAS stands for AIS Transformer Learning with Active Subpaths.

Notes:
The architecture is based on the following papers:

TrajFormer: Efficient Trajectory Classification with Transformers
Liang et al. (2022)
See: https://dl.acm.org/doi/pdf/10.1145/3511808.3557481


The assignment matrix is fixed and learned in an online fashion which is then use to
reduce dimensionality of the attention operations

TODO: create model output namedtuples
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from atlantes.atlas.atlas_utils import (AtlasActivityLabelsTraining,
                                        remove_module_from_state_dict)
from atlantes.atlas.basemodel import BaseModel
from atlantes.atlas.layers import CPE_Block, Local1DEmbeddingCNN
from atlantes.log_utils import get_logger
from torch import Tensor
from torch.nn.init import constant_, trunc_normal_

logger = get_logger(__name__)


class Atlas(BaseModel):
    """Main Transformer class"""

    def __init__(
        self,
        c_in: int,  # input_dimensionality, default features are lat, lon, sog, cog
        name: str = "ATLAS: AIS Transformer Learning with Active Subpaths",
        transformer_layers_pre_squeeze: int = 0,  # number of transformer layers before the squeeze transformer layer
        n_heads: int = 4,  # number of heads
        token_dim: int = 128,  # token dimension
        mlp_dim: int = 256,  # mlp dimension in Transformer
        cpe_layers: int = 2,  # the number of CPE layers
        cpe_kernel_size: int = 5,  # kernel size for CPE
        cnn_layers: int = 1,  # the number of CNN layers
        cnn_kernel_size: int = 3,  # kernel size for CNN
        dropout_p: float = 0.1,  # dropout rate
        use_residual_cnn: bool = False,  # whether to use residual connections in CNN
        use_channel_dim_only_layernorm_cnn: bool = False,  # whether to use only the channel dimension in CNN Normalization
        use_layer_norm_cnn: bool = False,  # whether to use layer normalization in CNN
        qkv_bias: bool = True,  # whether to use bias for qkv
        qk_scale: Optional[Any] = None,  # scale factor for qk
        use_binned_ship_type: bool = True,  # whether to use vessel category
        use_lora: bool = False,  # whether to use LoRA
    ):
        """init for the base trajectory transformer"""
        super().__init__()
        self.name = name
        self.c_in = c_in
        self.token_dim = token_dim
        self.local_layers = cpe_layers
        self.transformer_layers_pre_squeeze = transformer_layers_pre_squeeze
        self.cpe_kernel_size = cpe_kernel_size
        self.cnn_layers = cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = self.cnn_kernel_size // 2
        self.use_binned_ship_type = use_binned_ship_type
        self.n_heads = n_heads
        self.use_lora = use_lora
        # Continuous Point embedding layers
        if cpe_layers > 0:
            logger.debug("Using CPE layers")
            self.conv1d = nn.Sequential(
                nn.Conv1d(c_in, token_dim, 1, 1, 0),
                nn.ReLU(
                    inplace=True
                ),  # Should this be a ReLU? Initially restricted to 1x1 convolutions
            )
            self.involution = CPE_Block(  # Would we ever want stride to not be 1? Would htis even work?
                token_dim, self.cpe_kernel_size, 1, self.n_heads, cpe_layers, dropout_p
            )
        else:
            logger.debug("No CPE layers")
            self.conv1d = nn.Sequential(
                nn.Conv1d(c_in, token_dim, self.kernel_size, 1, 4),
                nn.ReLU(inplace=True),
            )

        if self.cnn_layers > 0:
            self.local_feature_embedding = Local1DEmbeddingCNN(
                self.cnn_layers,
                token_dim,
                token_dim // 2,
                token_dim,
                self.cnn_kernel_size,
                1,
                self.cnn_padding,
                use_residual=use_residual_cnn,
                channel_dim_only_layer_norm=use_channel_dim_only_layernorm_cnn,
                use_layer_norm=use_layer_norm_cnn,
            )
        self.norm = nn.LayerNorm(token_dim)

        if transformer_layers_pre_squeeze > 0:
            self.pre_squeeze_transformer = nn.TransformerEncoderLayer(
                token_dim,
                self.n_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout_p,
                batch_first=True,
            )
            self.transformer_pre_squeeze = nn.TransformerEncoder(
                self.pre_squeeze_transformer,
                transformer_layers_pre_squeeze,
                norm=self.norm,
                enable_nested_tensor=True,
                mask_check=True,
            )

        self.apply(self.init_weights)
        self.is_defaults_set = False

    def init_weights(self, m: Union[nn.Linear, nn.LayerNorm]) -> None:
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self,
        inputs: Tensor,
        dis: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute of the features


        Parameters
        ----------
        inputs : Tensor
            The input tensor has shape [b, n_time, c_in]
        dis : Tensor
            The spatio-temporal deltas (time, space) with shape [b, n_time,cpe_kernel_size, 2]
        padding_mask : Tensor
            The attention mask for messages for the padding (b, n_time)


        Returns
        -------
        out : Tensor
            The output tensor with shape [b, n_time, token_dim]

        """
        if self.use_lora:
            inputs.requires_grad = True
        if self.local_layers > 0:
            # Project the channel dim into the token dim
            out = self.conv1d(inputs.permute(0, 2, 1))  # b, token_dim, n_time
            out = self.involution(out, dis, padding_mask)  # b, token_dim, n_time
        else:
            out = self.conv1d(inputs.permute(0, 2, 1))  # b, token_dim, n_time

        if self.cnn_layers > 0:
            # wants B, C, T inputs
            out = self.local_feature_embedding(out, padding_mask)
        if self.transformer_layers_pre_squeeze > 0:
            # expects B, T, C
            # Flipping Mask as Transformer Encoder Expects Internally mask is filled and converted to float
            out = out.permute(0, 2, 1) if self.cnn_layers == 0 else out
            padding_mask = padding_mask if padding_mask is None else padding_mask.logical_not()
            out = self.transformer_pre_squeeze(
                out,
                src_key_padding_mask=padding_mask,
            )
        return out


class AtlasActivityEndOfSequenceTaskNet(Atlas):
    """ATLAS Activity Model for predicting the subpath activity class of an AIS trajectory"""

    def __init__(
        self,
        name: str = "ATLAS-Activity: AIS Transformer Learning with Active Subpaths",
        c_in: int = 4,  # input_dim # features (lat, lon, sog, cog)
        subpath_output_dim: int = 3,  # output_dim for subpath learning
        transformer_layers_pre_squeeze: int = 0,  # number of transformer layers before the squeeze transformer layer
        n_heads: int = 4,  # number of heads
        token_dim: int = 128,  # token dimension
        mlp_dim: int = 256,  # mlp dimension in Transformer
        cpe_layers: int = 2,  # the number of CPE layers
        cpe_kernel_size: int = 5,  # kernel size for CPE
        cnn_layers: int = 1,  # the number of CNN layers
        cnn_kernel_size: int = 3,  # kernel size for CNN
        use_residual_cnn: bool = False,
        use_channel_dim_only_layernorm_cnn: bool = False,
        use_layer_norm_cnn: bool = False,
        dropout_p: float = 0.1,  # dropout rate
        qkv_bias: bool = True,  # whether to use bias for qkv
        qk_scale: Optional[Any] = None,  # scale factor for qk
        use_binned_ship_type: bool = True,  # whether to use vessel category
        use_lora: bool = False,  # whether to use LoRA
    ):
        """init for ATLAS Activity Model"""
        super().__init__(
            name=name,
            c_in=c_in,
            transformer_layers_pre_squeeze=transformer_layers_pre_squeeze,
            n_heads=n_heads,
            token_dim=token_dim,
            mlp_dim=mlp_dim,
            cpe_layers=cpe_layers,
            cpe_kernel_size=cpe_kernel_size,
            cnn_layers=cnn_layers,
            cnn_kernel_size=cnn_kernel_size,
            use_residual_cnn=use_residual_cnn,
            use_channel_dim_only_layernorm_cnn=use_channel_dim_only_layernorm_cnn,
            use_layer_norm_cnn=use_layer_norm_cnn,
            dropout_p=dropout_p,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_binned_ship_type=use_binned_ship_type,
            use_lora=use_lora,
        )
        if self.use_binned_ship_type:
            token_dim += 1
        self.globalavgpool = nn.AdaptiveAvgPool2d((token_dim, 1))
        self.subpath_output_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_dim, subpath_output_dim),
        )

    def forward(  # type: ignore
        self,
        inputs: Tensor,
        spatiotemporal_tensor: Tensor,
        binned_ship_type: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Main forward method of ATLAS Activity Model


        Parameters
        ----------
        x : Tensor
            The trajectory
        dis : Tensor
            The spatio-temporal deltas (time, space)
        binned_ship_type : Tensor
            The vessel category
        padding_mask : Tensor
            The attention mask for messages



        Returns
        -------
        out : Tensor
            The output tensor with shape [b * max_num_subpaths_in_batch, subpath_output_dim]

        """
        # dis: [b, n_time, kernel_size, 2]
        out = self.forward_features(
            inputs=inputs,
            dis=spatiotemporal_tensor,
            padding_mask=padding_mask,
        )  # [b, n, token_dim]
        B, N, token_dim = out.size()
        out = self.globalavgpool(out).squeeze()
        if self.use_binned_ship_type:
            # Copy vessel category to each node
            binned_ship_type = binned_ship_type.unsqueeze(-1).repeat(1, N).unsqueeze(-1)
            out = torch.cat((out, binned_ship_type), dim=-1)  # B, N, token_dim+1
        out = self.subpath_output_layer(out)
        if out.dim() < 2:
            # Ensure output has batch dimension
            out = out.unsqueeze(0)
        return out


class AtlasEntity(Atlas):
    """ATLAS Entity Model for predicting the entity class of an AIS trajectory

    Entity class is the class of the entity transmitting the AIS messages.
    E.g Buoy, Vessel, etc.

    Can also be used for finer entity classification, e.g. vessel type."""

    def __init__(
        self,
        name: str = "ATLAS-Entity: AIS Transformer Learning with Active Subpaths",
        c_in: int = 4,  # input_dim # features (lat, lon, sog, cog)
        c_out: int = 2,  # output_dim.
        transformer_layers_pre_squeeze: int = 0,  # number of transformer layers before the squeeze transformer layer
        n_heads: int = 4,  # number of heads
        token_dim: int = 128,  # token dimension
        mlp_dim: int = 128,  # mlp dimension in Transformer
        cpe_layers: int = 2,  # the number of CPE layers
        cpe_kernel_size: int = 5,  # kernel size for CPE
        cnn_layers: int = 1,  # the number of CNN layers
        cnn_kernel_size: int = 3,  # kernel size for CNN
        use_residual_cnn: bool = False,
        use_channel_dim_only_layernorm_cnn: bool = False,
        use_layer_norm_cnn: bool = False,
        dropout_p: float = 0.1,  # dropout rate
        qkv_bias: bool = True,  # whether to use bias for qkv
        qk_scale: Optional[Any] = None,  # scale factor for qk
        use_binned_ship_type: bool = True,  # whether to use vessel category
        use_lora: bool = False,  # whether to use LoRA
    ):
        """init for ATLAS Activity Model"""
        super().__init__(
            name=name,
            c_in=c_in,
            transformer_layers_pre_squeeze=transformer_layers_pre_squeeze,
            n_heads=n_heads,
            token_dim=token_dim,
            mlp_dim=mlp_dim,
            cpe_layers=cpe_layers,
            cpe_kernel_size=cpe_kernel_size,
            cnn_layers=cnn_layers,
            cnn_kernel_size=cnn_kernel_size,
            use_residual_cnn=use_residual_cnn,
            use_channel_dim_only_layernorm_cnn=use_channel_dim_only_layernorm_cnn,
            use_layer_norm_cnn=use_layer_norm_cnn,
            dropout_p=dropout_p,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_binned_ship_type=use_binned_ship_type,
            use_lora=use_lora,
        )
        if self.use_binned_ship_type:
            self.globalavgpool = nn.AdaptiveAvgPool2d((token_dim, 1))
            token_dim += 1
        else:
            self.globalavgpool = nn.AdaptiveAvgPool2d((token_dim, 1))
        self.output_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_dim, c_out),
        )

    def forward(  # type: ignore
        self,
        inputs: Tensor,
        spatiotemporal_tensor: Tensor,
        binned_ship_type: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Main forward method of ATLAS Entity Model

        Parameters
        ----------
        x : Tensor
            The trajectory
        dis : Tensor
            The spatio-temporal deltas (time, space)
        binned_ship_type : Tensor
            The vessel category
        padding_mask : Tensor
            The attention mask for messages for the padding

        Returns
        -------
        entity_class_scores : Tensor
            The entity class scores
        """
        # dis: [b, n_time, kernel_size, 2]
        out = self.forward_features(
            inputs=inputs,
            dis=spatiotemporal_tensor,
            padding_mask=padding_mask,
        )  # [b, n, c]
        B, N, C = out.size()
        out_entity_class = self.globalavgpool(out).squeeze()  # [b, C]
        if self.use_binned_ship_type:
            out_entity_class = torch.hstack(
                [out_entity_class, binned_ship_type.unsqueeze(1)]
            )  # B, C+1
            # Add vessel category to each node for subpaths
            binned_ship_type = binned_ship_type.unsqueeze(-1).repeat(1, N).unsqueeze(-1)
            out = torch.cat((out, binned_ship_type), dim=-1)  # B, N, C+1
        entity_class_scores = self.output_layer(out_entity_class)  # [b, n, c_out]
        return entity_class_scores  # [b, c_out]


def initialize_atlas_activity_end_of_sequence_model(
    model_config: dict, data_config: dict
) -> AtlasActivityEndOfSequenceTaskNet:
    """Initialize the ATLAS activity end of sequence model for inference."""
    return AtlasActivityEndOfSequenceTaskNet(
        c_in=len(data_config["MODEL_INPUT_COLUMNS_ACTIVITY"]),
        subpath_output_dim=len(AtlasActivityLabelsTraining),
        transformer_layers_pre_squeeze=model_config["N_PRE_SQUEEZE_TRANSFORMER_LAYERS"],
        n_heads=model_config["N_HEADS"],
        token_dim=model_config["TOKEN_DIM"],
        mlp_dim=model_config["MLP_DIM"],
        cpe_kernel_size=model_config["CPE_KERNEL_SIZE"],
        cpe_layers=model_config["CPE_LAYERS"],
        cnn_layers=model_config["CNN_LAYERS"],
        cnn_kernel_size=model_config["CNN_KERNEL_SIZE"],
        dropout_p=model_config["DROPOUT_P"],
        qkv_bias=model_config["QKV_BIAS"],
        use_binned_ship_type=model_config["USE_SHIP_TYPE"],
        use_lora=model_config["USE_LORA"],
    )


def load_atlas_activity_end_of_sequence_model(
    path: Path, map_location: torch.device, data_config: dict, model_config: dict
) -> AtlasActivityEndOfSequenceTaskNet:
    """Load an instance of ATLAS activity end of sequence model into memory and return it."""
    model = initialize_atlas_activity_end_of_sequence_model(model_config, data_config)
    atlas_state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(remove_module_from_state_dict(atlas_state_dict))
    return model
