"""Main model inference module for the atlas entity application

Loads the model and runs inference on the data

This includes a base class and a ray serve deployment class
"""

import os
from datetime import datetime
from pathlib import Path

import torch
from atlantes.atlas.atlas_net import AtlasEntity
from atlantes.atlas.atlas_utils import (AtlasEntityLabelsTraining,
                                        AtlasEntityLabelsTrainingWithUnknown,
                                        get_atlas_entity_inference_config,
                                        remove_module_from_state_dict)
from atlantes.atlas.collate import (EntityDatasetCollatedDataOutput,
                                    ais_collate_entity_class_with_subpaths)
from atlantes.inference.atlas_entity.datamodels import (
    EntityMetadata, EntityPostprocessorInput, EntityPostprocessorInputDetails,
    PreprocessedEntityData)
from atlantes.log_utils import get_logger
from atlantes.utils import get_commit_hash
from ray import serve
from torch.nn import functional as F

logger = get_logger("ray.serve")

# Only matters for cI
if torch.cuda.is_available():
    NUM_GPUS = 1
else:
    logger.info("No GPU detected, using CPU")
    NUM_GPUS = 0

NUM_CPUS = 1
NUM_REPLICAS = 1
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


GIT_HASH = get_commit_hash()
MODEL_VERSION = f"{GIT_HASH}_{CURRENT_TIME}"

MODELS_DIR_DEFAULT = str(Path(__file__).parents[2] / "models")

ATLAS_ENTITY_BATCH_MAX_SIZE_DEFAULT = 4
ATLAS_ENTITY_BATCH_MAX_SIZE = int(
    os.environ.get("ATLAS_ENTITY_BATCH_MAX_SIZE", ATLAS_ENTITY_BATCH_MAX_SIZE_DEFAULT)
)
ATLAS_ENTITY_BATCH_WAIT_TIMEOUT_S_DEFAULT = 0.5  # Set Higher for testing
ATLAS_ENTITY_BATCH_WAIT_TIMEOUT_S = float(
    os.environ.get(
        "ATLAS_ENTITY_BATCH_WAIT_TIMEOUT_S", ATLAS_ENTITY_BATCH_WAIT_TIMEOUT_S_DEFAULT
    )
)


class AtlasEntityModel:
    """Class for running inference on the Atlantes model for entity classification

    Rename?"""

    def __init__(self) -> None:
        """Typical place to load model weights/other large files"""
        self.inference_config = get_atlas_entity_inference_config()
        self.MODEL_ID = self.inference_config["model"]["ATLAS_ENTITY_MODEL_ID"]
        self.device = self._get_device()
        self.ATLAS_ENTITY_MODEL_PATH = self.get_model_path()
        self.data_config = self.inference_config["data"]
        self.model_config = self.inference_config["hyperparameters"]
        self.atlas_entity_model = self._load_atlas_entity_model()
        self.atlas_entity_model.to(self.device)
        self.atlas_entity_model.eval()

    def get_model_path(self) -> Path:
        """Get the path to the model"""
        entity_model_dir = os.environ.get("ENTITY_MODEL_DIR", MODELS_DIR_DEFAULT)
        atlas_entity_model_path = Path(entity_model_dir, self.MODEL_ID)
        return atlas_entity_model_path

    def _get_device(self) -> torch.device:
        """Get the device to run the model on"""
        if torch.cuda.is_available():
            logger.info("GPU detected, using GPU")
            device = torch.device("cuda:0")

        else:
            logger.info("No GPU detected, using CPU")
            device = torch.device("cpu")
            logger.info("No GPUs available for ray serve.")
        return device

    def _initialize_atlas_entity_model_inference(self) -> AtlasEntity:
        """Initialize the ATLAS entity model for inference."""
        n_classes = len(AtlasEntityLabelsTraining.to_label_name_dict())
        return AtlasEntity(
            c_in=len(self.data_config["MODEL_INPUT_COLUMNS_ENTITY"]),
            c_out=n_classes,
            transformer_layers_pre_squeeze=self.model_config[
                "N_PRE_SQUEEZE_TRANSFORMER_LAYERS"
            ],
            n_heads=self.model_config["N_HEADS"],
            token_dim=self.model_config["TOKEN_DIM"],
            mlp_dim=self.model_config["MLP_DIM"],
            cpe_layers=self.model_config["CPE_LAYERS"],
            cpe_kernel_size=self.data_config["CPE_KERNEL_SIZE"],
            cnn_layers=self.model_config["CNN_LAYERS"],
            cnn_kernel_size=self.model_config["CNN_KERNEL_SIZE"],
            use_residual_cnn=self.model_config["USE_RESIDUAL_CNN"],
            use_layer_norm_cnn=self.model_config["USE_LAYERNORM_CNN"],
            use_channel_dim_only_layernorm_cnn=self.model_config[
                "USE_CHANNEL_DIM_LN_CNN"
            ],
            dropout_p=self.model_config["DROPOUT_P"],
            qkv_bias=self.model_config["QKV_BIAS"],
            use_binned_ship_type=self.model_config["USE_SHIP_TYPE"],
        )

    def _load_atlas_entity_model(self) -> AtlasEntity:
        """_summary_"""
        model: AtlasEntity = self._initialize_atlas_entity_model_inference()
        atlas_entity_state_dict = torch.load(
            self.ATLAS_ENTITY_MODEL_PATH, map_location=self.device
        )
        single_gpu_state_dict = remove_module_from_state_dict(atlas_entity_state_dict)
        # TODO: TEMP CODE TO REMOVE UNUSED sqztransformer
        single_gpu_state_dict = {
            k: v
            for k, v in single_gpu_state_dict.items()
            if ("sqztransformer" not in k) and ("subpath_output_layer" not in k)
        }
        model.load_state_dict(single_gpu_state_dict)
        logger.debug(f"Number of parameters in model: {model.param_num()}")
        logger.debug(f"model memory size (MB): {model.param_num() * 4 / 1024 ** 2}")
        return model

    def run_inference(
        self,
        preprocessed_data_stream: list[PreprocessedEntityData],
    ) -> list[EntityPostprocessorInput]:
        """Run inference on the preprocessed data using the model"""
        # Move model to device
        preprocessed_data_dict_stream = [
            preprocessed_data.model_dump()
            for preprocessed_data in preprocessed_data_stream
        ]
        preprocessed_data = ais_collate_entity_class_with_subpaths(
            preprocessed_data_dict_stream
        )
        if not isinstance(preprocessed_data, EntityDatasetCollatedDataOutput):
            raise ValueError(
                f"collated_data is not of type EntityDatasetCollatedDataOutput: {preprocessed_data=}"
            )

        inputs = preprocessed_data.input_tensor.to(self.device)
        spatiotemporal_intervals = preprocessed_data.spatiotemporal_interval_tensor.to(
            self.device
        )
        padding_mask = preprocessed_data.padding_mask.to(self.device)
        binned_ship_type = preprocessed_data.binned_ship_type_tensor.to(self.device)
        metadata = preprocessed_data.metadata
        # Perform inference
        with torch.no_grad():
            entity_class_outputs = self.atlas_entity_model(
                inputs=inputs,
                spatiotemporal_tensor=spatiotemporal_intervals,
                binned_ship_type=binned_ship_type,
                padding_mask=padding_mask,
            )
        if entity_class_outputs.dim() < 2:
            entity_class_outputs = entity_class_outputs.unsqueeze(0)
        batch_size = entity_class_outputs.shape[0]
        entity_class_outputs = entity_class_outputs.detach().cpu()
        softmax = F.softmax(entity_class_outputs, dim=1)

        max_prob_values, max_prob_indices = torch.max(softmax, dim=1)
        # I need to loop through the outputs and get the class and the details for each
        output_tuple_lst = []
        for i in range(batch_size):
            entity_class = AtlasEntityLabelsTrainingWithUnknown(
                max_prob_indices[i].item()
            )
            details = EntityPostprocessorInputDetails(
                model=self.MODEL_ID,
                confidence=float(max_prob_values[i].item()),
                outputs=[float(prob) for prob in softmax[i]],
            )
            metadata_i = EntityMetadata(**metadata[i])
            output_tuple_lst.append(
                EntityPostprocessorInput(
                    predicted_class=entity_class,
                    entity_classification_details=details,
                    metadata=metadata_i,
                )
            )
        return output_tuple_lst


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": NUM_CPUS,
        "num_gpus": NUM_GPUS,
    },
)
class AtlasEntityModelDeployment(AtlasEntityModel):
    """Class for Deployment of the Atlas Entity Model on Ray Serve"""

    def __init__(self) -> None:
        super().__init__()

    @serve.batch(
        max_batch_size=ATLAS_ENTITY_BATCH_MAX_SIZE,
        batch_wait_timeout_s=ATLAS_ENTITY_BATCH_WAIT_TIMEOUT_S,
    )
    async def batch_handler_run_inference(
        self, preprocessed_data_stream: list[PreprocessedEntityData]
    ) -> list[EntityPostprocessorInput]:
        logger.info(
            f"Running batch inference on the ATLAS Entity model with {len(preprocessed_data_stream)} items"
        )
        return self.run_inference(preprocessed_data_stream)

    def reconfigure(self, user_config: dict) -> None:
        """Reconfigure the entity modelwith a new user config

        Is called and updated when the serve config is updated and application is redeployed

        Parameters
        ----------
        user_config : dict
            The new user config to update different parameters of the deployment"""
        self.batch_handler_run_inference.set_max_batch_size(
            user_config["max_batch_size"]
        )
        self.batch_handler_run_inference.set_batch_wait_timeout_s(
            user_config["batch_wait_timeout_s"]
        )
