"""Main model inference module for the atlas activity application

Loads the model and runs inference on the data
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from atlantes.atlas.atlas_net import AtlasActivityEndOfSequenceTaskNet
from atlantes.atlas.atlas_utils import (
    AtlasActivityLabelsTraining,
    get_atlas_activity_inference_config,
    remove_module_from_state_dict,
)
from atlantes.atlas.collate import (
    ActivityDatasetEndOfSequenceCollatedDataOutput,
    RealTimeActivityCollater,
)
from atlantes.inference.atlas_activity.datamodels import PreprocessedActivityData
from atlantes.log_utils import get_logger
from atlantes.utils import get_commit_hash

logger = get_logger("atlas_activity_classifier")

# Only matters for cI
if torch.cuda.is_available():
    NUM_GPUS = 1
else:
    # logger.info("No GPU detected, using CPU, module level log")
    NUM_GPUS = 0

NUM_CPUS = 1
NUM_REPLICAS = 1
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


GIT_HASH = get_commit_hash()
MODEL_VERSION = f"{GIT_HASH}_{CURRENT_TIME}"

MODELS_DIR_DEFAULT = str(Path(__file__).parents[2] / "models")

ATLAS_ACTIVITY_BATCH_MAX_SIZE_DEFAULT = 4
ATLAS_ACTIVITY_BATCH_MAX_SIZE = int(
    os.environ.get(
        "ATLAS_ACTIVITY_BATCH_MAX_SIZE", ATLAS_ACTIVITY_BATCH_MAX_SIZE_DEFAULT
    )
)
ATLAS_ACTIVITY_BATCH_WAIT_TIMEOUT_S_DEFAULT = 0.5  # Set Higher for testing
ATLAS_ACTIVITY_BATCH_WAIT_TIMEOUT_S = float(
    os.environ.get(
        "ATLAS_ACTIVITY_BATCH_WAIT_TIMEOUT_S",
        ATLAS_ACTIVITY_BATCH_WAIT_TIMEOUT_S_DEFAULT,
    )
)


class AtlasActivityModel:
    """Class for running inference on the Atlantes model for activity classification"""

    def __init__(
        self, device: int = 0, inference_config: Optional[dict] = None
    ) -> None:
        """Initialize the Atlas Activity Model

        If no inference config is provided, the default config is used
        Parameters
        ----------
        device : int, optional
            The device to run the model on, by default 0
        inference_config : Optional[dict], optional
            The inference config to use, by default None
        """
        self.inference_config = (
            inference_config or get_atlas_activity_inference_config()
        )
        self.MODEL_ID = self.inference_config["model"]["ATLAS_ACTIVITY_MODEL_ID"]
        self.ATLAS_ACTIVITY_MODEL_PATH = self.get_model_path()
        self.device = self._get_device(device)
        self.data_config = self.inference_config["data"]
        self.model_config = self.inference_config["hyperparameters"]
        self.atlas_activity_model = self._load_atlas_activity_model()
        self.atlas_activity_model.to(self.device)
        self.atlas_activity_model.eval()
        self.collator = RealTimeActivityCollater(
            use_prepadding=self.model_config["USE_PREPAD"]
        )

    def get_model_path(
        self,
    ) -> Path:
        """Get the path to the model"""
        activity_model_dir = os.environ.get("ACTIVITY_MODEL_DIR", MODELS_DIR_DEFAULT)
        atlas_activity_model_path = Path(activity_model_dir, self.MODEL_ID)
        return atlas_activity_model_path

    def _load_atlas_activity_model(self) -> AtlasActivityEndOfSequenceTaskNet:
        """Load the ATLAS activity model"""
        model = self._initialize_atlas_activity_model_inference()
        atlas_activity_state_dict = torch.load(
            self.ATLAS_ACTIVITY_MODEL_PATH, map_location=self.device
        )
        # Here is where we could update the name of the model
        single_gpu_state_dict = remove_module_from_state_dict(atlas_activity_state_dict)
        model.load_state_dict(single_gpu_state_dict)
        logger.debug(f"Number of parameters in model: {model.param_num()}")
        logger.debug(f"model memory size (MB): {model.param_num() * 4 / 1024**2}")
        return model

    # Correctly manage device placement
    def _get_device(self, device: int = 0) -> torch.device:
        """Get the device to run the model on"""
        if torch.cuda.is_available():
            logger.info("GPU detected, using GPU")
            # Device count what is that
            device = torch.device(f"cuda:{device}")

        else:
            logger.info("No GPU detected, using CPU")
            device = torch.device("cpu")
        return device

    def _initialize_atlas_activity_model_inference(
        self,
    ) -> AtlasActivityEndOfSequenceTaskNet:
        """Initialize the ATLAS activity model for inference"""
        return AtlasActivityEndOfSequenceTaskNet(
            c_in=len(self.data_config["MODEL_INPUT_COLUMNS_ACTIVITY"]),
            subpath_output_dim=len(AtlasActivityLabelsTraining),
            transformer_layers_pre_squeeze=self.model_config[
                "N_PRE_SQUEEZE_TRANSFORMER_LAYERS"
            ],
            n_heads=self.model_config["N_HEADS"],
            token_dim=self.model_config["TOKEN_DIM"],
            mlp_dim=self.model_config["MLP_DIM"],
            cpe_kernel_size=self.data_config["CPE_KERNEL_SIZE"],
            cpe_layers=self.model_config["CPE_LAYERS"],
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

    def run_inference(
        self,
        preprocessed_data_stream: list[PreprocessedActivityData],
    ) -> list[tuple[AtlasActivityLabelsTraining, dict, dict]]:
        """Run inference on the preprocessed data using the model"""
        logger.info("Running inference on the ATLAS activity model")
        preprocessed_data_dict_stream = [
            data.model_dump() for data in preprocessed_data_stream
        ]
        collated_data = self.collator(preprocessed_data_dict_stream)
        if not isinstance(
            collated_data, ActivityDatasetEndOfSequenceCollatedDataOutput
        ):
            raise ValueError(
                f"collated_data is not of type \
                    ActivityDatasetEndOfSequenceCollatedDataOutput: {collated_data=}"
            )
        batch_size = len(collated_data.metadata)
        inputs = collated_data.input_tensor.to(self.device)
        spatiotemporal_intervals = collated_data.spatiotemporal_interval_tensor.to(
            self.device
        )
        padding_mask = collated_data.padding_mask.to(self.device)
        binned_ship_type = collated_data.binned_ship_type_tensor.to(self.device)
        # Perform inference
        with torch.no_grad():
            activity_class_outputs = self.atlas_activity_model(
                inputs,
                spatiotemporal_intervals,
                binned_ship_type,
                padding_mask=padding_mask,
            )
        # Need to deal with padding and unbatching
        if activity_class_outputs.dim() < 2:
            activity_class_outputs = activity_class_outputs.unsqueeze(0)
        probs = torch.nn.functional.softmax(activity_class_outputs, dim=1).cpu()
        if torch.isnan(probs).any().item():
            raise ValueError(
                f"NaN values detected in probabilities. for {probs=} \
                             {activity_class_outputs=} \
                             {inputs=} \
                                {spatiotemporal_intervals=} \
                                {padding_mask=} \
                                {binned_ship_type=}"
            )
        max_prob_values, top_pred_ids = torch.max(probs, dim=1)
        max_prob_values = max_prob_values.cpu().numpy()
        top_pred_ids = top_pred_ids.cpu().numpy()
        class_ouputs_with_details_batch = [
            (
                AtlasActivityLabelsTraining(top_pred_ids[i]),
                dict(
                    model=self.MODEL_ID,
                    confidence=float(max_prob_values[i]),
                    outputs=[float(prob.item()) for prob in probs[i]],
                    model_version=MODEL_VERSION,
                ),
                collated_data.metadata[i],
            )
            for i in range(batch_size)
        ]

        return class_ouputs_with_details_batch


'''

class AtlasActivityModelDeployment(AtlasActivityModel):
    """Class for Deployment of the Atlas Activity Model on Ray Serve"""

    def __init__(self) -> None:
        super().__init__()
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize metrics for the model"""
        self.activity_inference_batch_size = metrics.Histogram(
            "activity_inference_batch_size",
            description="Batch size for activity inference",
            boundaries=[1, 2, 4, 8, 16, 32],
        )

    async def batch_handler_run_inference(
        self, preprocessed_data_stream: list[PreprocessedActivityData]
    ) -> list[tuple[AtlasActivityLabelsTraining, dict, dict]]:
        batch_size = len(preprocessed_data_stream)
        logger.info(
            f"Running batch inference on the ATLAS activity model with {batch_size} items"
        )
        self.activity_inference_batch_size.observe(batch_size)
        return self.run_inference(preprocessed_data_stream)

    def reconfigure(self, user_config: dict) -> None:
        """Reconfigure the activity model with a new user config

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

'''