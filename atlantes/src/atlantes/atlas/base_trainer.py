"""Module for the base trainer class for ais models in the atlantes system"""

import os
import signal
import sys
import traceback
from abc import abstractmethod
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from atlantes.atlas.atlas_utils import remove_module_from_state_dict
from atlantes.atlas.training_utils import (OPTIMIZER_CLASSES,
                                           SCHEDULER_CLASSES,
                                           DistributedWeightedSampler,
                                           rank_zero_only)
from atlantes.datautils import DATE_FORMAT
from atlantes.log_utils import get_logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.optim.lr_scheduler import _LRScheduler as LR
from torch.utils.data import (DataLoader, Dataset, DistributedSampler, Subset,
                              WeightedRandomSampler)

from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
LABEL_PADDING_VALUE = (
    -100
)  # Used to signify to pytorch we shouldn't train on given example


class BaseAtlasTrainer:
    """Base Trainer abstraction for training the Atlas models

    call after distributed training is init
    if NUM_GPUS > 1:
        world_size = NUM_GPUS
        mp.spawn(setup_and_train_model, args=(world_size,), nprocs=world_size)
    else:
        rank = 0
        world_size = 1
        setup_and_train_model(rank, world_size)

        so we will still nead a laucnh function that sets up
        a trainer class in each process

    setup is called once the split happens"""

    def __init__(
        self,
        data_config: dict,
        experimental_config_train: dict,
        experimental_config_model: dict,
    ):
        self.rank = 0
        self.world_size = 1
        self.num_gpus = torch.cuda.device_count()
        self.collate_fn: Optional[Callable] = None

        self.data_config = data_config
        self.data_config["LABEL_PADDING_VALUE"] = LABEL_PADDING_VALUE
        self.data_config["DATE_FORMAT"] = DATE_FORMAT
        self.experimental_config_train = experimental_config_train
        self.experimental_config_model = experimental_config_model
        self._base_unpack_experiment_config_train()
        self.main_process_rank = 0
        self.model = None
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        # TODO: implement function that sets this to work more generally
        self.multiprocessing_context: Optional[str] = (
            "forkserver" if self.num_data_workers > 0 else None
        )
        self.wandb_config = {
            **self.data_config,
            **self.experimental_config_train,
            **self.experimental_config_model,
        }
        self.dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.git_hash = self._get_git_hash()
        self.dist_multiprocessing_timeout = timedelta(milliseconds=54000000)  # 15 hours

    def _base_unpack_experiment_config_train(self) -> None:
        """Unpack the experimental config for training"""
        self.project_name = self.experimental_config_train["PROJECT_NAME"]
        self.experiment_name = self.experimental_config_train.get("EXPERIMENT_NAME")
        if not self.experiment_name:  # This checks for None or an empty string
            self.experiment_name = wandb.util.generate_id()
        self.model_save_dir = self.experimental_config_train["MODEL_SAVE_DIR"]
        self.model_load_dir = self.experimental_config_train["MODEL_LOAD_DIR"]
        self.train_batch_size = self.experimental_config_train["TRAIN_BATCH_SIZE"]
        self.val_batch_size = self.experimental_config_train["VAL_BATCH_SIZE"]
        self.learning_rate = self.experimental_config_train["LEARNING_RATE"]
        self.sgd_momentum = self.experimental_config_train["SGD_MOMENTUM"]
        self.weight_decay = self.experimental_config_train["WEIGHT_DECAY"]
        self.max_grad_norm = self.experimental_config_train["MAX_GRAD_NORM"]
        self.optimizer_key = self.experimental_config_train["OPTIMIZER"]
        self.scheduler_key = self.experimental_config_train["LR_SCHEDULER"]
        self.annealing_tmax = self.experimental_config_train["ANNEALING_TMAX"]
        self.eta_min = self.experimental_config_train["ETA_MIN"]
        self.warmup_steps = self.experimental_config_train["WARMUP_STEPS"]
        self.model_resume_checkpoint = self.experimental_config_train[
            "MODEL_CHECKPOINT"
        ]
        self.num_data_workers = self.experimental_config_train["NUM_DATA_WORKERS"]
        self.pin_memory = self.experimental_config_train["PIN_MEMORY"]
        self.timeout = (
            self.experimental_config_train["TIMEOUT"]
            if self.num_data_workers > 0
            else 0
        )
        self.use_cached_run_id = self.experimental_config_train["USE_CACHED_RUN_ID"]
        self.use_features_only = self.experimental_config_train["USE_FEATURES_ONLY"]
        self.freeze_feature_layers = self.experimental_config_train["FREEZE_FEATURE_LAYER"]
        self.use_lora = self.experimental_config_train.get("USE_LORA", False)
        self.lora_r = self.experimental_config_train.get("LORA_R", 8)
        self.lora_alpha = self.experimental_config_train.get("LORA_ALPHA", 16)
        self.lora_target_modules = self.experimental_config_train.get("LORA_TARGET_MODULES", ['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        self.lora_dropout = self.experimental_config_train.get("LORA_DROPOUT", 0.1)

    def signal_handler(self, sig: int, frame: Any) -> None:
        logger.info(f"program killed by signal: {sig}")
        wandb.join()
        sys.exit(0)

    def register_signal_handler(
        self,
    ) -> None:
        """Register the signal handler

        This is used to handle the case where the user interrupts the training
        process with a keyboard interrupt or a SIGTERM signal and still uploads to wandb
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def save_device_instance(self, rank: int) -> None:
        """Save the device instance to the wandb config"""
        self.device = (
            torch.device(f"cuda:{rank}") if torch.cuda.is_available() else "cpu"
        )

    def _get_and_cache_run_id(self) -> str:
        """Get the run id from the wandb API and cache it to a file

        if the file exists, read the run id from the file

        Returns
        -------
        str
            The run id for the current run"""
        run_id_file = Path(self.model_save_dir) / "run_id.txt"
        if run_id_file.exists():
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()
            run_id_file.parent.mkdir(parents=True, exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(run_id)
        return run_id

    def init_wandb(
        self,
    ) -> None:
        """Initialize wandb

        Parameters
        ----------
        rank : int
            Rank of the process
        world_size : int
            Number of processes, i.e. number of GPUs
        """
        if self.rank == self.main_process_rank:
            try:
                run_id = (
                    self._get_and_cache_run_id()
                    if self.use_cached_run_id
                    else wandb.util.generate_id()
                )

                wandb.init(
                    project=self.project_name,
                    dir=".wandb",
                    config=self.wandb_config,
                    id=run_id,
                    resume="allow",
                    name=self.experiment_name,
                )
                wandb.config = self.wandb_config
                rank_zero_only(logger.info, self.rank)(f"Wandb config: {wandb.config}")
            except Exception as e:
                logger.exception(e)
                wandb.init(project=self.project_name, mode="disabled")
                sys.exit(1)
        else:
            # Only init wandb on the main process
            pass

    def _get_git_hash(self) -> Optional[str]:
        """Get the git hash of the current repository"""
        try:
            import git
        except ImportError:
            logger.error("GitPython not installed. Skipping git hash retrieval.")
            return None
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:7]
        except git.InvalidGitRepositoryError:
            return None
        except Exception as e:
            logger.error(f"Error getting git hash: {e}")
            return None

    def save_model_wandb_artifact(
        self, model: nn.Module, epoch: int, step: Optional[int] = None
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
        # Add run name and epoch to model save path
        git_hash = self.git_hash or "no_git_hash"
        file_name = f"{self.project_name}_{git_hash}_{self.dt_string}_epoch{epoch}.pt"
        if step is not None:
            # remove .pt extension
            file_name = file_name[:-3]
            file_name = f"{file_name}_step{step}.pt"
        logger.info(f" current dir: {os.getcwd()}")
        os.makedirs(self.model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, file_name))
        artifact = wandb.Artifact("model_weights", type="model")
        artifact.add_file(os.path.join(self.model_save_dir, file_name))
        logger.info(f"Saving model weights to wandb artifact {file_name}")
        wandb.log_artifact(artifact)

    def setup_distributed(self) -> None:
        """Setup function for distributed training"""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # TImeout is set to allow for longer and more comprehensive evaluation
        dist.init_process_group(
            "nccl",
            rank=self.rank,
            world_size=self.world_size,
            timeout=self.dist_multiprocessing_timeout,
        )

    def _create_ddp_model(
        self, model: nn.Module, find_unused_params: bool = True
    ) -> nn.Module:
        """Create a distributed data parallel model"""
        # move moodel to GPU with id rank
        model = model.to(self.rank)
        # create distributed data parallel model
        model = DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=find_unused_params,
        )
        return model

    def move_model_to_gpu(self, model: nn.Module) -> nn.Module:
        """Move the model to the GPU, and transform to DDP if necessary"""
        if self.world_size > 1:
            model = self._create_ddp_model(model)
        else:
            model = model.to(self.device)
        return model

    def cleanup(self) -> None:
        """Cleanup function for distributed training"""
        dist.destroy_process_group()

    def warmup(self, current_step: int, warmup_steps: int) -> float:
        """Warmup function for the learning rate scheduler"""
        # make a linear increase as we approach warmup steps
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    def get_cross_entropy_class_weights(
        self, targets: torch.Tensor, num_classes: int, padding_value: float
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

    def _get_binary_weighted_sampler(
        self, targets: np.ndarray, indices: np.ndarray, total_samples: int
    ) -> WeightedRandomSampler:
        """Make a binary weighted sampler from a set of indices"""
        class_probs_train = 1.0 / np.bincount(targets[indices])
        class_sampling_weights_train = class_probs_train[targets[indices]]
        return WeightedRandomSampler(
            class_sampling_weights_train, total_samples, replacement=True
        )

    def get_weighted_sampler_train(
        self, train_split: Subset
    ) -> Optional[Union[DistributedWeightedSampler, WeightedRandomSampler]]:
        """Setup the weighted samplers for the train and validation datasets"""
        logger.warning("Train weighted sampler not implemented")
        return None

    def get_weighted_sampler_val(
        self, val_split: Subset
    ) -> Optional[WeightedRandomSampler]:
        """Setup the weighted samplers for the validation datasets

        Validation is always done on a single GPU"""
        logger.warning("Validation weighted sampler not implemented")
        return None

    def initialize_data_loader_train(self, train_dataset: Subset) -> DataLoader:
        """Initialize the data loader"""
        sampler_train = (
            DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
            if self.num_gpus > 1
            else None
        )
        return DataLoader(
            train_dataset,
            batch_size=self.per_device_batch_size,  # TRAIN_BATCH_SIZE // NUM_GPUS,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=self.pin_memory,
            sampler=sampler_train,
            num_workers=self.num_data_workers,
            timeout=self.timeout,
            multiprocessing_context=self.multiprocessing_context,
        )

    def _set_dataset_mode_within_dataloader(
        self, data_loader: DataLoader, mode: str
    ) -> None:
        """Set the dataset mode within the data loader"""
        # Normal Pytorch Dataset
        if hasattr(data_loader.dataset, "mode"):
            data_loader.dataset.mode = mode
        # Subset of a Pytorch Dataset
        elif hasattr(data_loader.dataset.dataset, "mode"):
            data_loader.dataset.dataset.mode = mode
        else:
            logger.error("Dataset mode not set", exc_info=True)
            raise ValueError("Dataset mode not set")

    def set_dataset_based_on_model_mode(
        self, data_loader: DataLoader, model: nn.Module
    ) -> None:
        """Set the dataset mode based on if the model is in training or evaluation mode"""
        if model.training:
            self._set_dataset_mode_within_dataloader(data_loader, "train")
        else:
            self._set_dataset_mode_within_dataloader(data_loader, "eval")

    def initialize_data_loader_val(self, val_dataset: Subset) -> DataLoader:
        """Initialize the data loader"""
        weighted_sampler_val = None

        return DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=self.pin_memory,
            sampler=weighted_sampler_val,
            num_workers=self.num_data_workers,
            timeout=self.timeout,
            multiprocessing_context=self.multiprocessing_context,
        )

    def set_optimizer(self, model: nn.Module) -> None:
        """Sets the optimizer

        To add more hyperparameters for different optimizers, add them here
        Parameters
        ----------
        model : nn.Module
            The model to optimize and get parameters from

        Returns
        -------
        None
            Sets the optimizer as an attribute of the class"""
        if self.optimizer_key not in OPTIMIZER_CLASSES:
            raise ValueError(f"Optimizer {self.optimizer_key} not supported")
        optimizer_class = OPTIMIZER_CLASSES[self.optimizer_key]

        # Filter only parameters that require gradients
        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

        if optimizer_class == SGD:
            optimizer = optimizer_class(
                trainable_parameters,
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
            )
        elif optimizer_class == Adam:
            optimizer = optimizer_class(
                trainable_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_class == AdamW:
            optimizer = optimizer_class(
                trainable_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        self.optimizer = optimizer
        if not hasattr(self, "optimizer"):
            raise ValueError(
                f"Optimizer class {optimizer_class} is not handled properly"
            )

    def add_warmup(self, scheduler: LR) -> LR:
        """Add a warmup to the scheduler"""
        warmup_scheduler = LambdaLR(
            self.optimizer, lr_lambda=partial(self.warmup, self.warmup_steps)
        )
        scheduler = SequentialLR(
            self.optimizer, [warmup_scheduler, scheduler], [self.warmup_steps]
        )
        return scheduler

    def set_scheduler(
        self,
    ) -> None:
        """Get the scheduler

        To add more hyperparameters for different schedulers, add them here
        Parameters
        ----------
        optimizer : torch.optim
            The optimizer to use

        Returns
        -------
        None
            sets the scheduler as an attribute of the class"""
        if self.scheduler_key not in SCHEDULER_CLASSES:
            raise ValueError(f"Scheduler {self.scheduler_key} not supported")
        scheduler_class = SCHEDULER_CLASSES[self.scheduler_key]
        if scheduler_class is None:
            self.scheduler = None
            return None
        elif scheduler_class == CosineAnnealingLR:
            scheduler = scheduler_class(
                self.optimizer,
                T_max=self.annealing_tmax,
                eta_min=self.eta_min,
                verbose=False,
            )

        if self.warmup_steps:
            scheduler = self.add_warmup(scheduler)
        self.scheduler = scheduler

    def assert_device_is_gpu_if_gpu_requested(self) -> None:
        """Assert that the device is a GPU if the user requested it"""
        if self.num_gpus > 0:
            assert self.device != "cpu"

    def set_per_device_batch_size(self) -> None:
        """Set the per device batch size"""
        self.per_device_batch_size = (
            self.train_batch_size // self.world_size
            if self.num_gpus >= 1
            else self.train_batch_size
        )

    def setup(self, rank: int, world_size: int) -> None:
        """Setup function for the distributed training

        Parameters
        ----------
        rank : int
            Rank of the process
        world_size : int
            Number of processes, i.e. number of GPUs
        """
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == self.num_gpus
        self.set_per_device_batch_size()
        self.save_device_instance(rank)
        self.assert_device_is_gpu_if_gpu_requested()
        logger.info(
            f"Device {self.device} rank {self.rank} world size {self.world_size}"
        )
        self.init_wandb()
        self.register_signal_handler()

    def apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA configuration to the model if `use_lora` is enabled"""
        if self.use_lora:
            logger.info("Applying LoRA configuration")
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none"
            )
            model = get_peft_model(model, lora_config)
        return model

    def load_from_checkpoint(
        self,
    ) -> torch.nn.Module:
        """Load the model from the given path"""
        logger.info(f"Loading model from checkpoint {self.model_resume_checkpoint}")
        models_dir = Path(self.model_load_dir).resolve()
        model = self.initialize_model()
        model = self.apply_lora(model)

        model_state_dict = torch.load(
            f"{models_dir}/{self.model_resume_checkpoint}", map_location=self.device
        )
        strict = True

        if self.use_features_only:
            logger.info("Using only feature extractor from the model checkpoint")
            for key in list(model_state_dict.keys()):
                if "output_layer" in key:
                    del model_state_dict[key]
            if self.freeze_feature_layers:
                logger.info("Freezing all layers except the output layer")
                for name, param in model.named_parameters():
                    if "output_layer" not in name:
                        param.requires_grad = False

        if self.use_lora or self.use_features_only:
            strict = False
        try:
            # Here is where we could update the name of the model
            single_gpu_state_dict = remove_module_from_state_dict(model_state_dict)
            # TODO: TEMP CODE TO REMOVE UNUSED sqztransformer
            single_gpu_state_dict = {
                k: v
                for k, v in single_gpu_state_dict.items()
                if "sqztransformer" not in k
            }
            model.load_state_dict(single_gpu_state_dict, strict=strict)

        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            logger.error(
                "Ensure the correct model architecture hyperparameter configs "
                "are set in the config file."
            )
            traceback.print_exc()
            sys.exit(0)

        return model

    @abstractmethod
    def _initialize_dataset(self, labels_file: Any) -> Dataset:
        pass

    @abstractmethod
    def _split_train_val(self, full_dataset: Dataset) -> Tuple[Subset, Subset]:
        pass

    @abstractmethod
    def initialize_model(self) -> nn.Module:
       pass

    @abstractmethod
    def _log_model_architecture_info(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def _train(
        self,
        model: nn.Module,
        train_batches: DataLoader,
        val_batches: DataLoader,
    ) -> None:
        pass

    # add all the distributed to this if it can be configured abstractly
    @abstractmethod
    def setup_and_train(self, rank: int, world_size: int) -> None:
        """Setup and train function for the distributed training


        Parameters
        ----------
        rank : int
            Rank of the process
        world_size : int
            Number of processes, i.e. number of GPUs
        """
        pass
