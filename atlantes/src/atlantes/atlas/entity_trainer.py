"""Enitty model Trainer class for training the entity model

#TODO fix debug mode for entity model because it implements dataloader from base trainer
# Debug mode also laucnhes like 4 wandb """

import traceback
from copy import copy
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from atlantes.atlas.ais_dataset import AISTrajectoryEntityDataset
from atlantes.atlas.atlas_net import AtlasEntity
from atlantes.atlas.atlas_utils import log_val_predictions_entity_type
from atlantes.atlas.base_trainer import BaseAtlasTrainer
from atlantes.atlas.collate import (EntityDatasetCollatedDataOutput,
                                    ais_collate_entity_class_with_subpaths)
from atlantes.atlas.training_utils import (DistributedWeightedSampler,
                                           rank_zero_only)
from atlantes.log_utils import get_logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

logger = get_logger(__name__)
MODELS_DIR_DEFAULT = str(Path(__file__).parent.parent / "models")


class AtlasEntityTrainer(BaseAtlasTrainer):
    """Entity model Trainer class for training the entity model"""

    def __init__(
        self,
        data_config: dict,
        experimental_config_train: dict,
        experimental_config_model: dict,
        label_enum: Any,
    ):
        super().__init__(
            data_config, experimental_config_train, experimental_config_model
        )
        self._unpack_experimental_config_train()
        self.collate_fn = ais_collate_entity_class_with_subpaths
        self.label_enum = label_enum
        self.n_classes = len(label_enum)

    def _unpack_experimental_config_train(
        self,
    ) -> None:
        """Unpack and set the experimental configuration as class attributes


        It loads all the experimental configs that are not loaded in the base class
        This method should be called after the experimental configuration is loaded
        and before training starts. See the .yaml file for more details
        """
        self.val_size = self.experimental_config_train["VAL_SIZE"] / 100
        self.n_epochs = self.experimental_config_train["N_EPOCHS"]
        self.max_num_batches_to_log = self.experimental_config_train[
            "MAX_NUM_BATCHES_TO_LOG"
        ]
        self.max_num_batches_to_log_images = self.experimental_config_train[
            "MAX_NUM_BATCHES_TO_LOG_IMAGES"
        ]
        self.random_state = self.experimental_config_train["RANDOM_STATE"]
        self.model = self.experimental_config_train["MODEL_ARCHITECTURE"]
        self.sweep = self.experimental_config_train["SWEEP"]
        self.profiling = self.experimental_config_train["PROFILING"]
        self.entity_class_labels_path = self.experimental_config_train[
            "ENTITY_CLASS_LABELS_PATH"
        ]
        self.eval_cadence = self.experimental_config_train["EVAL_CADENCE"]
        self.num_gpus = self.experimental_config_train["NUM_GPUS"]
        self.debug_mode = self.experimental_config_train["DEBUG_MODE"]
        self.eval_before_train = self.experimental_config_train["EVAL_BEFORE_TRAIN"]
        self.use_class_weighted_loss = self.experimental_config_train[
            "USE_CLASS_WEIGHTED_LOSS"
        ]
        self.eval_epoch_cadence = self.experimental_config_train["EVAL_EPOCH_CADENCE"]

        if self.debug_mode:
            logger.warning("Entering Debug Mode, see experimental config if unexpected")
            if torch.cuda.is_available():
                self.num_gpus = 1
            else:
                raise (ValueError("No GPU available, quitting"))
            self.train_batch_size = 10
            self.val_batch_size = 10
            self.rank = 0
            self.world_size = 1
            self.skip_train = True
            self.project_name = self.project_name + "_DEBUG"
            self.num_data_workers = 0
            self.profiling = True

            torch.autograd.set_detect_anomaly(False)

    def _initialize_dataset(self, labels: Any) -> AISTrajectoryEntityDataset:
        # TODO: add the option to do a random transform on the data
        dataset = AISTrajectoryEntityDataset(
            entity_labels_path=labels,
            dataset_config=self.data_config,
            label_enum=self.label_enum,
        )
        self.class_descriptions = dataset.class_descriptions
        self.dataset_len = len(dataset)
        self.dataset_targets = dataset.targets
        return dataset

    def _split_train_val(
        self, dataset: AISTrajectoryEntityDataset
    ) -> Tuple[Subset, Subset]:
        # TODO: make stratify configurable
        train_indices, val_indices, _, _ = train_test_split(
            range(self.dataset_len),
            dataset.targets,
            stratify=dataset.targets,
            test_size=self.val_size,
            random_state=self.random_state,
        )

        train_split = Subset(dataset, train_indices)
        val_split = Subset(dataset, val_indices)
        return train_split, val_split

    def get_weighted_sampler_train(
        self, train_split: Subset
    ) -> Optional[Union[DistributedWeightedSampler, WeightedRandomSampler]]:
        """Setup the weighted samplers for the train and validation datasets"""
        if self.world_size > 1:
            weighted_sampler_train = DistributedWeightedSampler(
                train_split,
                self.dataset_targets,
            )
        else:
            weighted_sampler_train = self._get_binary_weighted_sampler(
                self.dataset_targets, train_split.indices, len(train_split)
            )
        return weighted_sampler_train

    def get_weighted_sampler_val(
        self, val_split: Subset
    ) -> Optional[WeightedRandomSampler]:
        """Setup the weighted samplers for the validation datasets

        Validation is always done on a single GPU"""
        weighted_sampler_val = self._get_binary_weighted_sampler(
            self.dataset_targets, val_split.indices, len(val_split)
        )
        return weighted_sampler_val

    def initialize_model(self) -> AtlasEntity:
        return AtlasEntity(
            c_in=len(self.data_config["MODEL_INPUT_COLUMNS_ENTITY"]),
            c_out=self.n_classes,
            transformer_layers_pre_squeeze=self.experimental_config_model[
                "N_PRE_SQUEEZE_TRANSFORMER_LAYERS"
            ],
            n_heads=self.experimental_config_model["N_HEADS"],
            token_dim=self.experimental_config_model["TOKEN_DIM"],
            mlp_dim=self.experimental_config_model["MLP_DIM"],
            cpe_layers=self.experimental_config_model["CPE_LAYERS"],
            cpe_kernel_size=self.data_config["CPE_KERNEL_SIZE"],
            cnn_layers=self.experimental_config_model["CNN_LAYERS"],
            cnn_kernel_size=self.experimental_config_model["CNN_KERNEL_SIZE"],
            use_residual_cnn=self.experimental_config_model["USE_RESIDUAL_CNN"],
            use_layer_norm_cnn=self.experimental_config_model["USE_LAYERNORM_CNN"],
            use_channel_dim_only_layernorm_cnn=self.experimental_config_model["USE_CHANNEL_DIM_LN_CNN"],
            dropout_p=self.experimental_config_model["DROPOUT_P"],
            qkv_bias=self.experimental_config_model["QKV_BIAS"],
            use_binned_ship_type=self.experimental_config_model["USE_SHIP_TYPE"],
            use_lora=self.experimental_config_train.get("USE_LORA", False)
        )

    def _log_model_architecture_info(self, model: nn.Module) -> None:
        """Log the model architecture info"""
        logger.info(
            f"Training on {self.n_classes} total classes: {self.class_descriptions}"
        )
        logger.info(f"Total # of trajectories in dataset: {self.dataset_len}")
        logger.info(f"Training on {self.world_size} GPUs")
        if self.world_size > 1:
            logger.info(f"Model Parameters: {model.module.param_num()}")
            logger.info(f"Model Architecture: {model.module.__str__}")
        else:
            logger.info(f"Model Architecture: {model.__str__}")
            logger.info(f"Model Parameters: {model.param_num()}")
        logger.info(f" N Epochs : {self.n_epochs}")
        logger.info(f"Train Batch size per device{self.per_device_batch_size}")
        logger.info(f"Val Batch size{self.val_batch_size}")

    def _get_loss(
        self,
        entity_class_outputs: torch.Tensor,
        entity_class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Get the loss"""
        if self.use_class_weighted_loss:
            if not isinstance(self.entity_class_criterion, nn.CrossEntropyLoss):
                raise ValueError("Criterion must be nn.CrossEntropyLoss to set weights")
            weights = self.get_cross_entropy_class_weights(
                entity_class_labels,
                entity_class_outputs.shape[1],
                self.data_config["LABEL_PADDING_VALUE"],
            )
            weights = weights.to(entity_class_outputs.device)
            self.entity_class_criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
                weight=weights, ignore_index=self.data_config["LABEL_PADDING_VALUE"]
            )
        entity_class_loss = self.entity_class_criterion(
            entity_class_outputs, entity_class_labels
        )
        return entity_class_loss

    def _train_step(
        self, model: nn.Module, train_batch: EntityDatasetCollatedDataOutput
    ) -> float:
        """Perform a single training step"""

        inputs = train_batch.input_tensor.to(self.device)
        spatiotemporal_intervals = train_batch.spatiotemporal_interval_tensor.to(
            self.device
        )
        entity_class_labels = train_batch.entity_class_targets.to(self.device)
        padding_mask = train_batch.padding_mask.to(self.device)
        binned_ship_type_tensor = train_batch.binned_ship_type_tensor.to(self.device)

        entity_class_outputs = model(
            inputs,
            spatiotemporal_intervals,
            binned_ship_type_tensor,
            padding_mask=padding_mask,
        )
        loss = self._get_loss(
            entity_class_outputs,
            entity_class_labels,
        )
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        self.optimizer.zero_grad(set_to_none=True)
        # add gradient clipping here
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        rank_zero_only(wandb.log, rank=self.rank)({"Training  loss batch": loss.item()})
        return loss_val

    def _train_batch(
        self,
        model: nn.Module,
        train_batches: DataLoader,
        val_batches: DataLoader,
    ) -> float:
        """Perform a single training batch"""
        loss_accumulation = 0.0
        i = 0
        for batched_data in train_batches:
            loss = self._train_step(model, batched_data)
            loss_accumulation += loss  # change to detach
            rank_zero_only(logger.info, rank=self.rank)(f"Batch {i} Loss: {loss}")
            rank_zero_only(wandb.log, rank=self.rank)(
                {"Training loss total(batch)": loss}
            )
            if self.scheduler is not None:
                rank_zero_only(wandb.log, rank=self.rank)(
                    {"learning_rate": self.scheduler.get_last_lr()[0]}
                )
                self.scheduler.step()
            if i % self.eval_cadence == 0 and i != 0:
                rank_zero_only(self._eval, rank=self.rank)(
                    model,
                    val_batches,
                )
                rank_zero_only(self.save_model_wandb_artifact, rank=self.rank)(
                    model,
                )
            i += 1
        return loss_accumulation

    def _train(
        self,
        model: nn.Module,
        train_batches: DataLoader,
        val_batches: DataLoader,
    ) -> None:
        rank_zero_only(wandb.watch, rank=self.rank)(
            model, log_freq=100, log="all", log_graph=True
        )
        if self.eval_before_train:
            rank_zero_only(self._eval, rank=self.rank)(
                model,
                val_batches,
            )
        model.train()
        self.set_dataset_based_on_model_mode(train_batches, model)
        for epoch in range(self.n_epochs):
            rank_zero_only(logger.info, rank=self.rank)(f"Epoch {epoch}")
            # mini batching in the train loop
            loss_accumulation = self._train_batch(model, train_batches, val_batches)
            rank_zero_only(wandb.log, rank=self.rank)(
                {"training loss (epoch)": loss_accumulation}
            )
            is_last_epoch = epoch == self.n_epochs - 1
            if ((epoch % self.eval_epoch_cadence == 0) and epoch > 0) or is_last_epoch:
                rank_zero_only(self._eval, rank=self.rank)(
                    model,
                    val_batches,
                )
            rank_zero_only(self.save_model_wandb_artifact, rank=self.rank)(
                model, epoch=epoch
            )

    def _eval_step(
        self, model: nn.Module, val_batch: EntityDatasetCollatedDataOutput
    ) -> Tuple:
        inputs = val_batch.input_tensor.to(self.device)
        spatiotemporal_intervals = val_batch.spatiotemporal_interval_tensor.to(
            self.device
        )
        entity_class_labels = val_batch.entity_class_targets.cpu()
        padding_mask = val_batch.padding_mask.to(self.device)
        binned_ship_type_tensor = val_batch.binned_ship_type_tensor.to(self.device)
        # List of dictionaries of information about the trajectories
        metadata_lst = val_batch.metadata
        seq_lengths = val_batch.seq_lengths
        entity_names = [metadata["entity_name"] for metadata in metadata_lst]
        trackIds = [metadata["trackId"] for metadata in metadata_lst]
        file_locations = [metadata["file_location"] for metadata in metadata_lst]
        self.all_flag_codes.extend([metadata["flag_code"] for metadata in metadata_lst])
        self.all_seq_lengths.extend(seq_lengths)

        entity_class_outputs = model(
            inputs,
            spatiotemporal_intervals,
            binned_ship_type_tensor,  # Used optionally
            padding_mask=padding_mask,
        )
        entity_class_outputs = entity_class_outputs.detach().cpu()

        loss = self._get_loss(
            entity_class_outputs,
            entity_class_labels,
        )
        return (
            entity_class_outputs,
            entity_class_labels,
            loss,
            inputs,
            seq_lengths,
            entity_names,
            trackIds,
            file_locations,
        )

    def _eval(self, model: nn.Module, val_batches: DataLoader) -> None:
        # Validation Run #
        model.eval()
        self.set_dataset_based_on_model_mode(val_batches, model)
        with torch.no_grad():
            # What to do for val loop for distributed training?
            # ✨ W&B: Create a Table to store predictions for each test step
            class_score_names = [f"{i}_score" for i in self.class_descriptions]
            table_column_names = [
                "id",
                "trackId",
                "name",
                "file_location",
                "image",
                "prediction",
                "ground_truth",
            ]
            table_column_names.extend(class_score_names)

            correct_table = wandb.Table(columns=table_column_names)
            incorrect_table = wandb.Table(columns=table_column_names)
            self.all_flag_codes: list = []
            self.all_seq_lengths: list = []
            all_prediction_correctness = []
            all_probs = []
            all_labels = []
            total_validation_loss = 0.0
            for val_i, val_batch in enumerate(val_batches):
                logger.info(f"Validation Batch {val_i}")
                if val_i > self.max_num_batches_to_log:
                    break
                (
                    entity_class_outputs,
                    entity_class_labels,
                    entity_class_loss,
                    inputs,
                    seq_lengths,
                    entity_names,
                    trackIds,
                    file_locations,
                ) = self._eval_step(model, val_batch)
                wandb.log({"Validation Loss": entity_class_loss.item()})
                total_validation_loss += entity_class_loss.item()
                p = torch.nn.functional.softmax(entity_class_outputs.detach(), dim=1)
                top_pred_ids = entity_class_outputs.argmax(axis=1)
                wandb.log(
                    {
                        "pr": wandb.plot.pr_curve(
                            entity_class_labels.numpy(),
                            entity_class_outputs.detach().numpy(),
                            labels=self.class_descriptions,
                            classes_to_plot=None,
                        )
                    }
                )
                # TEMPORARY WE SHOULD MAKE A SUBCLASS OF THIS TRAINER
                wandb.log(
                    {
                        "F1": f1_score(
                            entity_class_labels.numpy(),
                            top_pred_ids.detach().numpy(),
                            average="macro",
                        )
                    }
                )
                all_prediction_correctness.append(
                    (entity_class_labels.numpy() == top_pred_ids.detach().numpy()) * 1
                )
                all_probs.append(p.numpy())
                all_labels.append(entity_class_labels.numpy())
                # Log confusion matrix to include everything in the val set
                wandb.log(
                    {
                        "Confusion Matrix Batch": wandb.plot.confusion_matrix(
                            probs=p.numpy(),
                            y_true=entity_class_labels.numpy(),
                            class_names=self.class_descriptions,
                            title="Confusion Matrix Batch",
                        )
                    }
                )

                wandb.log(
                    {
                        "ROC": wandb.plot.roc_curve(
                            entity_class_labels.numpy(),
                            entity_class_outputs.numpy(),
                            labels=self.class_descriptions,
                            classes_to_plot=None,
                        )
                    }
                )
                # Log this table via succesive copying in wandb, #Takes like ~1.5 min for batch size 128
                if val_i < self.max_num_batches_to_log_images:
                    logger.info(
                        f"Logging Incorrect/Correct Tables for val batch {val_i}"
                    )
                    correct_outputs, incorrect_outputs = (
                        log_val_predictions_entity_type(
                            inputs.data.cpu(),
                            seq_lengths,
                            entity_names,
                            trackIds,
                            file_locations,
                            entity_class_labels.cpu().numpy(),
                            entity_class_outputs,
                            top_pred_ids,
                            correct_table,
                            incorrect_table,
                            val_i,
                            self.label_enum.to_label_name_dict(),
                        )
                    )

                    wandb.log({"Validation: Correct Predictions": copy(correct_outputs)})
                    wandb.log(
                        {"Validation: Incorrect Predictions": copy(incorrect_outputs)}
                    )
            wandb.log({"Total Validation Loss": total_validation_loss})
            all_prediction_correctness = np.concatenate(all_prediction_correctness)
            correct_per_flag_code_df = pd.DataFrame(
                data={
                    "flag_code": self.all_flag_codes,
                    "prediction_accuracy": all_prediction_correctness,
                }
            )

            flag_code_accuracy_df = (
                correct_per_flag_code_df.groupby("flag_code").mean().reset_index()
            )

            table_acc = wandb.Table(dataframe=flag_code_accuracy_df)
            wandb.log(
                {
                    "Accuracy per Country": wandb.plot.bar(
                        table_acc,
                        "flag_code",
                        "prediction_accuracy",
                        title="Accuracy per Country of Origin",
                    )
                }
            )

            flag_counts_df = pd.DataFrame(
                correct_per_flag_code_df.value_counts("flag_code")
            ).reset_index()

            table_counts = wandb.Table(dataframe=flag_counts_df)
            wandb.log(
                {
                    "Country Distribution": wandb.plot.bar(
                        table_counts,
                        "flag_code",
                        "count",
                        title="Count per Country of Origin",
                    )
                }
            )

            seq_length_accuracy_df = pd.DataFrame(
                {
                    "seq_length": self.all_seq_lengths,
                    "prediction_accuracy": all_prediction_correctness,
                }
            )
            bins = [10, 100, 500]
            bins.extend(range(1000, self.data_config["MAX_TRAJECTORY_LENGTH"], 1000))
            labels = [f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])]
            # Find average accuracy in each bin
            seq_length_accuracy_df["bins"] = pd.cut(
                self.all_seq_lengths, bins=bins, labels=labels
            )
            seq_length_accuracy_df = seq_length_accuracy_df.groupby("bins").mean()
            seq_length_accuracy_df = seq_length_accuracy_df.reset_index()
            table_seq_length = wandb.Table(dataframe=seq_length_accuracy_df)
            wandb.log(
                {
                    "Accuracy per Sequence Length": wandb.plot.bar(
                        table_seq_length,
                        "bins",
                        "prediction_accuracy",
                        title="Accuracy per Sequence Length",
                    )
                }
            )

            # Log confusion matrix to include everything in the val set
            probs_final_val = np.concatenate(all_probs)
            labels_final_val = np.concatenate(all_labels)
            wandb.log(
                {
                    "Confusion Matrix Total": wandb.plot.confusion_matrix(
                        probs=probs_final_val,
                        y_true=labels_final_val,
                        class_names=self.class_descriptions,
                        title="Confusion Matrix Total",
                    )
                }
            )

    def setup_and_train(self, rank: int, world_size: int) -> None:
        """Setup and train function for the distributed training


        Parameters
        ----------
        rank : int
            Rank of the process
        world_size : int
            Number of processes, i.e. number of GPUs
        """
        self.setup(rank, world_size)
        rank_zero_only(logger.info, self.rank)("Setup Dataset")
        full_dataset = self._initialize_dataset(self.entity_class_labels_path)
        train_dataset, val_dataset = self._split_train_val(full_dataset)

        if self.world_size > 1:
            self.setup_distributed()
        train_loader = self.initialize_data_loader_train(train_dataset)
        val_loader = self.initialize_data_loader_val(val_dataset)
        rank_zero_only(logger.info, self.rank)("Initialize Model")
        model = (
            self.load_from_checkpoint()
            if self.model_resume_checkpoint is not None
            else self.initialize_model()
        )
        model = self.move_model_to_gpu(model)

        # set up optimizer
        self.set_optimizer(model)
        self.set_scheduler()
        # set up criterion
        self.subpath_criterion = nn.CrossEntropyLoss(
            ignore_index=self.data_config["LABEL_PADDING_VALUE"]
        )
        self.entity_class_criterion = nn.CrossEntropyLoss(
            ignore_index=self.data_config["LABEL_PADDING_VALUE"]
        )

        rank_zero_only(self._log_model_architecture_info, self.rank)(model)

        try:
            self._train(
                model,
                train_loader,
                val_loader,
            )
        except Exception as e:
            logger.error(
                f"Training failed with error: {e}",
            )
            traceback.print_exc()
            logger.info(f"config used {self.wandb_config}")
        finally:
            if world_size > 1:
                self.cleanup()
            wandb.finish()
