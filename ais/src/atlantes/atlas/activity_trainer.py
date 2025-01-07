"""Atlas Activity Trainer class for training the Atlas Activity models.


# TODO: add ways to handle data versioning in this pipeline (current method is to just log the path in the confg file) """

import logging
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from atlantes.atlas.ais_dataset import ActivityDatasetEndOfSequence
from atlantes.atlas.atlas_eval_utils import (log_all_class_subpath_metrics,
                                             log_target_class_or_not_metrics)
from atlantes.atlas.atlas_net import AtlasActivityEndOfSequenceTaskNet
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.base_trainer import BaseAtlasTrainer
from atlantes.atlas.collate import (
    ActivityDatasetEndOfSequenceCollatedDataOutput, RealTimeActivityCollater)
from atlantes.atlas.training_utils import rank_zero_only
from atlantes.log_utils import get_logger
from atlantes.utils import NUM_TO_CATEGORY_DESC, get_vessel_type_name
from sklearn.model_selection import train_test_split
from torch.distributed import reduce
from torch.utils.data import DataLoader, Subset

logger = get_logger(__name__)
MODELS_DIR_DEFAULT = str(Path(__file__).parent.parent / "models")


class AtlasActivityRealTimeTrainer(BaseAtlasTrainer):
    """Trainer abstraction for training the Atlas Activity models


    Main method is setup_and_train, which sets up the distributed training
    (if using multiple gpus) and launches the training loop


    NB: some of the parameters are overwritten in debug mode

    """

    def __init__(
        self,
        data_config: dict,
        experimental_config_train: dict,
        experimental_config_model: dict,
        label_enum: Any = AtlasActivityLabelsTraining,
    ):
        super().__init__(
            data_config, experimental_config_train, experimental_config_model
        )
        self._unpack_experimental_config_train()
        self.multiprocessing_context = (
            "forkserver" if self.use_gcs and self.num_data_workers > 0 else None
        )
        self.timeout = (
            self.experimental_config_train["TIMEOUT"]
            if self.num_data_workers > 0
            else 0
        )
        self.collate_fn = RealTimeActivityCollater(
            use_prepadding=experimental_config_model["USE_PREPAD"]
        )
        np.random.seed(self.data_config["RANDOM_STATE"])
        self.label_enum = label_enum
        self.activity_class_dict = label_enum.to_label_name_dict()
        self.n_classes: int = len(self.activity_class_dict)

    def _unpack_experimental_config_train(
        self,
    ) -> None:
        """Unpack and set the experimental configuration as instance variables

        This method should be called after the experimental configuration is loaded
        and before training starts. See the .yaml file for more details

        Note that the values are overwritten by the DEBUG mode

        """
        # Constants are read from config/config.yml, but can be overridden here
        # We should Log them in W&B
        self.val_size = self.experimental_config_train["VAL_SIZE"] / 100
        self.optimizer_key = self.experimental_config_train["OPTIMIZER"]
        self.scheduler_key = self.experimental_config_train["LR_SCHEDULER"]
        self.n_epochs = self.experimental_config_train["N_EPOCHS"]
        self.train_batch_size = self.experimental_config_train["TRAIN_BATCH_SIZE"]
        self.val_batch_size = self.experimental_config_train["VAL_BATCH_SIZE"]
        self.sgd_momentum = self.experimental_config_train["SGD_MOMENTUM"]
        self.weight_decay = self.experimental_config_train["WEIGHT_DECAY"]
        self.learning_rate = self.experimental_config_train["LEARNING_RATE"]
        self.max_num_batches_to_log_images = self.experimental_config_train[
            "MAX_NUM_BATCHES_TO_LOG_IMAGES"
        ]
        # TODO: Use this to make a model registry if we want to use different models
        self.model = self.experimental_config_train["MODEL_ARCHITECTURE"]
        self.annealing_tmax = self.experimental_config_train["ANNEALING_TMAX"]
        self.eta_min = self.experimental_config_train["ETA_MIN"]
        self.cpe_kernel_size = self.experimental_config_train["KERNEL_SIZE"]
        self.num_data_workers = self.experimental_config_train["NUM_DATA_WORKERS"]
        self.sweep = self.experimental_config_train["SWEEP"]
        self.profiling = self.experimental_config_train["PROFILING"]
        self.activity_labels_file = self.experimental_config_train[
            "ACTIVITY_LABEL_DATASET_FILES"
        ]
        self.activity_label_val_dataset_files = self.experimental_config_train[
            "ACTIVITY_LABEL_VAL_DATASET_FILES"
        ]
        self.warmup_steps = self.experimental_config_train["WARMUP_STEPS"]
        self.eval_cadence = self.experimental_config_train["EVAL_CADENCE"]
        self.num_gpus = self.experimental_config_train["NUM_GPUS"]
        self.use_class_weighted_loss = self.experimental_config_train[
            "USE_CLASS_WEIGHTED_LOSS"
        ]
        self.use_gcs = self.experimental_config_train["USE_GCS"]

        self.pin_memory = self.experimental_config_train["PIN_MEMORY"]
        self.model_resume_checkpoint = self.experimental_config_train[
            "MODEL_CHECKPOINT"
        ]
        self.debug_mode = self.experimental_config_train["DEBUG_MODE"]
        self.eval_before_train = self.experimental_config_train["EVAL_BEFORE_TRAIN"]
        self.dist_test_labels_file = self.experimental_config_train.get(
            "DIST_TEST_LABELS_FILES", None
        )
        self.test_dist_before_train = self.experimental_config_train[
            "TEST_DIST_BEFORE_TRAIN"
        ]
        self.use_features_only = self.experimental_config_train["USE_FEATURES_ONLY"]
        self.skip_train = False
        self.eval_epoch_cadence = self.experimental_config_train["EVAL_EPOCH_CADENCE"]
        if self.debug_mode:
            logger.warning("Entering Debug Mode, see experimental config for details")
            if torch.cuda.is_available():
                self.num_gpus = 1
            else:
                raise (ValueError("No GPU available, quitting"))
            # Set logging level to debug
            logger.setLevel(getattr(logging, "DEBUG"))
            self.train_batch_size = 16
            self.val_batch_size = 16
            self.rank = 0
            self.world_size = 1
            self.skip_train = True
            self.project_name = self.project_name + "_DEBUG"
            self.num_data_workers = 4
            self.profiling = True
            self.max_num_batches_to_log_images = 1
            torch.autograd.set_detect_anomaly(False)

    def _initialize_dataset(self, labels_files: Any) -> ActivityDatasetEndOfSequence:
        """Initialize the dataset"""
        logger.info(f"Initializing the dataset: {labels_files}")
        full_dataset = ActivityDatasetEndOfSequence(
            activity_label_file_paths=labels_files,
            dataset_config=self.data_config,
            label_enum=self.label_enum,
        )
        self.class_descriptions = full_dataset.class_descriptions
        self.full_dataset_len = len(full_dataset)
        return full_dataset

    def _split_train_val(
        self, full_dataset: ActivityDatasetEndOfSequence
    ) -> Tuple[Subset, Subset]:
        """Split the dataset into train and validation datasets"""
        logger.info("Splitting the dataset into train and validation datasets")
        max_track_id_group = np.max(full_dataset.track_id_groups)
        train_group_ids, val_group_ids = train_test_split(
            range(max_track_id_group + 1), test_size=self.val_size
        )
        train_indices = np.where(
            np.isin(full_dataset.track_id_groups, train_group_ids)
        )[0]
        val_indices = np.where(np.isin(full_dataset.track_id_groups, val_group_ids))[0]
        if len(np.intersect1d(train_indices, val_indices)) > 0:
            raise ValueError("Train and Val datasets have overlapping track ids")
        train_split = Subset(full_dataset, train_indices)
        val_split = Subset(full_dataset, val_indices)
        return train_split, val_split

    def build_train_and_val_datasets_from_different_sources(
        self, full_dataset: ActivityDatasetEndOfSequence
    ) -> Tuple[Subset, Subset]:
        """Build train and val datasets from different sources

        This is useful for when you have a different source for the train and val datasets
        activity_labels_file will be train and activity_labels_file_val will be val"""
        logger.info("Building Val dataset from different source")
        val_dataset = ActivityDatasetEndOfSequence(
            activity_label_file_paths=self.activity_label_val_dataset_files,
            dataset_config=self.data_config,
        )
        logger.info("val dataset built")
        val_track_ids = val_dataset.track_ids
        # In place filter
        full_dataset.filter_out_trackids(val_track_ids)
        logger.info("Filtered out val track ids from train dataset")
        if len(np.intersect1d(full_dataset.track_ids, val_dataset.track_ids)) > 0:
            raise ValueError("Train and Val datasets have overlapping track ids")
        train_dataset = Subset(full_dataset, range(len(full_dataset)))

        val_dataset = Subset(val_dataset, range(len(val_dataset)))

        return train_dataset, val_dataset

    def initialize_model(self) -> AtlasActivityEndOfSequenceTaskNet:
        """Initialize the Atlas_activity_model

        if loading weights must match the model architecture"""
        return AtlasActivityEndOfSequenceTaskNet(
            c_in=len(self.data_config["MODEL_INPUT_COLUMNS_ACTIVITY"]),
            subpath_output_dim=self.n_classes,
            transformer_layers_pre_squeeze=self.experimental_config_model[
                "N_PRE_SQUEEZE_TRANSFORMER_LAYERS"
            ],
            n_heads=self.experimental_config_model["N_HEADS"],
            token_dim=self.experimental_config_model["TOKEN_DIM"],
            mlp_dim=self.experimental_config_model["MLP_DIM"],
            cpe_kernel_size=self.cpe_kernel_size,
            cpe_layers=self.experimental_config_model["CPE_LAYERS"],
            cnn_layers=self.experimental_config_model["CNN_LAYERS"],
            cnn_kernel_size=self.experimental_config_model["CNN_KERNEL_SIZE"],
            use_residual_cnn=self.experimental_config_model["USE_RESIDUAL_CNN"],
            use_layer_norm_cnn=self.experimental_config_model["USE_LAYERNORM_CNN"],
            use_channel_dim_only_layernorm_cnn=self.experimental_config_model[
                "USE_CHANNEL_DIM_LN_CNN"
            ],
            dropout_p=self.experimental_config_model["DROPOUT_P"],
            qkv_bias=self.experimental_config_model["QKV_BIAS"],
            use_binned_ship_type=self.experimental_config_model["USE_SHIP_TYPE"],
            use_lora=self.experimental_config_train.get("USE_LORA", False)
        )

    def _get_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Get the loss"""
        if self.use_class_weighted_loss:
            if not isinstance(self.criterion, nn.CrossEntropyLoss):
                raise ValueError("Criterion must be nn.CrossEntropyLoss to set weights")
            weights = self.get_cross_entropy_class_weights(
                labels, self.n_classes, self.data_config["LABEL_PADDING_VALUE"]
            )
            self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
                weight=weights, ignore_index=self.data_config["LABEL_PADDING_VALUE"]
            )
        else:
            pass  # setup in setup and train
        return self.criterion(outputs, labels)

    def _train_step(
        self,
        model: nn.Module,
        train_batch: ActivityDatasetEndOfSequenceCollatedDataOutput,
    ) -> torch.tensor:
        """Perform a single training step"""
        inputs = train_batch.input_tensor.to(self.device)
        spatiotemporal_intervals = train_batch.spatiotemporal_interval_tensor.to(
            self.device
        )
        activity_labels = train_batch.activity_labels.to(self.device)
        padding_mask = train_batch.padding_mask.to(self.device)
        binned_ship_types = train_batch.binned_ship_type_tensor.to(self.device)

        subpath_class_outputs = model(
            inputs=inputs,
            spatiotemporal_tensor=spatiotemporal_intervals,
            binned_ship_type=binned_ship_types,
            padding_mask=padding_mask,
        )
        loss = self._get_loss(subpath_class_outputs, activity_labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.world_size > 1:
            reduce(loss, dst=self.main_process_rank, op=torch.distributed.ReduceOp.AVG)
        return loss

    def _train_batch(
        self,
        model: nn.Module,
        train_batches: DataLoader,
        val_batches: DataLoader,
    ) -> float:
        """Perform a single training batch"""
        model.train()
        self.set_dataset_based_on_model_mode(train_batches, model)
        loss_accumulation = 0.0
        i = 0
        for batched_data in train_batches:
            if self.skip_train:
                loss_val = 0.0  # prefer this to be np.nan but that doesn't work
                logger.info("Skipping training loop, setting loss to nan")
                break
            if batched_data is None:
                logger.error("Batched data is None")
                continue
            loss = self._train_step(model, batched_data)
            if not torch.isfinite(loss):
                logger.error("Loss is not finite")
                batch_details = [
                    f"batch_size: {batched_data.input_tensor.shape[0]}",
                    f"seq_length: {batched_data.input_tensor.shape[1]}",
                    f"padding_mask: {batched_data.padding_mask}",
                    f"activity_labels: {batched_data.activity_labels}",
                    f"input_tensor: {batched_data.input_tensor}",
                    f"spatiotemporal_interval_tensor: {batched_data.spatiotemporal_interval_tensor}",
                ]
                logger.error("Batch data details: " + " | ".join(batch_details))
                raise ValueError("Loss is infinity or nan")
            loss_val = loss.item()

            loss_accumulation += loss_val
            rank_zero_only(logger.info, rank=self.rank)(f"Batch {i} Loss: {loss_val}")
            rank_zero_only(wandb.log, rank=self.rank)(
                {"Training Loss (batch)": loss_val}
            )

            if i % self.eval_cadence == 0 and i > 0:
                rank_zero_only(self.save_model_wandb_artifact, rank=self.rank)(
                    model, epoch=self.epoch, step=i
                )
                rank_zero_only(self._eval, rank=self.rank)(
                    model,
                    val_batches,
                )
            if self.scheduler is not None:
                rank_zero_only(wandb.log, rank=self.rank)(
                    {"learning_rate": self.scheduler.get_last_lr()[0]}
                )
                self.scheduler.step()
            i += 1
        return loss_accumulation

    def _train(
        self,
        model: nn.Module,
        train_batches: DataLoader,
        val_batches: DataLoader,
    ) -> None:
        """Main training loop

        Parameters
        ----------
        model : Any
            Model to train
        train_batches : DataLoader
            Training data
        val_batches : DataLoader
            Validation data
        Returns
        -------
        None
        """

        rank_zero_only(wandb.watch, rank=self.rank)(
            model, log_freq=100, log="all", log_graph=True
        )
        model.train()

        if self.dist_test_labels_file and self.test_dist_before_train:
            dist_test_dataset = self._initialize_dataset(self.dist_test_labels_file)
            dist_test_loader = self.initialize_data_loader_val(dist_test_dataset)
            logger.info("Running the distribution test before training")
            rank_zero_only(self._eval_dist_test, rank=self.rank)(
                model,
                dist_test_loader,
            )

        if self.eval_before_train:
            rank_zero_only(self._eval, rank=self.rank)(
                model,
                val_batches,
            )
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            rank_zero_only(print, rank=self.rank)(f"Epoch {epoch}")
            # eval will happen in the batch via the self.eval_cadence config, useful for large datasets
            loss_accumulation = self._train_batch(model, train_batches, val_batches)

            rank_zero_only(wandb.log, rank=self.rank)(
                {"training loss (epoch)": loss_accumulation}
            )
            logger.debug(
                f" Evaluating model at {epoch=} with {self.eval_epoch_cadence=}"
            )
            if self.dist_test_labels_file:
                logger.info(
                    f"Evaluating the model against the distribution found in: {self.dist_test_labels_file=}"
                )
                dist_test_dataset = self._initialize_dataset(self.dist_test_labels_file)
                dist_test_loader = self.initialize_data_loader_val(dist_test_dataset)
                logger.info("Running the distribution test before training")
                rank_zero_only(self._eval_dist_test, rank=self.rank)(
                    model,
                    dist_test_loader,
                )
            is_last_epoch = epoch == self.n_epochs - 1
            rank_zero_only(self.save_model_wandb_artifact, rank=self.rank)(
                model, epoch=epoch
            )
            if self.eval_epoch_cadence == 1:
                rank_zero_only(self._eval, rank=self.rank)(
                    model,
                    val_batches,
                )
            elif ((epoch + 1 % self.eval_epoch_cadence == 0)) or is_last_epoch:
                rank_zero_only(self._eval, rank=self.rank)(
                    model,
                    val_batches,
                )

    def _unpack_metadata(
        self, metadata_list: list
    ) -> Tuple[list[str], list[str], list[str], list[str]]:
        """
        Unpacks the metadata from a list of dictionaries for atlas activity evaluation.

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

    def _eval_dist_test(
        self,
        model: nn.Module,
        val_batches: DataLoader,
    ) -> None:
        """Evaluate the model on the validation set for end of sequence activity prediction"""
        model.eval()
        self.set_dataset_based_on_model_mode(val_batches, model)
        with torch.no_grad():
            total_validation_loss = 0.0
            labels_lst = []
            pred_ids_lst = []
            probs_lst = []
            metadata_lst = []
            cumulative_vessel_activity_distribution: dict[int, dict] = {}
            activity_to_vessel_distribution: dict[int, dict] = {}
            class_names = [label.name for label in self.label_enum]
            for batch_index, collated_data in enumerate(val_batches):

                logger.info(f"Validation Batch {batch_index}")

                if collated_data is None:
                    continue

                inputs = collated_data.input_tensor.to(self.device)
                spatiotemporal_intervals = (
                    collated_data.spatiotemporal_interval_tensor.to(self.device)
                )
                activity_labels = collated_data.activity_labels.to(self.device)
                padding_mask = collated_data.padding_mask.to(self.device)
                binned_ship_types = collated_data.binned_ship_type_tensor.to(self.device)

                logger.debug(f"Input tensor size: {inputs.size()}")
                logger.debug(
                    f"Spatiotemporal intervals tensor size: {spatiotemporal_intervals.size()}"
                )
                logger.debug(f"Activity labels tensor size: {activity_labels.size()}")
                logger.debug(f"Padding mask tensor size: {padding_mask.size()}")

                activity_class_outputs = model(
                    inputs=inputs,
                    spatiotemporal_tensor=spatiotemporal_intervals,
                    binned_ship_type=binned_ship_types,
                    padding_mask=padding_mask,
                )
                loss = self._get_loss(activity_class_outputs, activity_labels)
                val_loss = loss.item()
                total_validation_loss += (
                    val_loss  # keep loss because good for debugging
                )

                probs = torch.nn.functional.softmax(activity_class_outputs, dim=1)
                top_pred_ids = probs.argmax(dim=1)
                probs = probs.detach().cpu().numpy()
                top_pred_ids_arr = top_pred_ids.detach().cpu().numpy()
                logger.info(f"Validation Loss: {val_loss}")

                labels_lst.append(activity_labels.cpu().numpy())
                pred_ids_lst.append(top_pred_ids_arr)
                probs_lst.append(probs)
                metadata_lst.extend(collated_data.metadata)

                if batch_index % 10 == 0:
                    all_labels = np.concatenate(labels_lst)
                    all_pred_ids = np.concatenate(pred_ids_lst)
                    all_probs = np.concatenate(probs_lst)

                    unknown_label_mask = (
                        all_labels != self.data_config["LABEL_PADDING_VALUE"]
                    )
                    all_labels = all_labels[unknown_label_mask]
                    all_pred_ids = all_pred_ids[unknown_label_mask]
                    all_probs = all_probs[unknown_label_mask]

                    pred_ids_counts = Counter(all_pred_ids)
                    label_ids_counts = Counter(all_labels)

                    total_predictions = sum(pred_ids_counts.values())
                    total_labels = sum(label_ids_counts.values())

                    # Extract vessel types from metadata
                    vessel_types = np.array([m["binned_ship_type"] for m in metadata_lst])

                    # Update cumulative data for event distribution as a function of vessel type
                    for vessel_type in np.unique(vessel_types):
                        vessel_mask = vessel_types == vessel_type
                        vessel_activity_labels = all_pred_ids[vessel_mask]

                        vessel_activity_counts = Counter(vessel_activity_labels)

                        # If the vessel type already exists, update the cumulative counts
                        if vessel_type in cumulative_vessel_activity_distribution:
                            for activity, count in vessel_activity_counts.items():
                                cumulative_vessel_activity_distribution[vessel_type][
                                    activity
                                ] += count
                        else:
                            # If it's a new vessel type, add its counts to the dictionary
                            cumulative_vessel_activity_distribution[vessel_type] = (
                                vessel_activity_counts
                            )
                    logger.info(cumulative_vessel_activity_distribution)
                    # Log cumulative barplots for each vessel type

                    # Iterate over the cumulative vessel activity distribution
                    for (
                        vessel_type,
                        activity_counts,
                    ) in cumulative_vessel_activity_distribution.items():
                        for activity, count in activity_counts.items():
                            # If the activity is not yet in the dictionary, initialize it
                            if activity not in activity_to_vessel_distribution:
                                activity_to_vessel_distribution[activity] = {}

                            # Update the count for the vessel type within this activity
                            if vessel_type in activity_to_vessel_distribution[activity]:
                                activity_to_vessel_distribution[activity][
                                    vessel_type
                                ] += count
                            else:
                                activity_to_vessel_distribution[activity][
                                    vessel_type
                                ] = count

                    for (
                        activity,
                        vessel_distribution,
                    ) in activity_to_vessel_distribution.items():
                        activity_name = self.label_enum(
                            activity
                        ).name  # Assuming this retrieves the activity's name

                        # Prepare the data for wandb Table
                        table_activity_distribution = wandb.Table(
                            data=[
                                [
                                    get_vessel_type_name(
                                        vessel_type, NUM_TO_CATEGORY_DESC
                                    ),
                                    val,
                                ]
                                for vessel_type, val in vessel_distribution.items()
                            ],
                            columns=["Vessel Type", "Count"],
                        )

                        # Log the barplot for each activity type
                        wandb.log(
                            {
                                f"Vessel Distribution for Activity {activity_name} (Counts)": wandb.plot.bar(
                                    table_activity_distribution,
                                    "Vessel Type",
                                    "Count",
                                    title=f"Vessel Distribution for Activity: {activity_name} after Batch {batch_index}",
                                )
                            }
                        )

                        # Prepare data for New and Old counts
                        table_new_counts = wandb.Table(
                            data=[
                                [self.label_enum(key).name, val]
                                for key, val in pred_ids_counts.items()
                            ],
                            columns=["Activity Type", "Count"],
                        )

                        table_old_counts = wandb.Table(
                            data=[
                                [self.label_enum(key).name, val]
                                for key, val in label_ids_counts.items()
                            ],
                            columns=["Activity Type", "Count"],
                        )

                        # Prepare data for New and Old normalized values
                        table_new_normalized = wandb.Table(
                            data=[
                                [self.label_enum(key).name, val / total_predictions]
                                for key, val in pred_ids_counts.items()
                            ],
                            columns=["Activity Type", "Normalized Value"],
                        )

                        table_old_normalized = wandb.Table(
                            data=[
                                [self.label_enum(key).name, val / total_labels]
                                for key, val in label_ids_counts.items()
                            ],
                            columns=["Activity Type", "Normalized Value"],
                        )

                        # Log the counts side by side
                        wandb.log(
                            {
                                "New Activity Type Distribution (Counts)": wandb.plot.bar(
                                    table_new_counts,
                                    "Activity Type",
                                    "Count",
                                    title=f"New Activity Type Distribution after Batch {batch_index}",
                                ),
                                "Old Activity Type Distribution (Counts)": wandb.plot.bar(
                                    table_old_counts,
                                    "Activity Type",
                                    "Count",
                                    title=f"Old Activity Type Distribution after Batch {batch_index}",
                                ),
                                "Validation Loss": val_loss,
                            }
                        )

                        # Log the normalized values side by side
                        wandb.log(
                            {
                                "New Activity Type Distribution (Normalized)": wandb.plot.bar(
                                    table_new_normalized,
                                    "Activity Type",
                                    "Normalized Value",
                                    title=f"New Normalized Activity Type Distribution after Batch {batch_index}",
                                ),
                                "Old Activity Type Distribution (Normalized)": wandb.plot.bar(
                                    table_old_normalized,
                                    "Activity Type",
                                    "Normalized Value",
                                    title=f"Old Normalized Activity Type Distribution after Batch {batch_index}",
                                ),
                            }
                        )

                # Final logging after all batches
                wandb.log(
                    {
                        "Predicted Matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_labels,
                            preds=all_pred_ids,
                            class_names=class_names,
                        )
                    }
                )

    def _eval(
        self,
        model: nn.Module,
        val_batches: DataLoader,
    ) -> None:
        """Evaluate the model on the validation set for end of sequence activity prediction"""
        model.eval()
        self.set_dataset_based_on_model_mode(val_batches, model)
        with torch.no_grad():
            total_validation_loss = 0.0
            labels_lst = []
            pred_ids_lst = []
            probs_lst = []
            metadata_lst = []
            for batch_index, collated_data in enumerate(val_batches):
                # logger.info(f"Validation Batch {batch_index}")
                if collated_data is None:
                    continue
                # TODO: Make this configurable
                if batch_index > 3000:
                    break
                inputs = collated_data.input_tensor.to(self.device)
                spatiotemporal_intervals = (
                    collated_data.spatiotemporal_interval_tensor.to(self.device)
                )
                activity_labels = collated_data.activity_labels.to(self.device)
                padding_mask = collated_data.padding_mask.to(self.device)
                binned_ship_types = collated_data.binned_ship_type_tensor.to(self.device)

                subpath_class_outputs = model(
                    inputs=inputs,
                    spatiotemporal_tensor=spatiotemporal_intervals,
                    binned_ship_type=binned_ship_types,
                    padding_mask=padding_mask,
                )
                loss = self._get_loss(subpath_class_outputs, activity_labels)
                val_loss = loss.item()
                total_validation_loss += val_loss
                probs = torch.nn.functional.softmax(subpath_class_outputs, dim=1)
                top_pred_ids = probs.argmax(dim=1)
                probs = probs.detach().cpu().numpy()
                top_pred_ids_arr = top_pred_ids.detach().cpu().numpy()
                logger.info(f"Validation Loss: {val_loss}")
                wandb.log({"Validation Loss Batch": val_loss})
                labels_lst.append(activity_labels.cpu().numpy())
                pred_ids_lst.append(top_pred_ids_arr)
                probs_lst.append(probs)
                metadata_lst.extend(collated_data.metadata)

            wandb.log({"Total Validation Loss": total_validation_loss})
            # accumulate f1 scores and metrics here and calculate
            all_labels = np.concatenate(labels_lst)
            all_pred_ids = np.concatenate(pred_ids_lst)
            all_probs = np.concatenate(probs_lst)
            # Remove all where label is set to padding value includes examples wiht not enough context
            unknown_label_mask = all_labels != self.data_config["LABEL_PADDING_VALUE"]
            all_labels = all_labels[unknown_label_mask]
            all_pred_ids = all_pred_ids[unknown_label_mask]
            all_probs = all_probs[unknown_label_mask]
            unknown_label_mask_indices = np.where(unknown_label_mask)[0]
            all_metadata_lst = [metadata_lst[i] for i in unknown_label_mask_indices]
            dataset_membership_array = np.array(
                [metadata["dataset_membership"] for metadata in all_metadata_lst]
            )

            # Add title to these indicating it is All together
            log_all_class_subpath_metrics(
                all_labels,
                all_pred_ids,
                all_probs,
                title_addendum="All Dataset Splits",
            )
            #  Make this function work for any class rather than just fishing
            for class_id, class_name in self.activity_class_dict.items():
                log_target_class_or_not_metrics(
                    all_labels,
                    all_pred_ids,
                    all_probs,
                    class_id,
                    class_name,
                    title_addendum="All Dataset Splits",
                )
            # Log metrics per dataset name split
            for dataset_name in np.unique(dataset_membership_array):
                dataset_mask = dataset_membership_array == dataset_name
                dataset_labels = all_labels[dataset_mask]
                dataset_pred_ids = all_pred_ids[dataset_mask]
                dataset_probs = all_probs[dataset_mask]
                log_all_class_subpath_metrics(
                    dataset_labels,
                    dataset_pred_ids,
                    dataset_probs,
                    title_addendum=dataset_name,
                )
                for class_id, class_name in self.activity_class_dict.items():
                    log_target_class_or_not_metrics(
                        dataset_labels,
                        dataset_pred_ids,
                        dataset_probs,
                        class_id,
                        class_name,
                        title_addendum=dataset_name,
                    )

    def _log_model_architecture_info(self, model: nn.Module) -> None:
        """Log the model architecture info"""
        logger.info(
            f"Training on {self.n_classes} total classes: {self.class_descriptions}"
        )
        logger.info(f"Total # of trajectories in dataset: {self.full_dataset_len}")
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

    def setup_and_train(self, rank: int, world_size: int) -> None:
        """Sets up and trains the entire pipeline as it would occur in a single run"""
        # this where we launch the whole set up and training hook it all up!
        self.setup(rank, world_size)
        rank_zero_only(logger.info, self.rank)("Setup Dataset")
        full_dataset = self._initialize_dataset(self.activity_labels_file)

        rank_zero_only(logger.info, self.rank)("Split Train and Val")
        if self.activity_label_val_dataset_files is not None:
            train_dataset, val_dataset = (
                self.build_train_and_val_datasets_from_different_sources(full_dataset)
            )
        else:
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
        self.criterion = nn.CrossEntropyLoss(
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
