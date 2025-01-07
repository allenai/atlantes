"""Simple Evaluation Script for the EOS Task



I want to be able to look at in a sorted way the types of mistakes we are making
So will need the raw_paths and the send from the metadata and then the predicitons all accumulated then I can do a cpu bound parallel process to get the results
over the tasks



"""

import warnings
from copy import copy

import numpy as np
import pandas as pd
import torch
import wandb
from atlantes.atlas.activity_trainer import AtlasActivityRealTimeTrainer
from atlantes.atlas.atlas_utils import (
    AtlasActivityLabelsTraining,
    get_experiment_config_atlas_activity_real_time,
    log_val_predictions_end_of_sequence_task)
from atlantes.log_utils import get_logger
from torch import nn

logger = get_logger(__name__)


DATA_CONFIG = get_experiment_config_atlas_activity_real_time("data")
EXPERIMENTAL_CONFIG_TRAIN = get_experiment_config_atlas_activity_real_time()
EXPERIMENTAL_CONFIG_MODEL = get_experiment_config_atlas_activity_real_time(
    "hyperparameters"
)

TABLE_COLUMNS = ["id", "file_locations", "send_times", "image", "label", "pred", "prob"]
# If log some correct or incorrect images is the case we can do that all in a row here


def eval_incorrect_correct_tables() -> None:
    """Log the incorrect and correct tables for the model"""
    warnings.simplefilter(action="ignore", category=FutureWarning)
    trainer = AtlasActivityRealTimeTrainer(
        data_config=DATA_CONFIG,
        experimental_config_train=EXPERIMENTAL_CONFIG_TRAIN,
        experimental_config_model=EXPERIMENTAL_CONFIG_MODEL,
    )
    trainer.setup(rank=0, world_size=1)
    full_dataset = trainer._initialize_dataset(trainer.activity_labels_file)
    train_split, val_split = trainer._split_train_val(full_dataset)
    data_loader = trainer.initialize_data_loader_train(val_split)
    model = trainer.load_from_checkpoint()
    model = trainer.move_model_to_gpu(model)
    trainer.criterion = nn.CrossEntropyLoss(
        ignore_index=trainer.data_config["LABEL_PADDING_VALUE"]
    )
    val_losses = []
    model.eval()
    with torch.no_grad():
        i = 0
        total_validation_loss = 0.0
        labels_list = []
        pred_ids_list = []
        probs_list = []
        file_paths_list = []
        send_times_list = []
        input_list = []
        track_lengths_list = []
        for collated_data in data_loader:
            if collated_data is None:
                continue
            if i > 20:
                break
            logger.info(f"Eval Batch {i}")
            inputs = collated_data.input_tensor.to(trainer.device)
            spatiotemporal_intervals = collated_data.spatiotemporal_interval_tensor.to(
                trainer.device
            )
            activity_labels = collated_data.activity_labels.to(trainer.device)
            padding_mask = collated_data.padding_mask.to(trainer.device)
            binned_ship_types = collated_data.binned_ship_type_tensor.to(trainer.device)

            subpath_class_outputs = model(
                inputs=inputs,
                spatiotemporal_tensor=spatiotemporal_intervals,
                binned_ship_type=binned_ship_types,
                padding_mask=padding_mask,
            )
            file_paths = [x["file_location"] for x in collated_data.metadata]
            send_time = [x["send_time"] for x in collated_data.metadata]
            track_length = [x["track_length"] for x in collated_data.metadata]
            loss = trainer._get_loss(subpath_class_outputs, activity_labels)
            val_loss = loss.item()
            total_validation_loss += val_loss
            probs = torch.nn.functional.softmax(subpath_class_outputs, dim=1)
            top_pred_ids = probs.argmax(dim=1)
            probs = probs.detach().cpu().numpy()
            top_pred_ids_arr = top_pred_ids.detach().cpu().numpy()
            logger.info(f"Validation Loss: {val_loss}")
            wandb.log({"Validation Loss Batch": val_loss})
            val_losses.append(val_loss)
            labels_list.append(activity_labels.cpu().numpy())
            pred_ids_list.append(top_pred_ids_arr)
            probs_list.append(probs)
            file_paths_list.append(file_paths)
            send_times_list.append(send_time)
            # Remove the zero padding from the inputs
            inputs = inputs.cpu().numpy()
            input_list.append(inputs)
            track_lengths_list.append(track_length)
            i += 1

        all_labels = np.concatenate(labels_list)
        all_pred_ids = np.concatenate(pred_ids_list)
        all_probs = np.concatenate(probs_list)
        file_paths_list = [item for sublist in file_paths_list for item in sublist]
        send_times_list = [item for sublist in send_times_list for item in sublist]
        input_list = [item for sublist in input_list for item in sublist]
        track_lengths_list_flatten = [
            item for sublist in track_lengths_list for item in sublist
        ]
        # Remove all where label is set to padding value includes examples wiht not enough context
        unknown_label_mask = all_labels != trainer.data_config["LABEL_PADDING_VALUE"]
        all_labels = all_labels[unknown_label_mask]
        all_pred_ids = all_pred_ids[unknown_label_mask]
        all_probs = all_probs[unknown_label_mask]
        unknown_label_mask_indices = np.where(unknown_label_mask)[0]
        file_paths_list = [file_paths_list[i] for i in unknown_label_mask_indices]
        send_times_list = [send_times_list[i] for i in unknown_label_mask_indices]
        input_list = [input_list[i] for i in unknown_label_mask_indices]
        track_lengths_list_flatten = [
            track_lengths_list_flatten[i] for i in unknown_label_mask_indices
        ]
        logger.info(pd.Series(val_losses).describe())
        # Get all the supposed to be fishing incorrect indices
        fishing_mask = all_labels == AtlasActivityLabelsTraining.FISHING.value
        incorrect_mask = all_pred_ids != all_labels
        incorrect_fishing_mask = np.logical_and(fishing_mask, incorrect_mask)
        false_negative_fishing_indices = np.where(incorrect_fishing_mask)[0]
        # get false_positive_fishing_indices
        fishing_preds_mask = all_pred_ids == AtlasActivityLabelsTraining.FISHING.value
        false_positive_fishing_mask = np.logical_and(incorrect_mask, fishing_preds_mask)
        false_positive_fishing_indices = np.where(false_positive_fishing_mask)[0]
        false_negative_table = wandb.Table(columns=TABLE_COLUMNS)
        false_positive_table = wandb.Table(columns=TABLE_COLUMNS)
        class_label_name_dict = AtlasActivityLabelsTraining.to_label_name_dict()
        for i in range(len(false_negative_fishing_indices)):
            index = false_negative_fishing_indices[i]
            false_negative_table = log_val_predictions_end_of_sequence_task(
                input_list[index],
                file_paths_list[index],
                send_times_list[index],
                all_labels[index],
                all_pred_ids[index],
                all_probs[index],
                track_lengths_list_flatten[index],
                false_negative_table,
                i,
                class_label_name_dict,
            )
            wandb.log({"False Negative Fishing Table": copy(false_negative_table)})

        for i in range(len(false_positive_fishing_indices)):
            index = false_positive_fishing_indices[i]
            false_positive_table = log_val_predictions_end_of_sequence_task(
                input_list[index],
                file_paths_list[index],
                send_times_list[index],
                all_labels[index],
                all_pred_ids[index],
                all_probs[index],
                track_lengths_list_flatten[index],
                false_positive_table,
                i,
                class_label_name_dict,
            )
            wandb.log({"False Positive Fishing Table": copy(false_positive_table)})
    logger.info("Wandb Tables Logged FInished experiment")


if __name__ == "__main__":
    eval_incorrect_correct_tables()
