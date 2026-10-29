"""Module for evaluation utilities for the Atlas models

"""

import numpy as np
import wandb
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.log_utils import get_logger
from sklearn.metrics import f1_score

logger = get_logger(__name__)


def binarize_by_fishing(label_array: np.ndarray) -> np.ndarray:
    """Binarize the predicted and actual subpath activity labels by fishing

    Parameters
    ----------
    label_array : np.ndarray
        Array of subpath activity labels

    Returns
    -------
    np.ndarray
        Array of binary labels
    """
    # Get the class predictions
    fishing_label = AtlasActivityLabelsTraining.FISHING.value
    # Binarize based on class
    fishing_per_subpath = (label_array == fishing_label).astype(int)
    return fishing_per_subpath


def convert_output_probs_to_binary_fishing_probs(
    subpath_activity_class_probs: np.ndarray,
) -> np.ndarray:
    """Convert the output probabilities to binary fishing probabilities
    Parameters
    ----------
    subpath_activity_class_probs : np.ndarray
        Array of subpath activity class probabilities with shape (num_subpaths, num_classes)

    Returns
    -------
    np.ndarray
        Array of binary fishing probabilities
    """
    # Get the class predictions
    fishing_label = AtlasActivityLabelsTraining.FISHING.value
    # Combine all the non fishing probs
    fishing_probs = subpath_activity_class_probs[:, fishing_label]
    non_fishing_probs = 1 - fishing_probs
    return np.stack([non_fishing_probs, fishing_probs], axis=1)


def log_all_class_subpath_metrics(
    labels: np.ndarray,
    top_pred_ids: np.ndarray,
    probs: np.ndarray,
    title: str = "Confusion Matrix Total All Classes",
    title_addendum: str = "",
) -> None:
    """Log all class subpath metrics"""
    # What do we log here for non fishing examples
    weighted_f1 = f1_score(top_pred_ids, labels, average="weighted")
    macro_f1 = f1_score(top_pred_ids, labels, average="macro")
    micro_f1 = f1_score(top_pred_ids, labels, average="micro")
    logger.info(f"Weighted F1 ({title_addendum}): {weighted_f1}")
    logger.info(f"Macro F1 ({title_addendum}): {macro_f1}")
    logger.info(f"Micro F1 ({title_addendum}): {micro_f1}")
    wandb.log({f"Weighted F1 ({title_addendum})": weighted_f1})
    wandb.log({f"Macro F1 ({title_addendum})": macro_f1})
    wandb.log({f"Micro F1 ({title_addendum})": micro_f1})
    wandb.log(
        {
            f"{title} {title_addendum}": wandb.plot.confusion_matrix(
                probs=probs,
                y_true=labels,
                class_names=AtlasActivityLabelsTraining.to_class_descriptions(),
                title=title,
            )
        }
    )


def convert_output_probs_to_binary_class_probs(
    subpath_activity_class_probs: np.ndarray,
    class_label: int,
) -> np.ndarray:
    """Convert the output probabilities to binary class probabilities

    Parameters
    ----------
    subpath_activity_class_probs : np.ndarray
        Array of subpath activity class probabilities with shape (num_subpaths, num_classes)
    class_label : int
        The class label to convert to binary

    Returns
    -------
    np.ndarray
        Array of binary class probabilities
    """
    # Get the class predictions
    class_probs = subpath_activity_class_probs[:, class_label]
    non_class_probs = 1 - class_probs
    return np.stack([non_class_probs, class_probs], axis=1)


def binarize_by_class(label_array: np.ndarray, class_label: int) -> np.ndarray:
    """Binarize the predicted and actual subpath activity labels by class

    Parameters
    ----------
    label_array : np.ndarray
        Array of labels

    Returns
    -------
    np.ndarray
        Array of binary labels
    """
    # Get the class predictions
    # Binarize based on class
    binarized_arr = (label_array == class_label).astype(int)
    return binarized_arr


def log_target_class_or_not_metrics(
    labels: np.ndarray,
    top_pred_ids: np.ndarray,
    probs: np.ndarray,
    target_class_label: int,
    class_name: str,
    title_addendum: str = "",
) -> None:
    """Log class or not metrics"""
    class_or_not_probs = convert_output_probs_to_binary_class_probs(
        probs, target_class_label
    )
    class_or_not_predictions = binarize_by_class(top_pred_ids, target_class_label)
    class_or_not_labels = binarize_by_class(labels, target_class_label)

    # What do we log here for non class examples
    class_f1 = f1_score(class_or_not_predictions, class_or_not_labels)
    logger.info(f"{class_name} F1 {title_addendum}: {class_f1}")
    wandb.log({f"{class_name} F1 {title_addendum}": class_f1})
    class_names = ["Not " + class_name, class_name]
    wandb.log(
        {
            f"{class_name} Subpath pr{title_addendum}": wandb.plot.pr_curve(
                class_or_not_labels,
                class_or_not_probs,
                labels=class_names,
                title=f"{class_name} Subpath pr {title_addendum}",
            )
        }
    )

    wandb.log(
        {
            f"Confusion Matrix Total {class_name} {title_addendum}": wandb.plot.confusion_matrix(
                probs=class_or_not_probs,
                y_true=class_or_not_labels,
                class_names=class_names,
                title=f"Confusion Matrix Total {class_name}{title_addendum}",
            )
        }
    )
