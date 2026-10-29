"""Script/module to calculate inter-annotator agreement (IAA) for human annotations

Rules: A track must be completely annotated to be included in the IAA calculation
Or any other part of the pipeline
E.g
    $ python get_iaa_agreement.py --specific_annotator_username="annotatoruid"
1. Get all gold standard tracks

This script assumes that all data in the completed folder in the bucket has been annotated
and thus has an annotated_uid column with non-null values


# TODO: Don't re-run get gold standard for every annotator
# TODO: I need to Remove all the list blobs and maybe just cache and create in parallel all the paris instead of looping through
# TODO: Get all completed month, trackId, year, projects in completed
# Then, take this in and segment for the gold standards


1. Loop through all the tracks and get month year trackId, annotator_username, and projects for gold standards, and path
2. If a track is a match add to the list of tracks to compare
3. Compare tracks
"""

import csv
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
from atlantes.human_annotation.constants import (ACTIVITY_TYPE_LABELS,
                                                 COMPLETED_ANNOTATIONS_FOLDER,
                                                 DT_STRING,
                                                 GCP_BUCKET_DOWNSAMPLED,
                                                 GCP_BUCKET_RAW,
                                                 GOLD_STANDARD_PROJECTS)
from atlantes.human_annotation.human_annotation_utils import (
    check_projects_to_exclude,
    get_all_unique_trajectories_identities_in_projects,
    read_downsampled_track_handler)
from atlantes.human_annotation.schemas import (
    AnnotatedDownsampledAISTrackDataFrameModel, AnnotationIdentifierDataModel)
from atlantes.log_utils import get_logger
from google.cloud import storage
from pandera.typing import DataFrame
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

logger = get_logger(__name__)


def build_annotator_username_to_project_map(
    bucket: str, prefix: str = COMPLETED_ANNOTATIONS_FOLDER
) -> dict:
    """Build a map of annotator UIDs to projects.

    Assumes projects are stored in a directory structure like:
    gs://bucket/annotations/completed/project/track.csv

    Assumes that each project has been annotated by a single annotator.
    Thus, we only need to check the first file in each project

    Parameters
    ----------
    bucket : str
        The name of the Google Cloud Storage bucket"""
    client = storage.Client()
    gcp_bucket = client.bucket(bucket)
    # NOTE: The list blobs operation costs money and we should move to a metadata index
    # that is generated when we export the data if this process
    # is going to scale to a large number of files
    blobs = list(gcp_bucket.list_blobs(prefix=prefix, match_glob="**/*.{csv,parquet}"))
    annotator_username_to_project_map: dict[str, list[str]] = {}
    checked_projects = []
    for blob in tqdm(blobs, desc="Finding annotator project mappings"):
        blob_parts = Path(blob.name).parts
        project = blob_parts[-2]

        if check_projects_to_exclude(project) or project in checked_projects:
            continue
        path = f"gs://{bucket}/{blob.name}"
        df = read_downsampled_track_handler(path, nrows=1)
        if df is None:
            continue
        annotator_username = df.iloc[0]["annotator_username"]
        if annotator_username not in annotator_username_to_project_map:
            logger.info(f"Adding annotator {annotator_username} to map")
            annotator_username_to_project_map[annotator_username] = []
        annotator_username_to_project_map[annotator_username].append(project)
        logger.info(f"Adding project {project} to annotator {annotator_username}")
        checked_projects.append(project)
        logger.info(f"Added project {project} to checked projects")
    return annotator_username_to_project_map


def get_unique_gold_standard_track_identities(
    gold_standard_projects: list[str],
    bucket_name: str = GCP_BUCKET_DOWNSAMPLED,
    prefix_dir: str = COMPLETED_ANNOTATIONS_FOLDER,
) -> DataFrame[AnnotationIdentifierDataModel]:
    """Get all gold standard trajectories unique identifiers.

    Annotations are uniquely Identified by year, month, trackId and annotator_username
    """
    client = storage.Client()
    gold_standard_df = get_all_unique_trajectories_identities_in_projects(
        gold_standard_projects, client, bucket_name, prefix_dir
    )
    logger.info(f"Number of gold standard tracks: {len(gold_standard_df)}")
    return gold_standard_df


class SingleAnnotatorIaaReport(object):
    def __init__(
        self,
        annotator_username: str,
        annotator_project_names: list[str],
        annotion_folder_blob: str,
        download_bucket_name: str,
        upload_bucket_name: str,
        gold_standard_project_identities: DataFrame[AnnotationIdentifierDataModel],
        local_output_folder: str = "./iaa_scores",
    ) -> None:
        self.annotator_username = annotator_username
        self.annotator_project_names = annotator_project_names
        self.annotion_folder_blob = annotion_folder_blob
        self.storage_client = storage.Client()
        self.gold_standard_project_identities_df = gold_standard_project_identities
        self.local_output_folder = local_output_folder
        self.activity_labels = ACTIVITY_TYPE_LABELS
        self.download_bucket_name = download_bucket_name
        self.upload_bucket_name = upload_bucket_name
        self.cmap = plt.cm.Blues
        self.IAA_MATCH_COLUMNS = ["year", "month", "trackId"]

    def get_unique_annotator_track_identities(
        self,
    ) -> DataFrame[AnnotationIdentifierDataModel]:
        """Get all annotator trajectories unique identifiers.

        Annotations are uniquely Identified by year, month, trackId and annotator_username
        """
        return get_all_unique_trajectories_identities_in_projects(
            self.annotator_project_names,
            self.storage_client,
            self.download_bucket_name,
            self.annotion_folder_blob,
        )

    def get_iaa_pairs_to_compare(self) -> list[tuple[str, str]]:
        """Get pairs of tracks to compare for inter-annotator agreement."""
        annotator_tracks_df = self.get_unique_annotator_track_identities()
        logger.info(f"Number of annotator tracks: {len(annotator_tracks_df)}")
        logger.info(f"Annotator tracks: {annotator_tracks_df.head()}")
        # We will have 2 dfs and we just want to keep all pairs
        iaa_df = self.gold_standard_project_identities_df.merge(
            annotator_tracks_df, on=self.IAA_MATCH_COLUMNS
        )
        pairs_of_tracks_to_compare = iaa_df[["path_x", "path_y"]].values.tolist()
        if len(pairs_of_tracks_to_compare) == 0:
            logger.info(f"No tracks to compare for annotator {self.annotator_username}")
            raise ValueError(
                f"No gold standard tracks to compare for annotator {self.annotator_username}"
            )
        return pairs_of_tracks_to_compare

    @pa.check_types
    def calculate_iaa_between_two_tracks(
        self,
        annotation_df: DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
        gold_annotated_df: DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
    ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], bool, bool]:
        """Calculate inter-annotator agreement between two tracks.


        Parameters
        ----------
        annotation_df : DataFrame[AnnotatedDownsampledAISTrackDataFrameModel]
            Annotations for the first track.
        gold_annotated_df : DataFrame[AnnotatedDownsampledAISTrackDataFrameModel]
            Annotations for the second track.

        Returns
        -------
        None
        """
        activity_types_annotation = annotation_df["activity_type_name"]
        activity_types_gold = gold_annotated_df["activity_type_name"]
        object_types_annotation = annotation_df["object_type_name"]
        object_types_gold = gold_annotated_df["object_type_name"]
        object_subtype_annotation = annotation_df["object_type_subtype"]
        object_subtype_gold = gold_annotated_df["object_type_subtype"]
        object_type_agrees = (
            object_types_annotation.values[0] == object_types_gold.values[0]
        )
        object_subtype_agrees = (
            object_subtype_annotation.values[0] == object_subtype_gold.values[0]
        )
        object_type_or_subtype_differs_or_is_not_fishing_vessel = (
            not object_type_agrees
            or not object_subtype_agrees
            or object_subtype_annotation.values[0] != "fishing"
        )

        if object_type_or_subtype_differs_or_is_not_fishing_vessel:
            activity_type_iaa = None
            activity_type_agreement_percentage = None
            confusion_matrix_activity_type = None
        else:
            activity_type_iaa = cohen_kappa_score(
                activity_types_annotation,
                activity_types_gold,
                labels=ACTIVITY_TYPE_LABELS,
            )
            activity_type_agreement_percentage = (
                activity_types_annotation == activity_types_gold
            ).sum() / len(activity_types_annotation)
            confusion_matrix_activity_type = confusion_matrix(
                activity_types_annotation,
                activity_types_gold,
                labels=ACTIVITY_TYPE_LABELS,
            )
        return (
            activity_type_iaa,
            activity_type_agreement_percentage,
            confusion_matrix_activity_type,
            object_subtype_agrees,
            object_type_agrees,
        )

    def run_iaa_analysis(self) -> None:
        """Calculate the overall inter-annotator agreement for a single annotator."""
        self.iaa_scores: list[float] = []
        self.agreement_perc: list[float] = []
        self.low_agreement_track_pairs: list[tuple[str, str]] = []
        self.object_subtype_agrees_lst: list[bool] = []
        self.activity_confusion: np.ndarray = np.zeros(
            (len(ACTIVITY_TYPE_LABELS), len(ACTIVITY_TYPE_LABELS)), dtype=np.int64
        )
        for annot_path, gold_path in self.pairs_of_tracks_to_compare:
            annot = read_downsampled_track_handler(annot_path)
            gold = read_downsampled_track_handler(gold_path)
            # I need the track_csv name and the two annotators name for the report
            (
                activity_type_iaa,
                activity_type_agreement_percentage,
                confusion_matrix_activity_type,
                object_subtype_agrees,
                object_type_agrees,
            ) = self.calculate_iaa_between_two_tracks(annot, gold)
            self.object_subtype_agrees_lst.append(object_subtype_agrees)
            if confusion_matrix_activity_type is not None:
                self.activity_confusion += np.array(
                    confusion_matrix_activity_type, dtype=np.int64
                )
            if activity_type_iaa is not None and activity_type_iaa is not pd.NA:
                if activity_type_iaa < 0.5:
                    self.low_agreement_track_pairs.append((annot_path, gold_path))
                self.iaa_scores.append(activity_type_iaa)
                self.agreement_perc.append(activity_type_agreement_percentage)

        logger.info(f"IAA scores: {self.iaa_scores}")
        logger.info(f"Agreement with GOLD percentages: {self.agreement_perc}")
        logger.info(f"Low agreement tracks: {self.low_agreement_track_pairs}")
        logger.info(f"Confusion matrix: \n {self.activity_confusion}")
        logger.info(f"Object subtype agreement: {self.object_subtype_agrees_lst}")

    def plot_confusion_matrix(self, cm: np.ndarray, savepath: str = "test.png") -> None:
        """
        Plots a confusion matrix with labels and title.

        Args:
            cm: NumPy array representing the confusion matrix.
            labels: list of labels for each class.
            title: Title for the plot.
            cmap: Colormap to use for the heatmap.
        """
        title = f"Confusion matrix for {self.annotator_username}"
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=self.cmap)
        plt.colorbar()
        tick_marks = np.arange(len(self.activity_labels))
        plt.xticks(tick_marks, self.activity_labels, rotation=45)
        plt.yticks(tick_marks, self.activity_labels)
        plt.tight_layout()
        plt.xlabel("Annotator Label")
        plt.ylabel("Gold Label")
        plt.title(title)
        plt.savefig(savepath)

    def save_iaa_scores_to_folder(self) -> None:
        """Save the inter-annotator agreement scores to a file."""
        object_agreement = np.mean(np.array(self.object_subtype_agrees_lst))
        # Save to txt file
        output_folder = (
            f"{self.local_output_folder}/{self.annotator_username}/{DT_STRING}"
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = f"{output_folder}/iaa_scores.txt"
        with open(output_path, "w") as f:
            if np.sum(self.activity_confusion) > 0:
                f.write(
                    f" Activity type IAA for annotator {self.annotator_username} \n"
                )
                f.write(f"IAA Scores: {self.iaa_scores} \n")
                f.write(f"IAA summary: {pd.Series(self.iaa_scores).describe()} \n")
                f.write(f"Agreement with GOLD percentages: {self.agreement_perc} \n")
                f.write(
                    f"Agreement with Gold Summary: {pd.Series(self.agreement_perc).describe()} \n"
                )
                f.write(
                    f"Low agreement track pairs (annotator, gold): {self.low_agreement_track_pairs} \n"
                )
                f.write(f"Confusion matrix: \n \n {self.activity_confusion} \n")
                f.write(f"Object subtype agreement: {object_agreement} \n")
                f.write(
                    f"Total number of tracks compared: {len(self.pairs_of_tracks_to_compare)} \n"
                )
                f.write(f"All compared tracks: {self.pairs_of_tracks_to_compare} \n")
            else:
                f.write(
                    f"No tracks compared for annotator {self.annotator_username} \n"
                )

        confusion_matrix_path = f"{output_folder}/confusion_matrix.png"
        self.plot_confusion_matrix(
            self.activity_confusion, savepath=confusion_matrix_path
        )

    def save_low_agreement_csv_to_folder(self) -> None:
        """Save the low-agreement tracks to a csv file."""
        output_folder = (
            f"{self.local_output_folder}/{self.annotator_username}/{DT_STRING}"
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = f"{output_folder}/iaa_low_agreement.csv"
        with open(output_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the column headers
            csvwriter.writerow(
                ["Annotator", "Low agreement track", "Gold standard track"]
            )

            # Write the data rows using zip_longest to handle uneven lists
            if len(self.low_agreement_track_pairs) == 0:
                csvwriter.writerow(
                    [
                        "No low agreement tracks found for annotator",
                        self.annotator_username,
                    ]
                )
                return
            low_agreement_track_pairs: list[tuple[str, str]] = (
                self.low_agreement_track_pairs
            )

            for iaa_pair in low_agreement_track_pairs:
                track, gold_track = iaa_pair
                csvwriter.writerow([self.annotator_username, track, gold_track])

    def publish_iaa_scores_to_gcp(self) -> None:
        """Publish the inter-annotator agreement scores to GCP."""
        bucket = self.storage_client.bucket(self.upload_bucket_name)
        local_folder_path = Path(self.local_output_folder)

        for source_file in local_folder_path.rglob("*"):
            if source_file.is_file():
                logger.info(f"Uploading {source_file}")
                # Construct the destination path using relative path
                dest_path = source_file.relative_to(local_folder_path.parent)
                # Upload the file
                blob = bucket.blob(str(dest_path))
                blob.upload_from_filename(str(source_file))

    def cleanup(self) -> None:
        """Clean up the local output folder."""
        shutil.rmtree(self.local_output_folder)

    def run_iaa_report(self) -> None:
        """Run the inter-annotator agreement report."""

        logger.info(f"Running IAA report for annotator {self.annotator_username}")
        self.pairs_of_tracks_to_compare = self.get_iaa_pairs_to_compare()
        logger.info(f"Calculating IAA for annotator {self.annotator_username}")
        self.run_iaa_analysis()
        self.save_iaa_scores_to_folder()
        self.save_low_agreement_csv_to_folder()
        logger.info(f"Publishing IAA scores for annotator {self.annotator_username}")
        self.publish_iaa_scores_to_gcp()
        self.cleanup()


@click.command()
@click.option(
    "-a",
    "--specific_annotator_username",
    type=str,
    default="",
    help="Specific annotator UID to run IAA report for.",
)
def get_reports_for_all_annotators(
    specific_annotator_username: str,
) -> None:
    """Get inter-annotator agreement reports for all annotators."""
    # make it only work for single annotator
    logger.info("Building annotator UID to project map")
    annotator_usernames_to_projects_map = build_annotator_username_to_project_map(
        GCP_BUCKET_DOWNSAMPLED
    )
    gold_standard_identities_df = get_unique_gold_standard_track_identities(
        GOLD_STANDARD_PROJECTS
    )
    if len(specific_annotator_username) > 0:
        annotator_usernames = [specific_annotator_username]
    else:

        annotator_usernames = list(annotator_usernames_to_projects_map.keys())
    logger.info(f"Number of annotators: {len(annotator_usernames)}")
    for annotator_username in annotator_usernames:
        try:
            annotator_projects = annotator_usernames_to_projects_map[annotator_username]
            report = SingleAnnotatorIaaReport(
                annotator_username,
                annotator_projects,
                COMPLETED_ANNOTATIONS_FOLDER,
                GCP_BUCKET_DOWNSAMPLED,
                GCP_BUCKET_RAW,
                gold_standard_identities_df,
            )
            report.run_iaa_report()
        except KeyError:
            logger.error(f"Annotator {annotator_username} has no completed projects")
            raise KeyError(f"Annotator {annotator_username} has no completed projects")
        except Exception as e:
            logger.info(f"Annotator {annotator_username} failed with error: {e}")
            continue


if __name__ == "__main__":
    get_reports_for_all_annotators()
    logger.info("Inter-annotator agreement reports completed")
