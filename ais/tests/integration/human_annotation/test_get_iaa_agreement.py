"""Module for testing the creation of IAA reports.

Because this is an offline process, we do not test the publishing to gcp"""

import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import pytest
from atlantes.human_annotation.constants import DT_STRING, GCP_BUCKET_RAW
from atlantes.human_annotation.get_iaa_agreement import (
    SingleAnnotatorIaaReport, build_annotator_username_to_project_map,
    get_unique_gold_standard_track_identities)
from atlantes.human_annotation.human_annotation_utils import \
    load_downsampled_track_df
from atlantes.human_annotation.schemas import \
    AnnotatedDownsampledAISTrackDataFrameModel
from google.cloud import storage
from pandera.typing import DataFrame

GOLD_STANDARD_PROJECTS = ["test_goldstandardproj"]


@pytest.fixture(scope="class")
def make_single_annotator_iaa_report(gcp_test_projects_folder_name: str) -> Callable:
    """Return a SingleAnnotatorIaaReport instance."""

    def make(
        annotator_username: str,
        annotator_projects: list[str],
        annotion_folder_blob: str = gcp_test_projects_folder_name,
        download_bucket_name: str = GCP_BUCKET_RAW,
        upload_bucket_name: str = GCP_BUCKET_RAW,
        gold_standard_projects: pd.DataFrame = get_unique_gold_standard_track_identities(
                GOLD_STANDARD_PROJECTS,
                GCP_BUCKET_RAW,
                gcp_test_projects_folder_name,
            ),
        local_output_folder: str = str(Path(__file__).parent / "temp-iaa-test"),
    ) -> SingleAnnotatorIaaReport:
        return SingleAnnotatorIaaReport(
            annotator_username=annotator_username,
            annotator_project_names=annotator_projects,
            annotion_folder_blob=annotion_folder_blob,
            download_bucket_name=download_bucket_name,
            upload_bucket_name=upload_bucket_name,
            gold_standard_project_identities=gold_standard_projects,
            local_output_folder=local_output_folder,
        )

    return make


@pytest.fixture(scope="class")
def pair_of_tracks_to_compare(test_projects_folder: str) -> Tuple[
    DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
    DataFrame[AnnotatedDownsampledAISTrackDataFrameModel],
]:
    """Return a pair of tracks to compare."""
    annotated_project = load_downsampled_track_df(
        os.path.join(test_projects_folder, "test_proj1", "B_664512000_1583828216_2798418_963143_05_0.csv")
    )
    gold_standard_project = load_downsampled_track_df(
        os.path.join(test_projects_folder, "test_goldstandardproj", "B_664512000_1583828216_2798418_963143_05_0.csv")
    )
    return (gold_standard_project, annotated_project)


class TestSingleAnnotatorIaaReport:
    """Test the SingleAnnotatorIaaReport class."""

    def test_get_iaa_pairs_to_compare(
        self,
        gcp_test_projects_folder_name: str,
        test_projects_folder: str,
        make_single_annotator_iaa_report: Callable,
    ) -> None:
        """Test the setup_tracks_for_iaa method."""
        expected_pairs_of_tracks_to_compare = [
            [
                f"{test_projects_folder}/completed/test_goldstandardproj/B_664512000_1583828216_2798418_963143_05_0.csv",
                f"{test_projects_folder}/completed/test_proj1/B_664512000_1583828216_2798418_963143_05_0.csv",
            ],
            [
                f"{test_projects_folder}/completed/test_goldstandardproj/B_664527000_1602662335_1208844_851028_04_0.csv",
                f"{test_projects_folder}/completed/test_proj1/B_664527000_1602662335_1208844_851028_04_0.csv",
            ],
            [
                f"{test_projects_folder}/completed/test_goldstandardproj/B_664530412_1530084529_2356617_862333_02_0.csv",
                f"{test_projects_folder}/completed/test_proj1/B_664530412_1530084529_2356617_862333_02_0.csv",
            ],
            [
                f"{test_projects_folder}/completed/test_goldstandardproj/B_664543000_1584861079_2293229_851342_02_0.csv",
                f"{test_projects_folder}/completed/test_proj1/B_664543000_1584861079_2293229_851342_02_0.csv",
            ],
        ]
        annotator_username = "test_annotator"
        annotator_map = build_annotator_username_to_project_map(
            GCP_BUCKET_RAW, gcp_test_projects_folder_name
        )
        annotator_projects = annotator_map[annotator_username]
        single_annotator_iaa_report: SingleAnnotatorIaaReport = (
            make_single_annotator_iaa_report(
                annotator_username=annotator_username,
                annotator_projects=annotator_projects,
            )
        )
        pairs_of_tracks_to_compare = (
            single_annotator_iaa_report.get_iaa_pairs_to_compare()
        )
        assert pairs_of_tracks_to_compare == expected_pairs_of_tracks_to_compare

    def test_run_iaa_analysis(
        self,
        make_single_annotator_iaa_report: Callable,
        gcp_test_projects_folder_name: str,
        test_projects_folder: str,
    ) -> None:
        """Test the run_iaa_analysis method."""
        annotator_username = "test_annotator"
        annotator_map = build_annotator_username_to_project_map(
            GCP_BUCKET_RAW, gcp_test_projects_folder_name
        )
        annotator_projects = annotator_map[annotator_username]
        single_annotator_iaa_report: SingleAnnotatorIaaReport = (
            make_single_annotator_iaa_report(
                annotator_username=annotator_username,
                annotator_projects=annotator_projects,
            )
        )
        single_annotator_iaa_report.pairs_of_tracks_to_compare = (
            single_annotator_iaa_report.get_iaa_pairs_to_compare()
        )
        single_annotator_iaa_report.run_iaa_analysis()

        expected_low_agreement_track_pairs = [
            (
                f"{test_projects_folder}/completed/test_goldstandardproj/B_664543000_1584861079_2293229_851342_02_0.csv",
                f"{test_projects_folder}/completed/test_proj1/B_664543000_1584861079_2293229_851342_02_0.csv",
            )
        ]
        assert single_annotator_iaa_report.iaa_scores == [0.20692936545698348]
        assert single_annotator_iaa_report.agreement_perc == [0.5936842105263158]
        assert (
            single_annotator_iaa_report.low_agreement_track_pairs
            == expected_low_agreement_track_pairs
        )
        assert np.all(
            single_annotator_iaa_report.activity_confusion
            == np.array(
                [
                    [230, 193, 0, 0, 0, 0, 0],
                    [0, 52, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )
        assert single_annotator_iaa_report.object_subtype_agrees_lst == [
            False,
            True,
            False,
            True,
        ]

    def test_save_iaa_scores_to_folder(
        self,
        make_single_annotator_iaa_report: Callable,
        gcp_test_projects_folder_name: str,
    ) -> None:
        """Test the save_iaa_scores_to_folder method."""
        annotator_username = "test_annotator"
        annotator_map = build_annotator_username_to_project_map(
            GCP_BUCKET_RAW, gcp_test_projects_folder_name
        )
        annotator_projects = annotator_map[annotator_username]
        single_annotator_iaa_report: SingleAnnotatorIaaReport = (
            make_single_annotator_iaa_report(
                annotator_username=annotator_username,
                annotator_projects=annotator_projects,
            )
        )
        single_annotator_iaa_report.pairs_of_tracks_to_compare = (
            single_annotator_iaa_report.get_iaa_pairs_to_compare()
        )
        single_annotator_iaa_report.run_iaa_analysis()
        single_annotator_iaa_report.save_iaa_scores_to_folder()

        output_txt_file = (
            Path(single_annotator_iaa_report.local_output_folder)
            / single_annotator_iaa_report.annotator_username
            / DT_STRING
            / "iaa_scores.txt"
        )
        assert os.path.exists(output_txt_file)
        with open(output_txt_file, "r") as f:
            first_line = f.readline()
        assert first_line == " Activity type IAA for annotator test_annotator \n"

        output_png_file = (
            Path(single_annotator_iaa_report.local_output_folder)
            / single_annotator_iaa_report.annotator_username
            / DT_STRING
            / "confusion_matrix.png"
        )
        assert os.path.exists(output_png_file)
        # clean up
        single_annotator_iaa_report.cleanup()

    def test_publish_iaa_scores_to_gcp(
        self,
        make_single_annotator_iaa_report: Callable,
        gcp_test_projects_folder_name: str,
    ) -> None:
        """Test the publish_iaa_scores_to_gcp method."""
        annotator_username = "test_annotator"
        annotator_map = build_annotator_username_to_project_map(
            GCP_BUCKET_RAW, gcp_test_projects_folder_name
        )
        annotator_projects = annotator_map[annotator_username]
        single_annotator_iaa_report: SingleAnnotatorIaaReport = (
            make_single_annotator_iaa_report(
                annotator_username=annotator_username,
                annotator_projects=annotator_projects,
            )
        )
        single_annotator_iaa_report.run_iaa_report()
        client = storage.Client()
        bucket = client.get_bucket(GCP_BUCKET_RAW)
        blob_txt = bucket.blob(
            f"temp-iaa-test/test_annotator/{DT_STRING}/confusion_matrix.png"
        )
        blob_png = bucket.blob(
            f"temp-iaa-test/test_annotator/{DT_STRING}/iaa_scores.txt"
        )
        assert blob_txt.exists()
        assert blob_png.exists()
        blob_txt.delete()
        blob_png.delete()
