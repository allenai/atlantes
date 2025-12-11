#!/usr/bin/env python3
# type: ignore
"""
Compare int and prod model outputs for AIS data from different providers.

Set up port forwards before running:
  # Int endpoints
  kubectl --context=gke_skylight-int-a-r2d2_us-west1-a_sky-int-a port-forward service/subpath-api 5102:5102 &
  kubectl --context=gke_skylight-int-a-r2d2_us-west1-a_sky-int-a port-forward deployment/subpath-activity-classification-worker-and-atlas 8080:8080 &

  # Prod endpoints (on different ports)
  kubectl --context=gke_skylight-prod-a-a72a_us-west1-a_sky-prod-a port-forward service/subpath-api 5103:5102 &
  kubectl --context=gke_skylight-prod-a-a72a_us-west1-a_sky-prod-a port-forward deployment/subpath-activity-classification-worker-and-atlas 8081:8080 &

CSV Format:
    The CSV file should have columns: int_subpath_id, prod_subpath_id
    Example:
        int_subpath_id,prod_subpath_id
        subpath_123,subpath_456
        subpath_789,subpath_012

Output Artifacts:
    1. comparison_statistics.csv - Database with comparison statistics and difference indicators
    2. confusion_matrix_postprocessing.png - Confusion matrix of post-processed outputs (int vs prod)
    3. confusion_matrix_preprocessing.png - Confusion matrix of pre-processed outputs (int vs prod)

Usage:
    python compare_int_prod_models.py --csv subpath_pairs.csv --output-dir results/
"""
import argparse
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from atlantes.inference.common import AtlasModelTrackInputs
from atlantes.inference.atlas_activity.classifier import (
    AtlasActivityClassifier,
    PipelineInput,
)
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.postprocessor import AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import AtlasActivityPreprocessor
from atlantes.log_utils import get_logger

logger = get_logger("compare_int_prod_models")


def fetch_trajectory(subpath_id: str, subpath_url: str) -> list:
    """Fetch trajectory data from subpath API."""
    resp = requests.get(
        f"{subpath_url}/subpaths/debug/classification/trajectory",
        params={"subpath_id": subpath_id, "model": "atlas_activity", "format": "fastapi"},
        timeout=60
    )
    resp.raise_for_status()

    if not resp.text:
        raise ValueError(f"Empty response from {resp.url}")

    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response. Status: {resp.status_code}, error: {e}")
        logger.error(f"Response text: {resp.text[:500]}")
        raise

    return data["track"]


# Unknown value constants
UNKNOWN_SOG = 102.3  # m/s - filter this out from SOG statistics
UNKNOWN_COG_THRESHOLD = 359  # degrees - filter out values > 359 from COG statistics

# Missing label for confusion matrix when one model has no classification
MISSING_LABEL = "missing"


def calculate_track_statistics(track_data: list) -> dict:
    """Calculate statistics from track data.

    Filters out unknown values:
    - SOG: removes values of 102.3 (UNKNOWN_SOG)
    - COG: removes values > 359 (e.g., 511 is UNKNOWN_COG)
    """
    if not track_data:
        return {
            "sog_mean": np.nan,
            "sog_min": np.nan,
            "sog_max": np.nan,
            "sog_std": np.nan,
            "cog_mean": np.nan,
            "cog_min": np.nan,
            "cog_max": np.nan,
            "cog_std": np.nan,
            "num_messages": 0,
            "last_timestamp": None,
            "avg_time_gap_seconds": np.nan,
        }

    df = pd.DataFrame(track_data)

    # Convert send to datetime if it's a string
    if "send" in df.columns:
        df["send"] = pd.to_datetime(df["send"])

    # Filter SOG: remove NaN and unknown values (102.3)
    sog_values = pd.Series() if "sog" not in df.columns else df["sog"].dropna()
    sog_values = sog_values[sog_values != UNKNOWN_SOG]

    # Filter COG: remove NaN and values > 359 (unknown COG is often 511)
    cog_values = pd.Series() if "cog" not in df.columns else df["cog"].dropna()
    cog_values = cog_values[cog_values <= UNKNOWN_COG_THRESHOLD]

    # Calculate time gaps
    time_gaps = []
    if len(df) > 1 and "send" in df.columns:
        df_sorted = df.sort_values("send")
        time_diffs = df_sorted["send"].diff().dropna()
        time_gaps = time_diffs.dt.total_seconds().dropna().tolist()

    stats = {
        "sog_mean": float(sog_values.mean()) if len(sog_values) > 0 else np.nan,
        "sog_min": float(sog_values.min()) if len(sog_values) > 0 else np.nan,
        "sog_max": float(sog_values.max()) if len(sog_values) > 0 else np.nan,
        "sog_std": float(sog_values.std()) if len(sog_values) > 0 else np.nan,
        "cog_mean": float(cog_values.mean()) if len(cog_values) > 0 else np.nan,
        "cog_min": float(cog_values.min()) if len(cog_values) > 0 else np.nan,
        "cog_max": float(cog_values.max()) if len(cog_values) > 0 else np.nan,
        "cog_std": float(cog_values.std()) if len(cog_values) > 0 else np.nan,
        "num_messages": len(df),
        "last_timestamp": str(df["send"].iloc[-1]) if "send" in df.columns and len(df) > 0 else None,
        "avg_time_gap_seconds": float(np.mean(time_gaps)) if time_gaps else np.nan,
    }

    return stats


def calculate_timestamp_overlap(
    int_track: list,
    prod_track: list,
    tolerance_seconds: float = 1.0,
) -> dict:
    """Calculate timestamp overlap between int and prod tracks.

    Parameters
    ----------
    int_track : list
        Track data from int
    prod_track : list
        Track data from prod
    tolerance_seconds : float
        Tolerance in seconds for considering timestamps as matching (default: 1.0)

    Returns
    -------
    dict
        Dictionary with overlap statistics
    """
    if not int_track or not prod_track:
        return {
            "num_int_timestamps": 0,
            "num_prod_timestamps": 0,
            "num_overlapping_timestamps": 0,
            "overlap_percentage_of_int": 0.0,
            "overlap_percentage_of_prod": 0.0,
            "overlap_percentage_of_union": 0.0,
        }

    int_df = pd.DataFrame(int_track)
    prod_df = pd.DataFrame(prod_track)

    if "send" not in int_df.columns or "send" not in prod_df.columns:
        return {
            "num_int_timestamps": 0,
            "num_prod_timestamps": 0,
            "num_overlapping_timestamps": 0,
            "overlap_percentage_of_int": 0.0,
            "overlap_percentage_of_prod": 0.0,
            "overlap_percentage_of_union": 0.0,
        }

    int_times = pd.to_datetime(int_df["send"])
    prod_times = pd.to_datetime(prod_df["send"])

    num_int = len(int_times)
    num_prod = len(prod_times)

    # Count overlapping timestamps (within tolerance)
    # For efficiency, convert to Unix timestamps
    int_unix = int_times.astype(np.int64) / 1e9  # Convert to seconds
    prod_unix = prod_times.astype(np.int64) / 1e9

    # For each int timestamp, check if there's a prod timestamp within tolerance
    overlapping = 0
    for int_ts in int_unix:
        if np.any(np.abs(prod_unix - int_ts) <= tolerance_seconds):
            overlapping += 1

    # Calculate percentages
    overlap_pct_int = (overlapping / num_int * 100) if num_int > 0 else 0.0
    overlap_pct_prod = (overlapping / num_prod * 100) if num_prod > 0 else 0.0

    # Union-based overlap (Jaccard-style)
    union_size = num_int + num_prod - overlapping
    overlap_pct_union = (overlapping / union_size * 100) if union_size > 0 else 0.0

    return {
        "num_int_timestamps": num_int,
        "num_prod_timestamps": num_prod,
        "num_overlapping_timestamps": overlapping,
        "overlap_percentage_of_int": round(overlap_pct_int, 2),
        "overlap_percentage_of_prod": round(overlap_pct_prod, 2),
        "overlap_percentage_of_union": round(overlap_pct_union, 2),
    }

def classify_trajectory(track_data: list, sidecar_url: str) -> dict:
    """Run classification via sidecar API and normalize output to match local format."""
    track_id = track_data[0]["trackId"]
    classification_request = {
        "tracks": [{"track_id": track_id, "track_data": track_data}]
    }

    resp = requests.post(
        f"{sidecar_url}/classify",
        json=classification_request,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    resp.raise_for_status()
    result = resp.json()

    # Normalize response to match local run_classification output format
    if result.get("predictions") and len(result["predictions"]) > 0:
        prediction = result["predictions"][0]
        details = prediction.get("details", {})
        return {
            "pre_postprocessing_class": details.get("original_classification"),
            "post_postprocessing_class": prediction.get("classification"),
            "confidence": details.get("confidence", np.nan),
            "original_classification": details.get("original_classification"),
            "postprocessed_classification": details.get("postprocessed_classification"),
        }
    elif result.get("preprocess_failures") and len(result["preprocess_failures"]) > 0:
        failure = result["preprocess_failures"][0]
        return {
            "pre_postprocessing_class": None,
            "post_postprocessing_class": None,
            "preprocessing_error": failure.get("error", "Unknown preprocessing error"),
        }
    elif result.get("postprocess_failures") and len(result["postprocess_failures"]) > 0:
        failure = result["postprocess_failures"][0]
        return {
            "pre_postprocessing_class": failure.get("classification"),
            "post_postprocessing_class": None,
            "postprocessing_error": failure.get("error", "Unknown postprocessing error"),
        }
    else:
        return {
            "pre_postprocessing_class": None,
            "post_postprocessing_class": None,
            "error": "No predictions or failures returned",
        }


def run_classification_local(track_data: list, classifier: AtlasActivityClassifier) -> dict:
    """Run classification and return both pre and post postprocessing results."""
    if not track_data:
        return {
            "pre_postprocessing_class": None,
            "post_postprocessing_class": None,
            "error": "Empty track_data",
        }

    track_inputs = [AtlasModelTrackInputs(track_id=track_data[0]["trackId"], track_data=track_data)]
    tracks = [PipelineInput.from_track_data(ti) for ti in track_inputs]

    # Get preprocessed data
    preprocessed_data = []
    for input_data in tracks:
        try:
            preprocessed = classifier.preprocessor.preprocess(input_data.track_data)
            preprocessed_data.append(preprocessed)
        except Exception as e:
            logger.warning(f"Error preprocessing {input_data.track_id=}: {e}")
            return {
                "pre_postprocessing_class": None,
                "post_postprocessing_class": None,
                "preprocessing_error": str(e),
            }

    # Get model output (pre-postprocessing)
    classifications = classifier.model.run_inference(preprocessed_data)

    if not classifications:
        return {
            "pre_postprocessing_class": None,
            "post_postprocessing_class": None,
            "inference_error": "No classifications returned",
        }

    classification = classifications[0]
    pre_postprocessing_class = classification[0]  # AtlasActivityLabelsTraining enum
    pre_postprocessing_class_name = pre_postprocessing_class.name.lower()

    # Get postprocessed output
    try:
        post_postprocessing_class_name, details = classifier.postprocessor.postprocess(classification)
    except Exception as e:
        logger.warning(f"Error postprocessing: {e}")
        return {
            "pre_postprocessing_class": pre_postprocessing_class_name,
            "post_postprocessing_class": None,
            "postprocessing_error": str(e),
        }

    return {
        "pre_postprocessing_class": pre_postprocessing_class_name,
        "post_postprocessing_class": post_postprocessing_class_name,
        "confidence": details.get("confidence", np.nan),
        "original_classification": details.get("original_classification", None),
        "postprocessed_classification": details.get("postprocessed_classification", None),
    }


def analyze_statistical_differences(
    df_results: pd.DataFrame,
    threshold_percentile: float = 90.0,
) -> list[dict]:
    """Analyze differences between int and prod statistics and identify big differences."""
    # Define pairs of int/prod columns to compare
    comparison_pairs = [
        ("sog_mean", "int_sog_mean", "prod_sog_mean"),
        ("sog_min", "int_sog_min", "prod_sog_min"),
        ("sog_max", "int_sog_max", "prod_sog_max"),
        ("sog_std", "int_sog_std", "prod_sog_std"),
        ("cog_mean", "int_cog_mean", "prod_cog_mean"),
        ("cog_min", "int_cog_min", "prod_cog_min"),
        ("cog_max", "int_cog_max", "prod_cog_max"),
        ("cog_std", "int_cog_std", "prod_cog_std"),
        ("num_messages", "int_num_messages", "prod_num_messages"),
        ("avg_time_gap_seconds", "int_avg_time_gap_seconds", "prod_avg_time_gap_seconds"),
    ]

    big_differences = []

    logger.info("Analyzing statistical differences between int and prod...")

    for field_name, int_col, prod_col in comparison_pairs:
        if int_col not in df_results.columns or prod_col not in df_results.columns:
            continue

        int_values = pd.to_numeric(df_results[int_col], errors='coerce')
        prod_values = pd.to_numeric(df_results[prod_col], errors='coerce')

        # Calculate absolute differences
        abs_diff = np.abs(int_values - prod_values)

        # Calculate relative differences (as percentage)
        mean_values = (int_values + prod_values) / 2
        rel_diff = np.where(
            mean_values != 0,
            (abs_diff / np.abs(mean_values)) * 100,
            np.where(abs_diff != 0, np.inf, 0)
        )

        # Calculate statistics
        valid_mask = ~(int_values.isna() | prod_values.isna())
        if valid_mask.sum() == 0:
            continue

        abs_diff_valid = abs_diff[valid_mask]
        rel_diff_valid = rel_diff[valid_mask]
        rel_diff_valid_finite = rel_diff_valid[rel_diff_valid != np.inf]

        if len(rel_diff_valid_finite) == 0:
            continue

        # Find threshold for "big differences"
        abs_threshold = np.percentile(abs_diff_valid, threshold_percentile)
        rel_threshold = np.percentile(rel_diff_valid_finite, threshold_percentile)

        # Find rows with big differences
        big_diff_mask = (abs_diff > abs_threshold) | (
            (rel_diff > rel_threshold) & (rel_diff != np.inf)
        )
        big_diff_rows = df_results[big_diff_mask]

        if big_diff_mask.sum() > 0:
            for idx, row in big_diff_rows.iterrows():
                int_val = row[int_col]
                prod_val = row[prod_col]
                abs_d = abs_diff.iloc[idx]
                rel_d = rel_diff[idx] if rel_diff[idx] != np.inf else None

                # Determine which is higher
                if pd.notna(int_val) and pd.notna(prod_val):
                    if int_val > prod_val:
                        higher = "int"
                    elif prod_val > int_val:
                        higher = "prod"
                    else:
                        higher = "equal"
                else:
                    higher = "unknown"

                big_differences.append({
                    "field": field_name,
                    "int_subpath_id": row.get("int_subpath_id", "N/A"),
                    "prod_subpath_id": row.get("prod_subpath_id", "N/A"),
                    "int_value": float(int_val) if pd.notna(int_val) else None,
                    "prod_value": float(prod_val) if pd.notna(prod_val) else None,
                    "abs_diff": float(abs_d) if pd.notna(abs_d) else None,
                    "rel_diff": float(rel_d) if rel_d is not None and pd.notna(rel_d) else None,
                    "higher": higher,
                    "abs_threshold": float(abs_threshold),
                    "rel_threshold": float(rel_threshold),
                })

    return big_differences


def create_confusion_matrix(
    int_labels: list[str],
    prod_labels: list[str],
    title: str,
    output_path: Path,
    class_names: Optional[list[str]] = None,
) -> None:
    """Create and save a confusion matrix."""
    # Get all unique labels
    all_labels = sorted(set(int_labels + prod_labels))
    all_labels = [label for label in all_labels if label is not None]

    if not all_labels:
        logger.warning(f"No labels found for {title}, skipping confusion matrix")
        return

    # Create confusion matrix
    cm = confusion_matrix(int_labels, prod_labels, labels=all_labels)

    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
    )
    plt.title(title)
    plt.ylabel("Int Model")
    plt.xlabel("Prod Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare int and prod model outputs for AIS data"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file with subpath_id pairs. Expected columns: int_subpath_id, prod_subpath_id",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--int-subpath-url",
        default="http://localhost:5102",
        help="Int subpath API URL",
    )
    parser.add_argument(
        "--int-sidecar-url",
        default="http://localhost:8080",
        help="Int sidecar API URL",
    )
    parser.add_argument(
        "--prod-subpath-url",
        default="http://localhost:5103",
        help="Prod subpath API URL (different port)",
    )
    parser.add_argument(
        "--prod-sidecar-url",
        default="http://localhost:8081",
        help="Prod sidecar API URL (different port)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local models, not sidecar APIs",
    )
    parser.add_argument(
        "--save-tracks",
        action="store_true",
        help="Save track data as JSON files in output directory",
    )
    parser.add_argument(
        "--analyze-differences",
        action="store_true",
        help="Analyze and report statistical differences between int and prod",
    )
    parser.add_argument(
        "--difference-threshold-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for identifying big differences (default: 90.0)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize classifiers (using local models, not sidecar APIs if --local is set)
    if args.local:
        preprocessor = AtlasActivityPreprocessor()
        model = AtlasActivityModel()
        postprocessor = AtlasActivityPostProcessor()
        classifier = AtlasActivityClassifier(
            preprocessor=preprocessor,
            model=model,
            postprocessor=postprocessor,
        )

    # Read CSV
    results = []
    int_post_labels = []
    prod_post_labels = []
    int_pre_labels = []
    prod_pre_labels = []

    # Read the CSV into a DataFrame for efficient processing
    df_pairs = pd.read_csv(args.csv)
    num_rows = len(df_pairs)

    for row_num, row in enumerate(tqdm(df_pairs.itertuples(index=False), total=num_rows, desc="Processing rows"), start=1):
        int_subpath_id = getattr(row, "int_subpath_id", None)
        prod_subpath_id = getattr(row, "prod_subpath_id", None)

        # Treat NaN as None for robustness
        if pd.isna(int_subpath_id):
            int_subpath_id = None
        if pd.isna(prod_subpath_id):
            prod_subpath_id = None

        if not int_subpath_id and not prod_subpath_id:
            logger.warning(f"Row {row_num}: Both subpath_ids missing, skipping")
            continue

        logger.info(f"Processing row {row_num}: int={int_subpath_id}, prod={prod_subpath_id}")

        # Fetch trajectories individually
        int_track = None
        prod_track = None

        if int_subpath_id:
            try:
                int_track = fetch_trajectory(int_subpath_id, args.int_subpath_url)
            except Exception as e:
                logger.error(f"Error fetching int trajectory for row {row_num}: {e}")

        if prod_subpath_id:
            try:
                prod_track = fetch_trajectory(prod_subpath_id, args.prod_subpath_url)
            except Exception as e:
                logger.error(f"Error fetching prod trajectory for row {row_num}: {e}")

        # Save track data if requested
        if args.save_tracks:
            tracks_dir = output_dir / "tracks"
            tracks_dir.mkdir(exist_ok=True)

            if int_track and int_subpath_id:
                int_track_file = tracks_dir / f"int_{int_subpath_id}.json"
                with open(int_track_file, "w") as f:
                    json.dump(int_track, f, indent=2, default=str)

            if prod_track and prod_subpath_id:
                prod_track_file = tracks_dir / f"prod_{prod_subpath_id}.json"
                with open(prod_track_file, "w") as f:
                    json.dump(prod_track, f, indent=2, default=str)

            logger.debug(f"Saved track data to {tracks_dir}")

        # Calculate statistics (only for tracks that exist)
        int_stats = calculate_track_statistics(int_track) if int_track else {}
        prod_stats = calculate_track_statistics(prod_track) if prod_track else {}

        # Calculate timestamp overlap (only if both tracks exist)
        if int_track and prod_track:
            overlap_stats = calculate_timestamp_overlap(int_track, prod_track)
        else:
            overlap_stats = {
                "num_overlapping_timestamps": None,
                "overlap_percentage_of_int": None,
                "overlap_percentage_of_prod": None,
                "overlap_percentage_of_union": None,
            }

        # Run classifications (sidecar or local) - only for tracks that exist
        int_result = {}
        prod_result = {}

        if int_track:
            try:
                if args.local:
                    int_result = run_classification_local(int_track, classifier)
                else:
                    int_result = classify_trajectory(int_track, args.int_sidecar_url)
            except Exception as e:
                logger.error(f"Error classifying int trajectory for row {row_num}: {e}")

        if prod_track:
            try:
                if args.local:
                    prod_result = run_classification_local(prod_track, classifier)
                else:
                    prod_result = classify_trajectory(prod_track, args.prod_sidecar_url)
            except Exception as e:
                logger.error(f"Error classifying prod trajectory for row {row_num}: {e}")

        print(int_result)
        print(prod_result)
        # Collect labels for confusion matrices
        # Use MISSING_LABEL when track/result is missing
        int_post_label = int_result.get("post_postprocessing_class") if int_result else None
        prod_post_label = prod_result.get("post_postprocessing_class") if prod_result else None
        int_pre_label = int_result.get("pre_postprocessing_class") if int_result else None
        prod_pre_label = prod_result.get("pre_postprocessing_class") if prod_result else None

        # Mark as missing if no track was available
        if not int_track:
            int_post_label = MISSING_LABEL
            int_pre_label = MISSING_LABEL
        if not prod_track:
            prod_post_label = MISSING_LABEL
            prod_pre_label = MISSING_LABEL

        # Include in confusion matrix if at least one side has a label
        if int_post_label or prod_post_label:
            int_post_labels.append(int_post_label if int_post_label else MISSING_LABEL)
            prod_post_labels.append(prod_post_label if prod_post_label else MISSING_LABEL)

        if int_pre_label or prod_pre_label:
            int_pre_labels.append(int_pre_label if int_pre_label else MISSING_LABEL)
            prod_pre_labels.append(prod_pre_label if prod_pre_label else MISSING_LABEL)

        # Calculate timestamp difference
        timestamp_diff = None
        if int_stats.get("last_timestamp") and prod_stats.get("last_timestamp"):
            try:
                int_ts = pd.to_datetime(int_stats["last_timestamp"])
                prod_ts = pd.to_datetime(prod_stats["last_timestamp"])
                timestamp_diff = (int_ts - prod_ts).total_seconds()
            except Exception as e:
                logger.warning(f"Error calculating timestamp diff: {e}")

        # Determine if outputs are different
        post_outputs_different = int_post_label != prod_post_label
        pre_outputs_different = int_pre_label != prod_pre_label

        # Store results
        result_row = {
            "int_subpath_id": int_subpath_id,
            "prod_subpath_id": prod_subpath_id,
            # Statistics (use .get() since stats may be empty dict if track missing)
            "int_sog_mean": int_stats.get("sog_mean"),
            "int_sog_min": int_stats.get("sog_min"),
            "int_sog_max": int_stats.get("sog_max"),
            "int_sog_std": int_stats.get("sog_std"),
            "int_cog_mean": int_stats.get("cog_mean"),
            "int_cog_min": int_stats.get("cog_min"),
            "int_cog_max": int_stats.get("cog_max"),
            "int_cog_std": int_stats.get("cog_std"),
            "int_num_messages": int_stats.get("num_messages"),
            "int_last_timestamp": int_stats.get("last_timestamp"),
            "int_avg_time_gap_seconds": int_stats.get("avg_time_gap_seconds"),
            "prod_sog_mean": prod_stats.get("sog_mean"),
            "prod_sog_min": prod_stats.get("sog_min"),
            "prod_sog_max": prod_stats.get("sog_max"),
            "prod_sog_std": prod_stats.get("sog_std"),
            "prod_cog_mean": prod_stats.get("cog_mean"),
            "prod_cog_min": prod_stats.get("cog_min"),
            "prod_cog_max": prod_stats.get("cog_max"),
            "prod_cog_std": prod_stats.get("cog_std"),
            "prod_num_messages": prod_stats.get("num_messages"),
            "prod_last_timestamp": prod_stats.get("last_timestamp"),
            "prod_avg_time_gap_seconds": prod_stats.get("avg_time_gap_seconds"),
            "timestamp_diff_seconds": timestamp_diff,
            # Timestamp overlap metrics
            "num_overlapping_timestamps": overlap_stats.get("num_overlapping_timestamps"),
            "overlap_percentage_of_int": overlap_stats.get("overlap_percentage_of_int"),
            "overlap_percentage_of_prod": overlap_stats.get("overlap_percentage_of_prod"),
            "overlap_percentage_of_union": overlap_stats.get("overlap_percentage_of_union"),
            # Classifications
            "int_pre_postprocessing_class": int_pre_label,
            "int_post_postprocessing_class": int_post_label,
            "prod_pre_postprocessing_class": prod_pre_label,
            "prod_post_postprocessing_class": prod_post_label,
            "int_confidence": int_result.get("confidence", np.nan) if int_result else np.nan,
            "prod_confidence": prod_result.get("confidence", np.nan) if prod_result else np.nan,
            # Difference indicators
            "post_outputs_different": post_outputs_different,
            "pre_outputs_different": pre_outputs_different,
        }

        results.append(result_row)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save comparison statistics database
    output_csv = output_dir / "comparison_statistics.csv"
    df_results.to_csv(output_csv, index=False)
    logger.info(f"Saved comparison statistics to {output_csv}")

    # Analyze statistical differences if requested
    if args.analyze_differences and len(df_results) > 0:
        big_differences = analyze_statistical_differences(
            df_results,
            threshold_percentile=args.difference_threshold_percentile,
        )

        if big_differences:
            # Save big differences to CSV
            df_big_diffs = pd.DataFrame(big_differences)
            big_diffs_csv = output_dir / "big_differences.csv"
            df_big_diffs.to_csv(big_diffs_csv, index=False)
            logger.info(f"Saved big differences analysis to {big_diffs_csv}")

            # Print summary
            print("\n" + "=" * 80)
            print("STATISTICAL DIFFERENCES ANALYSIS")
            print("=" * 80)
            print(f"Total big differences found: {len(big_differences)}")

            # Count which is higher more often
            int_higher = sum(1 for d in big_differences if d.get("higher") == "int")
            prod_higher = sum(1 for d in big_differences if d.get("higher") == "prod")
            equal_count = sum(1 for d in big_differences if d.get("higher") == "equal")
            print(f"Int higher: {int_higher}, Prod higher: {prod_higher}, Equal: {equal_count}")

            # Group by field
            by_field = {}
            for diff in big_differences:
                field = diff["field"]
                if field not in by_field:
                    by_field[field] = []
                by_field[field].append(diff)

            for field, diffs in sorted(by_field.items()):
                field_int_higher = sum(1 for d in diffs if d.get("higher") == "int")
                field_prod_higher = sum(1 for d in diffs if d.get("higher") == "prod")
                print(f"\n{field}: {len(diffs)} big differences (int higher: {field_int_higher}, prod higher: {field_prod_higher})")
                for diff in diffs[:5]:  # Show first 5
                    int_id = str(diff['int_subpath_id'])[:8] if diff['int_subpath_id'] != "N/A" else "N/A"
                    prod_id = str(diff['prod_subpath_id'])[:8] if diff['prod_subpath_id'] != "N/A" else "N/A"
                    print(f"  Int={int_id}..., Prod={prod_id}...")
                    int_val = diff['int_value'] if diff['int_value'] is not None else 0
                    prod_val = diff['prod_value'] if diff['prod_value'] is not None else 0
                    abs_d = diff['abs_diff'] if diff['abs_diff'] is not None else 0
                    rel_d = diff['rel_diff'] if diff['rel_diff'] is not None else 0
                    higher = diff.get('higher', 'unknown')
                    print(f"    Int={int_val:.4f}, Prod={prod_val:.4f}, "
                          f"AbsDiff={abs_d:.4f}, RelDiff={rel_d:.2f}%, Higher={higher}")
                if len(diffs) > 5:
                    print(f"  ... and {len(diffs) - 5} more")
        else:
            print("\nNo big statistical differences found.")

    # Generate confusion matrices
    if int_post_labels and prod_post_labels:
        create_confusion_matrix(
            int_post_labels,
            prod_post_labels,
            "Confusion Matrix: Post-Processing Outputs (Int vs Prod)",
            output_dir / "confusion_matrix_postprocessing.png",
        )

    if int_pre_labels and prod_pre_labels:
        create_confusion_matrix(
            int_pre_labels,
            prod_pre_labels,
            "Confusion Matrix: Pre-Processing Outputs (Int vs Prod)",
            output_dir / "confusion_matrix_preprocessing.png",
        )

    # Print summary
    print("\nSummary:")
    print(f"Total pairs processed: {len(results)}")
    if len(results) > 0:
        post_different_count = sum(1 for r in results if r.get("post_outputs_different", False))
        pre_different_count = sum(1 for r in results if r.get("pre_outputs_different", False))
        print(f"Post-processing outputs different: {post_different_count} / {len(results)} ({100*post_different_count/len(results):.1f}%)")
        print(f"Pre-processing outputs different: {pre_different_count} / {len(results)} ({100*pre_different_count/len(results):.1f}%)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
