"""Script for seeing how the number of fishing events changed on local files
"""

import logging
from pathlib import Path

import click
from atlantes.inference.atlas_activity.model import AtlasActivityModel
from atlantes.inference.atlas_activity.pipeline import AtlasActivityClassifier
from atlantes.inference.atlas_activity.postprocessor import \
    AtlasActivityPostProcessor
from atlantes.inference.atlas_activity.preprocessor import \
    AtlasActivityPreprocessor
from atlantes.utils import read_df_file_type_handler
from tqdm import tqdm


def setup_pipeline() -> AtlasActivityClassifier:
    return AtlasActivityClassifier(
        preprocessor=AtlasActivityPreprocessor(),
        model=AtlasActivityModel(),
        postprocessor=AtlasActivityPostProcessor(),
    )


def process_track(
    pipeline: AtlasActivityClassifier, track_path: str
) -> tuple[str, dict]:
    track_df = read_df_file_type_handler(track_path)
    return pipeline.run_pipeline(track_df)  # type: ignore


def write_results(tracks: list[str], output_file: str) -> None:
    with open(output_file, "w") as f:
        for track in tracks:
            f.write(f"{track}\n")


@click.command()
@click.option(
    "--input_folder", type=click.Path(exists=True), help="Folder containing track files"
)
@click.option(
    "--output_file",
    type=click.Path(),
    default="fishing_not_predicted_tracks.csv",
    help="Output file for tracks not predicted as fishing",
)
def run_through_negatives(input_folder: str, output_file: str) -> None:
    """Apply the model to each track in a given folder and record the model outputs per class."""

    logging.getLogger("ray.serve").setLevel(logging.ERROR)

    pipeline = setup_pipeline()
    tracks = [str(path) for path in Path(input_folder).rglob("*.parquet")]

    num_fishing_events = 0
    total_tracks_processed = 0
    fishing_not_predicted_tracks = []

    for track in tqdm(tracks):
        activity_class, details = process_track(pipeline, track)
        total_tracks_processed += 1

        if activity_class == "fishing":
            num_fishing_events += 1
            percent_fishing_events = num_fishing_events / total_tracks_processed
            click.echo(f"Percent of fishing events: {percent_fishing_events}")
        else:
            click.echo(f"Track {track} is not a fishing event")
            click.echo(f"{activity_class}, {details}")
            fishing_not_predicted_tracks.append(track)

    write_results(fishing_not_predicted_tracks, output_file)
    click.echo(f"Number of fishing events: {num_fishing_events}")


if __name__ == "__main__":
    run_through_negatives()
