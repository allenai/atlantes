"""This scripts generates a csv file containing trackIds and the labels for the buoy/vessel classification task.
Dask is use to parallelize the annotation process. Reccomended to run this on a
machine with at least 8 cores and 32GB of RAM to effectively paralellize.

NOTE: number of files per loop is a hyperparameter that should be tuned to your machines hardware

1. Set up GCLOUD credentials on your machine using gcloud default application credentials
2. Ensure the bucket name is correct and the track data is in the specified directory

Example
-------
python3 gen_entity_annotations.py \
    gen
    --metadata-index-path your path
    --bucket-name yourbucketname \
    --output-dir path/to/output/dir \
    --num-files-per-loop 100000 \

bucket-name: name of the gcloud bucket where the data is stored

root-dir: path to the input data

output-dir: path to the output data

num-files-per-loop: number of files to process per loop, should be tuned to your machines hardware

#TODO: Add validation step for buoy vessel task"""

from datetime import datetime

import click
from atlantes.datautils import GCP_TRACK_DATA_BUCKET
from atlantes.machine_annotation.buoy_vessel_annotate import \
    generate_buoy_vessel_labels

DT_STRING = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


@click.group()
def gen_val_buoy_vessel_machine_annotations() -> None:
    """Group of commands to generate and validate machine annotations"""
    pass


@gen_val_buoy_vessel_machine_annotations.command()
@click.option(
    "--metadata-index-path",
    type=str,
    required=True,
    help="Path to the metadata index file",
)
@click.option(
    "--bucket-name",
    type=str,
    default=GCP_TRACK_DATA_BUCKET,
    help="name of gcloud bucket",
)
@click.option(
    "--output-dir",
    default=f"2023/labels/entity_class_labels-{DT_STRING}",
    type=str,
    help="path to the output data",
)
@click.option(
    "--num-files-per-loop",
    type=int,
    required=False,
    default=100000,
    help="Minimum number of files to process per loop",
)
def gen(
    metadata_index_path: str, bucket_name: str, output_dir: str, num_files_per_loop: int
) -> None:
    """Generates a csv file containing the paths to the incremental data and the labels for the buoy vessel task

    Process can be viewed by going to the dask dashboard link and tunneling
    gcloud compute ssh [vm-name] -- -NL [localport]:localhost:8787
    and then going to https://localhost:2222/status
    Parameters
    ----------
    metadata_index_path : str
        path to the metadata index file
    bucket_name : str
        name of gcloud bucket for uploading the labels
    output_dir : str
        path to the output data
    num_files_per_loop : int
        number of files to process per loop, should be tuned to your machines hardware
    """
    generate_buoy_vessel_labels(
        metadata_index_path, bucket_name, output_dir, num_files_per_loop
    )


if __name__ == "__main__":
    gen_val_buoy_vessel_machine_annotations()
