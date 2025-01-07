"""
gen_vessel_dict_from_avro.py

This script converts a set of vessel avro files into a single CSV for use as a lookup for
training data for ATLAS. It's designed to handle vessel data stored in GCP's avro files.

Usage:
    python gen_vessel_dict_from_avro.py --data-dir /path/to/your/data

You can also specify the number of cores to use with the `--num-cores` flag:
    python gen_vessel_dict_from_avro.py --data-dir /path/to/your/data --num-cores 4

By default, the script will use the number of cores on the machine minus 2. If the machine has 2 or fewer cores,
it'll default to using 1 core.

We recommend running this script on a machine with at least 32 cores, which will result
in a runtime of approximately 1 minute to complete the copy and conversion. The final
CSV will be approximately 2 GB in size (as of late 2023) -- expect that number to grow
over time as additional vessels are added to the database.

Before using this script, ensure you've copied most recent avro file from avro files from the desired GCP bucket into a local directory.
    gcloud alpha storage cp gs://skylight-data-sky-int-a-wxbc/vessel/vessel-snapshot/vessel.2023-11-08/*.avro .

"""

import csv
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
from atlantes.log_utils import get_logger
from fastavro import reader
from tqdm import tqdm

logger = get_logger(__name__)


delete_keys = [
    "track_id",
    "attribution",
    "updated",
    "fs_vessel_type",
    "owner",
    "owner_address",
    "owner_country_code",
    "registered_owner",
    "registered_owner_address",
    "registered_owner_country_code",
    "beneficial_owner",
    "beneficial_owner_address",
    "beneficial_owner_country_code",
    "operator",
    "operator_address",
    "operator_country_code",
    "tonnage",
    "cc_code",
    "fs_vessel_id",
]


def avro_to_csv(vessel_data_dir: str, filename: Path) -> None:
    """Reads a track increment avro file and writes to csv."""
    head = True
    output_path = os.path.join(vessel_data_dir, f"{filename.stem}.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w+") as outfile:
        f = csv.writer(outfile)
        with open(filename, "rb") as fo:
            avro_reader = reader(fo)
            for emp in tqdm(avro_reader):
                track_ids = emp["track_id"]

                for key in delete_keys:
                    if key in emp:
                        emp.pop(key)

                for track_id in track_ids:
                    emp["unique_id"] = track_id
                    if head:
                        header = emp.keys()
                        f.writerow(header)
                        head = False
                    f.writerow(emp.values())


def merge_csvs(vessel_data_dir: str) -> None:
    """Merges all csvs in vessel_data_dir into a single csv."""
    files = list(Path(vessel_data_dir).rglob("*.csv"))
    with open(os.path.join(vessel_data_dir, "vessel_metadata.csv"), "w") as outfile:
        for filename in files:
            with open(filename) as infile:
                outfile.write(infile.read())


@click.command()
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True),
    default="./vessel_data",
    help="Directory containing the vessel data.",
)
@click.option(
    "--num-cores",
    type=int,
    default=cpu_count() - 2,
    help="Number of cores to use for processing.",
)
def main(data_dir: str, num_cores: int) -> None:
    """Converts vessel avro files into a single csv."""
    files = list(Path(data_dir).rglob("*.avro"))
    logger.info(f"Found {len(files)} avro files.")
    with Pool(num_cores) as par_pool:
        par_pool.starmap(avro_to_csv, [(data_dir, file) for file in files])
    logger.info("Merging CSVs...")
    merge_csvs(data_dir)


if __name__ == "__main__":
    main()
