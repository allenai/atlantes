"""
Script to visualize a month of worldwide marine traffic data in a GeoTIFF file.

Requires data from GCP stored locally

gcloud alpha storage cp gs://ais_2023_subpath /path/to/ais_2023_subpath --recursive

python atlantes/human_annotation/visualize_monthly_marine_traffic.py \
    --root_dir "/path/to/ais_2023_subpath" \
    --file_type "parquet" \
    --year 2023

TODO:
The script uses the 'find' command to locate files in the specified directory.
This is not secure and should be replaced with a more secure method.
There are now metadata files that enable bypassing the use of find entirely and using
the metadata to directly locate the required files.

"""

import os
import subprocess  # nosec
from typing import Optional

import click
import colorcet as cc
import dask
import dask.dataframe as dd
import datashader as ds
import numpy as np
import pandas as pd
import rasterio
from atlantes.human_annotation.constants import (
    BOUNDARY_COORDS,
    COORD_REF_SYS,
    WIDTH,
    MonthsToLoad,
)
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.data_annotate_utils import get_ais_vessel_category
from dask.diagnostics import ProgressBar
from datashader import transfer_functions as tf
from PIL import Image
from rasterio.transform import from_bounds
from tqdm import tqdm

logger = get_logger(__name__)


colors = cc.glasbey_bw_minc_20_minl_30


def find_files_with_find_command(
    root_path: str, pattern: str, vessel_category_nums: Optional[list[int]] = None
) -> list:
    """This function uses the 'find' command to find files in an efficient way, because
    searching millions of files via Python's glob module is very slow. This function
    is not secure, due to its reliance on subprocess module, and we expect to replace it
    near term with pre-computed metadata files that provides the required paths matching
    the desired attributes directly, without searching.

    Parameters
    ----------
    root_path : str
    pattern : str
    vessel_category_nums : list[int], optional

    Returns
    -------
    list

    """
    try:
        # Prepare the base command
        command = ["find", root_path, "-type", "f"]

        # If there are specific vessel category numbers, construct the path filter
        if vessel_category_nums:
            # Start the group condition
            group_conditions = ["("]  # Begin grouping
            for num in vessel_category_nums:
                category_path = f"{root_path}/{num}/*"
                group_conditions.extend(["-path", category_path, "-o"])
            group_conditions.pop()  # Remove the last "-o"
            group_conditions.append(")")  # End grouping
            command.extend(group_conditions)

        # Add the name pattern at the end of the command
        command += ["-name", pattern]

        logger.info("Command to be executed: %s", " ".join(command))
        # Execute the command without invoking a shell
        completed_process = subprocess.run(
            command, capture_output=True, text=True, check=True
        )  # nosec

        # Execute the command capturing both stdout and stderr
        completed_process = subprocess.run(
            command, capture_output=True, text=True
        )  # nosec
        if completed_process.returncode != 0:
            logger.error("Error output: %s", completed_process.stderr)
            return []
        else:
            logger.info("Command output: %s", completed_process.stdout)

        # Process the output to create a list of paths
        paths = completed_process.stdout.strip().split("\n")
        paths = [path for path in paths if path]  # Remove empty entries
        return paths

    except Exception as e:
        logger.exception("An error occurred: %s", e)
        return []


def load_data(
    data_path: str,
    data_format: str,
    formatted_month: int,
    vessel_categories: Optional[list[str]] = None,
) -> dd.DataFrame:
    """Loads the data from the specified format files and converts it to a Dask dataframe
    that includes only lat and lon columns for specified vessel categories."""

    dfcols = ["lat", "lon"]  # Only load lat and lon, ais_type

    if data_format == "csv":
        file_pattern = f"*_{formatted_month}.csv"
    elif data_format == "parquet":
        file_pattern = f"*_{formatted_month}.parquet"
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    vessel_category_nums = get_ais_vessel_category(vessel_categories)

    logger.info(f"{vessel_categories=}")
    logger.info(f"{vessel_category_nums=}")

    # Use find command to locate files
    matching_files = find_files_with_find_command(
        str(data_path), file_pattern, vessel_category_nums
    )

    if not matching_files:
        logger.warning(
            f"No files found matching the pattern for month {formatted_month}."
        )
        return dd.from_pandas(pd.DataFrame(columns=dfcols), npartitions=1)

    logger.info(f"Found {len(matching_files)} files for month {formatted_month}.")

    with ProgressBar():
        tasks = []
        for file in tqdm(matching_files, desc="Files processed"):
            df = dask.delayed(pd.read_parquet)(file, columns=dfcols)
            tasks.append(df)
        results = dask.compute(*tasks)
    df = pd.concat(results)
    logger.info(f"finished writing for month {formatted_month}")

    return df  # Returning Dask DataFrame for further computation


def create_georeferenced_image(
    df: pd.DataFrame,
    output_dir: str,
    month: int,
    year: int,
    width: int,
    coords: dict,
    vessel_category: str,
) -> None:
    """
    Create a georeferenced image from a DataFrame of points.

    The function now dynamically calculates the height of the output image
    based on the aspect ratio of the geographical bounds to ensure that the
    entire geographical area is accurately represented.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'lon' and 'lat' columns.
    output_dir : str
        Output path for the GeoTIFF file.
    month : int
        Month for which to generate the image.
    year : int
        Year for which to generate the image.
    width : int
        Width of the output image in pixels.
    bounds : Tuple[float, float, float, float]
        Tuple of (min_lon, min_lat, max_lon, max_lat).
    vessel_category : str, optional
        Vessel category to include, by default 'all'.

    Returns
    -------
    None
        GeoTIFF file is written to output_path.
    """

    # Calculate the geographical width and height
    height = width // 2

    # Create a canvas with the specified bounds and size
    cvs = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(coords["WEST"], coords["EAST"]),
        y_range=(coords["SOUTH"], coords["NORTH"]),
    )

    # Aggregate the data
    agg = cvs.points(df, "lon", "lat")

    # Create an image
    img = tf.shade(agg, how="eq_hist")
    img = tf.set_background(img, "black")

    # Convert Datashader image to NumPy array

    img_data = np.array(img.to_pil())

    pil_img = Image.fromarray(img_data)

    png_path = os.path.join(output_dir, f"ais_{month}_{year}_{vessel_category}.png")
    pil_img.save(png_path)

    # Define transformation for the georeferenced image
    transform = from_bounds(
        coords["WEST"], coords["SOUTH"], coords["EAST"], coords["NORTH"], width, height
    )

    # Write to a GeoTIFF file
    gtiff_path = png_path.replace(".png", ".tif")
    with rasterio.open(
        gtiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img_data.dtype,
        crs=COORD_REF_SYS,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img_data[:, :, i], i + 1)

    logger.info(f"GeoTIFF saved to {gtiff_path}")


@click.command()
@click.option("--root_dir", required=True, type=str)
@click.option("--file_type", type=click.Choice(["csv", "parquet"]), default="csv")
@click.option("--width", default=WIDTH, type=int)
@click.option("--start_month", default=0, type=int)
@click.option("--end_month", default=12, type=int)
@click.option("--year", default=2023, type=int)
@click.option(
    "--coords",
    default=BOUNDARY_COORDS,
    type=dict,
    help="Custom geographic coordinates in the format 'WEST EAST NORTH SOUTH'. Uses default boundaries if not specified.",
)
@click.option(
    "--vessel_categories",
    multiple=True,
    default=["Fishing"],
    type=str,
    help="List of vessel categories to include. Can be specified multiple times.",
)
def generate_marine_traffic_maps(
    root_dir: str,
    file_type: str,
    width: int,
    start_month: int,
    end_month: int,
    year: int,
    coords: dict,
    vessel_categories: list[str],
) -> None:
    """
    Generate monthly marine traffic maps from AIS data.

    This script processes AIS data, filtering for specified track IDs associated with fishing activities or unknown vessel categories, to visualize marine traffic worldwide. The visualization is overlayed on a GeoTIFF map for each month within a specified range. The input AIS data can be in CSV or Parquet format.

    Parameters
    ----------
    root_dir : str
        Path to the directory containing the AIS data files.
    file_type : str
        The format of the input AIS data files ('csv' or 'parquet').
    width : int
        Width of the output image in pixels.
    height : int
        Height of the output image in pixels.
    start_month : int
        The starting month for which to generate maps (inclusive).
    end_month : int
        The ending month for which to generate maps (exclusive).
    year : int
        The year for which to generate marine traffic maps.
    coords : dict
        Dictionary of geographic coordinates in the format 'WEST EAST NORTH SOUTH'.
    vessel_categories : list[str]
        List of vessel categories to include.

    Returns
    -------
    None
        GeoTIFF and PNG images visualizing monthly marine traffic are saved to the specified output directory.

    """

    if (
        width > 15000
    ):  # todo change this so that we make pieces of the map to enable higher resolution
        raise
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)  # Directory of the script
    output_dir = os.path.join(script_dir, f"world_AIS_maps_{year}")
    os.makedirs(output_dir, exist_ok=True)
    months_num_str = [month.value for month in MonthsToLoad]
    months_requested = months_num_str[start_month:end_month]
    vessel_categories = ["Fishing"]
    for month in tqdm(months_requested, desc="Processing AIS data for each month"):

        logger.info(f"Plotting marine traffic for month {month}...")
        for vessel_category in vessel_categories:
            df = load_data(
                root_dir,
                file_type,
                formatted_month=month,
                vessel_categories=[vessel_category],
            )
            create_georeferenced_image(
                df, output_dir, month, year, width, coords, vessel_category
            )

    vessel_categories = ["Unknown"]
    for month in tqdm(months_requested, desc="Processing AIS data for each month"):

        logger.info(f"Plotting marine traffic for month {month}...")
        for vessel_category in vessel_categories:
            df = load_data(
                root_dir,
                file_type,
                formatted_month=month,
                vessel_categories=[vessel_category],
            )
            create_georeferenced_image(
                df, output_dir, month, year, width, coords, vessel_category
            )


logger.info("Computation complete!")

if __name__ == "__main__":
    generate_marine_traffic_maps()
