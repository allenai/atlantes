"""
Note that this module references a file named standard_rendezvous.csv, which is
generated from a ES query. Regenerate it to ensure that it is up to date.

Note that we could use both 0 and 1 to 2x our data, but that would mean training on
both sides of the TSR, which probably is not a good idea, so here we are just going to
training against a single vessel id, as there is plenty of data, and then we don't have
to worry about the possible data leakage/careful stratification.
"""
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import dask.dataframe as dd
import pandas as pd
import spatialpandas as sp
import yaml
from datautils import DFCOLS, EXAMPLE
from holoviews.util.transform import lon_lat_to_easting_northing as ll2en

logger = logging.getLogger(__name__)

CONFIG_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
CONFIG_DIR = os.environ.get("ATLANTES_CONFIG_DIR", CONFIG_DIR_DEFAULT)
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yml")

DT_STRING = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
logging.basicConfig(filename=f"{DT_STRING}.log", encoding="utf-8", level=logging.DEBUG)

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["utils"]

LABEL_MAPPING = config["OSR_TRAINING"]
LABEL_STR_TO_INT = {v: k for k, v in LABEL_MAPPING.items()}
CSV_FILES_INPUT = "/home/patrickb/csvs/"
OUTPUT_DIR = "/home/patrickb/tsr-tracks/"
YEAR_TO_PROCESS = 2022
EVENTS_PATH = "/home/patrickb/skylight-ml/ais/data/machine_annotations/tsr/standard_rendezvous.csv"

ES_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
AVRO_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
VESSEL_ID_TO_EXTRACT = "vessels.vessel_0.track_id"
ES_TRACK_ID_COL = "trackId"


config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")

NAV_NAN = 9999
VES_NAN = 9999

MONTHS_TO_LOAD = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]


def convert_partition(df: dd) -> sp.GeoDataFrame:
    """converts a partition of a Dask DataFrame to a spatialpandas DataFrame"""
    east, north = ll2en(df.lon.astype("float32"), df.lat.astype("float32"))
    return sp.GeoDataFrame(
        {
            "geometry": sp.geometry.PointArray((east, north)),
            "mmsi": df.mmsi.fillna(0).astype("int32"),
            "category": df.ais_type.fillna(VES_NAN).astype("int32"),
            "nav": df.nav.fillna(NAV_NAN).astype("int32"),
            "trackId": df.trackId.fillna("none").astype("str"),
            "name": df.name.fillna("none").astype("str"),
            "send": df.send.fillna("none").astype("str"),
            "lon": df.lon.fillna(0).astype("float32"),
            "lat": df.lat.fillna(0).astype("float32"),
            "sog": df.sog.fillna(0).astype("float32"),
            "cog": df.cog.fillna(0).astype("float32"),
            "flag_code": df.flag_code.fillna("none").astype("str"),
            "vessel_class": df.vessel_class.fillna("none").astype("str"),
            "dist2coast": df.dist2coast.fillna(0).astype("float32"),
        }
    )


def read_csvs(csvs: os.PathLike) -> dd.DataFrame:
    """reads a list of csvs into a dask dataframe"""
    dtype = {"flag_code": "object", "name": "object", "vessel_class": "object"}
    gdf = dd.read_csv(csvs, usecols=DFCOLS, dtype=dtype, assume_missing=True)
    gdf = gdf.map_partitions(convert_partition, meta=EXAMPLE).persist()
    return gdf


def get_every_doy(year_to_process: int = YEAR_TO_PROCESS) -> list:
    """gets every day of the year in YYYY-MM-DD format

    Parameters
    ----------
    YEAR_TO_PROCESS : int, optional
         , by default 2022

    Returns
    -------
    list

    """
    start_date = date(
        YEAR_TO_PROCESS, 1, 8
    )  # we start on doy=8 so we have at least 7 days for each event
    end_date = date(YEAR_TO_PROCESS, 12, 31)
    delta = timedelta(days=1)
    dates = []
    while start_date <= end_date:
        dates.append(start_date.strftime("%Y-%m-%d"))
        start_date += delta
    return dates


def read_events_into_dataframe(EVENTS_PATH: str) -> dd:
    """reads events (e.g. from ES) into a dask dataframe

    Parameters
    ----------
    EVENTS_PATH : str

    Returns
    -------
    dd
        dask dataframe
    """
    return dd.read_csv(EVENTS_PATH, assume_missing=True)


def write_to_csv(df: pd.DataFrame, single_day_tsrs: pd.DataFrame) -> pd.DataFrame:
    """reads events (e.g. from ES) into a dask dataframe

    Parameters
    ----------
    df : pd.DataFrame
    single_day_tsrs : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    ves_code = df["category"].iloc[0]

    flag_code = df["flag_code"].iloc[0]
    output_dir = (
        Path(OUTPUT_DIR)
        / str(ves_code)
        / str(flag_code)
        / str((len(list(Path(OUTPUT_DIR).glob("*"))) // 1000))
        / Path(df.name)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    i = 0
    out_name = f"{df.name}_{i}.csv"
    output_path = os.path.join(output_dir, out_name)
    while os.path.exists(output_path):
        logger.debug(f"File {output_path} already exists, incrementing")
        i += 1
        out_name = f"{df.name}_{i}.csv"
        output_path = os.path.join(output_dir, out_name)

    df.sort_values(by="send", ascending=True, inplace=True)
    df.to_csv(output_path, sep=",", header=True, index=False)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df[["send"]] = df[["send"]].apply(pd.to_datetime, format=AVRO_TIME_FORMAT)
    df.reset_index(inplace=True)
    start_time = single_day_tsrs[
        single_day_tsrs[VESSEL_ID_TO_EXTRACT].isin([df.iloc[0][ES_TRACK_ID_COL]])
    ]["start.time"].values[0]
    end_time = single_day_tsrs[
        single_day_tsrs[VESSEL_ID_TO_EXTRACT].isin([df.iloc[0][ES_TRACK_ID_COL]])
    ]["end.time"].values[0]
    df["label"] = LABEL_STR_TO_INT["none"]  # must redo this with subpath level labels
    df.loc[
        (df["send"] >= start_time) & (df["send"] <= end_time), "label"
    ] = LABEL_STR_TO_INT["one_sided_rendezvous"]
    df.sort_values(by="send", ascending=True, inplace=True)
    df.to_csv(output_path, sep=",", header=True, index=False)

    return df


def gen_all_csvs(EVENTS_PATH: str) -> None:
    """generates CSV for each event file

    Parameters
    ----------
    EVENTS_PATH : str

    """
    ddf = read_events_into_dataframe(EVENTS_PATH)
    dates = get_every_doy(YEAR_TO_PROCESS)
    for day in dates:
        logger.debug(day)
        start = time.time()
        single_day_tsrs = ddf.loc[
            ddf["start.time"].str.contains(day, regex=False)
        ].compute()
        single_day_tsrs[["start.time", "end.time"]] = single_day_tsrs[
            ["start.time", "end.time"]
        ].apply(pd.to_datetime, format=ES_TIME_FORMAT)

        gdf = read_csvs(list(Path(CSV_FILES_INPUT).glob(f"*{day}*.csv"))[0])

        track_ids = single_day_tsrs[VESSEL_ID_TO_EXTRACT].values

        gdf.loc[gdf[ES_TRACK_ID_COL].isin(list(track_ids))].compute().groupby(
            ["trackId"]
        ).apply(write_to_csv, single_day_tsrs=single_day_tsrs)
        logger.debug(time.time() - start)


@click.command()
@click.argument("events_path", type=click.Path(exists=True, dir_okay=False))
def cli(events_path: str) -> None:
    """Generates CSV for each event file from given events path."""
    gen_all_csvs(events_path)


if __name__ == "__main__":
    cli()
