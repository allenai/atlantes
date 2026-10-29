"""Create entity vessel type labels for the Atlantes dataset for pretraining a model for Atlas Activity Recognition.

VERY ROUGH SCRIPT TO Create an entity dataset for the vessel type labels for the Atlantes dataset for pretraining a model for Atlas Activity Recognition."""

import os

import click
from atlantes.atlas.atlas_utils import AtlasEntityVesselTypeLabelClass
from atlantes.datautils import (ALL_META_DATA_INDEX_PATHS, DT_STRING,
                                GCP_BUCKET_NAME_AIS_TRACKS)
from atlantes.log_utils import get_logger
from atlantes.utils import (AIS_CATEGORIES, export_dataset_to_gcp,
                            load_all_metadata_indexes)

logger = get_logger(__name__)

VESSEL_TYPES_BIN_DICT = AIS_CATEGORIES.set_index("num", drop=True)["category"].to_dict()
LABEL_TO_NAME = (
    AIS_CATEGORIES.set_index("category", drop=True)["category_desc"]
    .apply(lambda x: x.lower())
    .to_dict()
)


def get_top_n_class_names(num_classes: int) -> list[str]:
    return [
        list(AtlasEntityVesselTypeLabelClass)[i].name.lower()
        for i in range(num_classes)
    ]


@click.command()
@click.option(
    "--num-classes",
    default=5,
    type=int,
    help="The number of classes to use for the vessel type labels",
)
def create_entity_vessel_type_labels(num_classes: int) -> None:
    df = load_all_metadata_indexes(ALL_META_DATA_INDEX_PATHS)
    df.ais_type = df.ais_type.astype(int)
    # Create a binned ais category index
    df.loc[:, "binned_ais_category"] = df.ais_type.map(VESSEL_TYPES_BIN_DICT)
    logger.info(df.columns)
    # For the given enum keep all those whose name mathces a dif
    df.loc[:, "entity_class_label"] = df.binned_ais_category.map(LABEL_TO_NAME).values
    # Filter out the ones that are not in the enum
    filter_to_vessel_type_class_df = df[
        df.entity_class_label.isin(get_top_n_class_names(num_classes))
    ]

    logger.info(filter_to_vessel_type_class_df["entity_class_label"].value_counts())
    logger.info(filter_to_vessel_type_class_df.columns)
    vessel_type_entity_label_df = filter_to_vessel_type_class_df[
        ["Path", "entity_class_label"]
    ]

    shuffled_df = vessel_type_entity_label_df.sample(frac=1)
    output_dir = f"labels/entity_vessel_type_labels_{DT_STRING}"
    os.makedirs(output_dir, exist_ok=True)
    export_dataset_to_gcp(
        output_dir,
        shuffled_df,
        f"entity_vessel_type_labels_{DT_STRING}",
        GCP_BUCKET_NAME_AIS_TRACKS,
        plot_png=False,
    )


if __name__ == "__main__":
    create_entity_vessel_type_labels()
