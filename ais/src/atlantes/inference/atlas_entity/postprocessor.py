"""Entity postprocessor class"""

from typing import Optional

from atlantes.atlas.atlas_utils import (
    AtlasEntityLabelsTrainingWithUnknown,
    get_atlas_entity_inference_config,
)
from atlantes.inference.atlas_entity.datamodels import (
    EntityPostprocessorInput,
    EntityPostprocessorOutput,
    EntityPostprocessorOutputDetails,
)
from atlantes.log_utils import get_logger
from atlantes.machine_annotation.buoy_vessel_annotate import is_buoy_based_on_name
from atlantes.utils import AIS_CATEGORIES, VESSEL_TYPES_BIN_DICT
from prometheus_client import CollectorRegistry, Counter

logger = get_logger("atlas_entity_postprocessor")


class KnownShipTypeAndBuoyName(ValueError):
    """Exception raised when a known ship type is classified as a buoy."""

    ERROR_MESSAGE = "The ship type is known and the name indicates it is a buoy,\
                this should not happen either the category field is wrong, the name is wrong \
                or someone is using a real mmsi/vessel_id for a buoy."

    def __init__(
        self,
        message: Optional[str] = None,
    ) -> None:
        self.message = message or self.ERROR_MESSAGE
        super().__init__(self.message)


NUM_REPLICAS = 1
NUM_CPUS = 1
NUM_GPUS = 0


class AtlasEntityPostProcessor:
    """Class for postprocessing AIS trajectory classifications"""

    def __init__(self) -> None:
        """Initialize the confidence threshold for entity classification"""
        self.ENTITY_CLASS_CONFIDENCE_THRESHOLD = 0.5
        self.ais_categories = AIS_CATEGORIES
        self.vessel_types_bin_dict = VESSEL_TYPES_BIN_DICT
        logger.info(f"Vessel types bin dict: {self.vessel_types_bin_dict}")
        self.data_config = get_atlas_entity_inference_config()["data"]
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize the metrics for the postprocessor

        Always add a counter for each postprocessing rule"""

        REGISTRY = CollectorRegistry(auto_describe=True)

        self.buoy_post_processed_rule_applied = Counter(
            "entity_post_processed_buoy_in_name",
            "Buoy in name",
            registry=REGISTRY,
        )
        self.known_binned_ship_type_post_processed_rule_applied = Counter(
            "entity_post_processed_known_binned_ship_type",
            "Known binned ship type",
            registry=REGISTRY,
        )
        self.confidence_threshold_rule_applied = Counter(
            "entity_post_processed_confidence_threshold",
            "Confidence threshold",
            registry=REGISTRY,
        )

        self.not_enough_messages_rule_applied = Counter(
            "entity_post_processed_not_enough_messages",
            "Not enough messages",
            registry=REGISTRY,
        )

    def is_binned_ship_type_known(self, binned_ship_type: int) -> bool:
        """Check if the ship type is known"""
        if binned_ship_type not in self.ais_categories["category"].values:
            raise ValueError(f"Unknown ship type {binned_ship_type=}")
        # TODO: SHould this be inclusive of known rather then exclusive of unknown
        unknown_category = int(
            self.ais_categories[self.ais_categories["category_desc"] == "Unknown"][
                "category"
            ].iloc[0]
        )
        return binned_ship_type != unknown_category

    def check_confidence_threshold(
        self, confidence: float, predicted_class: AtlasEntityLabelsTrainingWithUnknown
    ) -> tuple[bool, AtlasEntityLabelsTrainingWithUnknown]:
        """Check if the confidence is below the threshold and return the postprocessed class"""
        if confidence < self.ENTITY_CLASS_CONFIDENCE_THRESHOLD:
            return True, AtlasEntityLabelsTrainingWithUnknown.UNKNOWN
        return False, AtlasEntityLabelsTrainingWithUnknown(predicted_class.value)

    def postprocess(
        self,
        entity_postprocessor_input: EntityPostprocessorInput,
    ) -> EntityPostprocessorOutput:
        """Postprocess the AIS trajectory data for entity classification using the Atlantes system

        Parameters
        ----------
        entity_postprocessor_input : EntityPostprocessorInput
            The input to the entity postprocessor

        Returns
        -------
        EntityPostprocessorOutput
            The postprocessed entity classification and its details
        """
        predicted_class = entity_postprocessor_input.predicted_class
        entity_classification_details = (
            entity_postprocessor_input.entity_classification_details
        )
        metadata = entity_postprocessor_input.metadata
        # Set the default postprocessed class to the predicted class
        postprocessed_entity_class = predicted_class
        if metadata.binned_ship_type is None:
            binned_ship_type = int(self.vessel_types_bin_dict[metadata.ais_type])
            logger.info(f"Parsed binned ship type: {binned_ship_type}")
        else:
            logger.info(
                f"Using passed in binned ship type: {metadata.binned_ship_type}"
            )
            binned_ship_type = int(metadata.binned_ship_type)
        is_buoy_name = is_buoy_based_on_name(metadata.mmsi, metadata.entity_name)
        # TODO: Ask SME if Fishing reporting vessels can be buoys
        is_known_binned_ship_type = self.is_binned_ship_type_known(binned_ship_type)
        has_not_enough_messages = (
            metadata.track_length < self.data_config["MIN_AIS_MESSAGES"]
        )

        # Postprocessing Rules
        if is_buoy_name and is_known_binned_ship_type:
            logger.error(
                f"known ship type and buoyish name {binned_ship_type=}, {metadata.entity_name=}"
            )
            raise KnownShipTypeAndBuoyName()
        elif was_post_processed := is_buoy_name:
            logger.info(f"Buoyish name {metadata.entity_name=}")
            self.buoy_post_processed_rule_applied.inc()
            postprocessed_entity_class = AtlasEntityLabelsTrainingWithUnknown.BUOY
        elif was_post_processed := is_known_binned_ship_type:
            self.known_binned_ship_type_post_processed_rule_applied.inc()
            logger.info(f"Known ship type {binned_ship_type=}")
            postprocessed_entity_class = AtlasEntityLabelsTrainingWithUnknown.VESSEL
        elif was_post_processed := has_not_enough_messages:
            logger.info(
                f"Not enough messages {metadata.track_length=} \
                    expected at least {self.data_config['MIN_AIS_MESSAGES']} \
                        messages"
            )
            self.not_enough_messages_rule_applied.inc()
            postprocessed_entity_class = AtlasEntityLabelsTrainingWithUnknown.UNKNOWN
        else:
            # Confidence thresholding
            is_below_confidence_threshold, postprocessed_entity_class = (
                self.check_confidence_threshold(
                    entity_classification_details.confidence, predicted_class
                )
            )
            if was_post_processed := is_below_confidence_threshold:
                postprocessed_entity_class = (
                    AtlasEntityLabelsTrainingWithUnknown.UNKNOWN
                )
                self.confidence_threshold_rule_applied.inc()
            else:
                postprocessed_entity_class = AtlasEntityLabelsTrainingWithUnknown(
                    predicted_class.value
                )

        output_details = EntityPostprocessorOutputDetails(
            predicted_classification=predicted_class.name.lower(),
            model=entity_classification_details.model,
            confidence=entity_classification_details.confidence,
            outputs=entity_classification_details.outputs,
            postprocessed_classification=postprocessed_entity_class.name.lower(),
            postprocess_rule_applied=was_post_processed,
            confidence_threshold=self.ENTITY_CLASS_CONFIDENCE_THRESHOLD,
        )
        return EntityPostprocessorOutput(
            entity_class=postprocessed_entity_class.name.lower(),
            entity_classification_details=output_details,
        )
