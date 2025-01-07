"""Deployment Pipeline for the Atlas Entity Model
"""

from typing import Any, Awaitable

import pandas as pd
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_entity.datamodels import (
    EntityPostprocessorInput, EntityPostprocessorOutput,
    PreprocessedEntityData)
from atlantes.inference.atlas_entity.postprocessor import \
    KnownShipTypeAndBuoyName
from atlantes.inference.common import (AtlasInferenceError, ATLASRequest,
                                       ATLASResponse)
from atlantes.log_utils import get_logger
from fastapi import FastAPI
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

entity_app = FastAPI()

logger = get_logger("ray.serve")


@entity_app.get("/")
async def entity_home() -> dict:
    return {"message": "ATLAS Entity Ray App"}


# This configuration only comes into play locally and during CI otherwise it is overridden by Kuberay and ray config

NUM_CPUS = 1
NUM_REPLICAS = 1


class AtlasEntityClassifier:
    """Class for identifying the entity of a trajectory using the Atlantes system for AIS behavior classification

    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to identify the entity of the trajectory.
    tHe entity will be either a vessel or equipment/buoy.

    Note: This pipeline does not support batching
    """

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        postprocessor: Any,
    ) -> None:
        """Load the preprocessor, model, and postprocessor for the activity classifier DeploymentHandle"""
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def _apply_preprocessing(
        self, track_data: DataFrame[TrackfileDataModelTrain]
    ) -> PreprocessedEntityData:
        return self.preprocessor.preprocess(track_data)

    def _apply_model(
        self, preprocessed_data_stream: PreprocessedEntityData
    ) -> EntityPostprocessorInput:
        """Run inference on the preprocessed data using the mode

        Although the model supports batching, this pipeline does not.
        This is because Ray serve handles batching inputs from the
        preprocessor and unbatching
        to the postprocessor.
        """
        if isinstance(preprocessed_data_stream, list):
            raise ValueError(
                "The preprocessed_data_stream must not be a list, as batching is not supported"
            )
        return self.model.run_inference([preprocessed_data_stream])[0]

    def _apply_postprocessing(
        self,
        entity_outputs_with_details_metadata_tuples: EntityPostprocessorInput,
    ) -> EntityPostprocessorOutput:
        """Postprocess the AIS trajectory data for entity classification
        using the Atlantes system"""
        return self.postprocessor.postprocess(
            entity_outputs_with_details_metadata_tuples
        )

    def run_pipeline(
        self, track_data: DataFrame[TrackfileDataModelTrain]
    ) -> EntityPostprocessorOutput | Awaitable[EntityPostprocessorOutput]:
        """
        ATLAS entity requires data in TrackfileDataModelTrain format

        Pandera Validation occurs during preprocessing

        Parameters
        ----------
        track_data : DataFrame[TrackfileDataModelTrain]
            see atlantes.atlas.schemas.TrackfileDataModelTrain for the required columns
        Returns
        -------
        tuple[str, dict]
            Returns a tuple of the predicted entity class and a dict of inference details e.g confidence, outputs

        """
        try:
            preprocessed_data_stream = self._apply_preprocessing(track_data)
            entity_outputs_with_details_metadata_tuples = self._apply_model(
                preprocessed_data_stream
            )
            class_details_tuples = self._apply_postprocessing(
                entity_outputs_with_details_metadata_tuples
            )
            # Return a list of the enum item name (lowered) and a dict of inference details.
            return class_details_tuples
        except SchemaError as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
        except KnownShipTypeAndBuoyName as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e
        except Exception as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": NUM_CPUS,
    },
)
@serve.ingress(entity_app)
class AtlasEntityClassifierDeployment(AtlasEntityClassifier):
    """Ray Serve FastAPI app for entity classification"""

    def __init__(
        self,
        preprocessor: DeploymentHandle,
        model: DeploymentHandle,
        postprocessor: DeploymentHandle,
    ) -> None:
        """Load the preprocessor, model, and postprocessor for the activity classifier DeploymentHandle"""
        if not isinstance(preprocessor, DeploymentHandle):
            raise ValueError(
                f"The preprocessor must be a DeploymentHandle, not a {type(preprocessor)} "
            )
        if not isinstance(model, DeploymentHandle):
            raise ValueError(
                f"The model must be a DeploymentHandle, not a {type(model)} "
            )
        if not isinstance(postprocessor, DeploymentHandle):
            raise ValueError(
                f"The postprocessor must be a DeploymentHandle, not a {type(postprocessor)} "
            )

        super().__init__(preprocessor, model, postprocessor)

    def _apply_preprocessing(
        self, track_data: DeploymentResponse
    ) -> DeploymentResponse:
        return self.preprocessor.preprocess.remote(track_data)

    def _apply_model(
        self, preprocessed_data_dict_stream: DeploymentResponse
    ) -> DeploymentResponse:
        return self.model.batch_handler_run_inference.remote(
            preprocessed_data_dict_stream
        )

    def _apply_postprocessing(
        self,
        entity_outputs_with_details_metadata_tuples: DeploymentResponse,
    ) -> DeploymentResponse:
        return self.postprocessor.postprocess.remote(
            entity_outputs_with_details_metadata_tuples
        )

    def convert_to_serializable_output(
        self,
        entity_postprocessor_output: EntityPostprocessorOutput,
    ) -> tuple[str, dict]:
        postprocessed_classification_details = (
            entity_postprocessor_output.entity_classification_details
        )
        entity_classification_details = {
            "predicted_classification": postprocessed_classification_details.predicted_classification,
            "model": postprocessed_classification_details.model,
            "confidence": postprocessed_classification_details.confidence,
            "outputs": postprocessed_classification_details.outputs,
            "postprocessed_classification": postprocessed_classification_details.postprocessed_classification,
            "postprocess_rule_applied": postprocessed_classification_details.postprocess_rule_applied,
            "confidence_threshold": postprocessed_classification_details.confidence_threshold,
        }
        return (
            entity_postprocessor_output.entity_class,
            entity_classification_details,
        )

    @entity_app.post("/classify", response_model=ATLASResponse)
    async def classify_entity(self, request: ATLASRequest) -> Any:
        predictions = await self.run_pipeline(pd.DataFrame(request.track))  # type: ignore
        return ATLASResponse(
            predictions=self.convert_to_serializable_output(predictions)
        )
