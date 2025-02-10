"""Deployment Pipeline for the Atlas Activity Model

# TODO: Make all steps have simple pydantic response objects
"""

from typing import Any, Awaitable

import pandas as pd
from atlantes.atlas.atlas_utils import AtlasActivityLabelsTraining
from atlantes.atlas.schemas import TrackfileDataModelTrain
from atlantes.inference.atlas_activity.preprocessor import PreprocessedActivityData
from atlantes.inference.common import AtlasInferenceError, ATLASRequest, ATLASResponse
from atlantes.log_utils import get_logger
from fastapi import FastAPI
from pandera.errors import SchemaError
from pandera.typing import DataFrame

app = FastAPI()

logger = get_logger("activity_classifier")


@app.get("/")
async def home() -> dict:
    return {"message": "ATLAS Activity Classifier"}


class AtlasActivityClassifier:
    """Class for classifying the activity of a trajectory using the Atlantes system for AIS behavior classification

    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to classify the activity of the trajectory.
    The activity will be one of the predefined activity classes."""

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
    ) -> PreprocessedActivityData:
        return self.preprocessor.preprocess(track_data)

    def _apply_model(
        self, preprocessed_data_stream: PreprocessedActivityData
    ) -> tuple[
        AtlasActivityLabelsTraining, dict, dict
    ]:
        """Run inference on the preprocessed data"""
        if not isinstance(preprocessed_data_stream, PreprocessedActivityData):
            raise ValueError(
                f"The preprocessed_data_stream must be a PreprocessedActivityData, not a {type(preprocessed_data_stream)} \
                    as batching is handled by Ray Serve and not supported here"
            )
        return self.model.run_inference([preprocessed_data_stream])[0]

    def _apply_postprocessing(
        self,
        activity_class_details_metadata_tuples: tuple[
            AtlasActivityLabelsTraining, dict, dict
        ],
    ) -> tuple[str, dict]:
        return self.postprocessor.postprocess(activity_class_details_metadata_tuples)

    def run_pipeline(
        self, track_data: DataFrame[TrackfileDataModelTrain]
    ) -> tuple[str, dict] | Awaitable[tuple[str, dict]]:
        """
        ATLAS activity requires data in TrackfileDataModelTrain format

        Pandera Validation occurs during preprocessing

        Parameters
        ----------
        track_data : DataFrame[TrackfileDataModelTrain]
            see atlantes.atlas.schemas.TrackfileDataModelTrain for the required columns
        Returns
        -------
        tuple[str, dict]
            Returns a tuple of the predicted activity class and a dict of inference details e.g confidence, outputs

        """
        try:
            # Preprocess steps

            preprocessed_data_stream = self._apply_preprocessing(track_data)

            activity_class_details_metadata_tuples = self._apply_model(
                preprocessed_data_stream
            )

            activity_class_details_tuples = self._apply_postprocessing(
                activity_class_details_metadata_tuples
            )

            return activity_class_details_tuples
        except SchemaError as e:
            raise AtlasInferenceError(f"Error while running inference: {e}") from e


class AtlasActivityClassifierDeployment(AtlasActivityClassifier):
    """Class for classifying the activity of a trajectory using the Atlantes system for AIS behavior classification

    The trajectory is passed through a pipeline of preprocessor, model, and postprocessor to classify the activity of the trajectory.
    The activity will be one of the predefined activity classes."""

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
        activity_class_details_metadata_tuples: DeploymentResponse,
    ) -> DeploymentResponse:
        return self.postprocessor.postprocess.remote(
            activity_class_details_metadata_tuples
        )

    @app.post("/classify", response_model=ATLASResponse)
    async def classify_track_activity(self, request: ATLASRequest) -> ATLASResponse:
        predictions = await self.run_pipeline(pd.DataFrame(request.track))  # type: ignore
        return ATLASResponse(predictions=predictions)
