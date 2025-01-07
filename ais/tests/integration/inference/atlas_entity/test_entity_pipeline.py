"""Integration tests for the entity pipeline."""

from typing import Generator

import numpy as np
import pandas as pd
import pytest
import ray
from atlantes.atlas.atlas_utils import (ATLAS_COLUMNS_WITH_META,
                                        get_atlas_entity_inference_config)
from atlantes.inference.atlas_entity.datamodels import \
    EntityPostprocessorOutput
from atlantes.inference.atlas_entity.model import (AtlasEntityModel,
                                                   AtlasEntityModelDeployment)
from atlantes.inference.atlas_entity.pipeline import (
    AtlasEntityClassifier, AtlasEntityClassifierDeployment)
from atlantes.inference.atlas_entity.postprocessor import (
    AtlasEntityPostProcessor, AtlasEntityPostProcessorDeployment,
    KnownShipTypeAndBuoyName)
from atlantes.inference.atlas_entity.preprocessor import (
    AtlasEntityPreprocessor, AtlasEntityPreprocessorDeployment)
from atlantes.inference.common import ATLASRequest
from atlantes.log_utils import get_logger
from pandera.errors import SchemaError
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve import Application
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)


@pytest.fixture(scope="class")
def in_memory_ais_track_df(test_ais_df1: pd.DataFrame) -> pd.DataFrame:
    """Build an in-memory AIS track stream with CPD subpaths."""
    test_ais_df1_inference = test_ais_df1[ATLAS_COLUMNS_WITH_META].head(4000).copy()
    return test_ais_df1_inference


@pytest.fixture(scope="class")
def buoy_df(test_buoy_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Build an in-memory AIS track stream."""
    test_buoy_df_inference = test_buoy_df[ATLAS_COLUMNS_WITH_META].copy()
    # drop current subpath_num column
    return test_buoy_df_inference


@pytest.fixture(scope="class")
def entity_classifier_pipeline_deployment() -> Generator[DeploymentHandle, None, None]:
    """Return the entity classifier pipeline."""
    entity_app: Application = AtlasEntityClassifierDeployment.bind(  # type: ignore
        AtlasEntityPreprocessorDeployment.bind(),  # type: ignore
        AtlasEntityModelDeployment.bind(),  # type: ignore
        AtlasEntityPostProcessorDeployment.bind(),  # type: ignore
    )
    entity_handle: DeploymentHandle = serve.run(entity_app)
    yield entity_handle
    ray.shutdown()


@pytest.fixture(scope="class")
def entity_classifier_pipeline() -> AtlasEntityClassifier:
    """Return the entity classifier pipeline."""
    return AtlasEntityClassifier(
        AtlasEntityPreprocessor(),
        AtlasEntityModel(),
        AtlasEntityPostProcessor(),
    )


class TestEntityClassifier:
    """Tests for the pipelines."""

    def test_entity_pipeline(
        self,
        in_memory_ais_track_df: pd.DataFrame,
        entity_classifier_pipeline: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        output = entity_classifier_pipeline.run_pipeline(in_memory_ais_track_df)
        assert isinstance(output, EntityPostprocessorOutput)


class TestEntityClassifierDeployment:
    """Tests for the Ray Deployed pipelines."""

    def test_pipeline_deployment_component_names_unchanged(self) -> None:
        """Test that the deployment component names are not changed."""
        assert (
            AtlasEntityPreprocessorDeployment.name  # type: ignore
            == "AtlasEntityPreprocessorDeployment"
        ), "The deployment name should not be changed, if it Must be change it also must be changed in serve config"
        assert (
            AtlasEntityModelDeployment.name == "AtlasEntityModelDeployment"  # type: ignore
        ), "The deployment name should not be changed, if it Must be change it also must be changed in serve config"
        assert (
            AtlasEntityPostProcessorDeployment.name  # type: ignore
            == "AtlasEntityPostProcessorDeployment"
        ), "The deployment name should not be changed, if it Must be change it also must be changed in serve config"

    def test_deployment_name_unchanged(
        self, entity_classifier_pipeline_deployment: DeploymentHandle
    ) -> None:
        """Test that the deployment name is not changed."""
        assert (
            entity_classifier_pipeline_deployment.deployment_name
            == "AtlasEntityClassifierDeployment"
        ), "The deployment name should not be changed, if it Must be change it also must be changed in serve config"

    def test_entity_pipeline_vessel(
        self,
        in_memory_ais_track_df: pd.DataFrame,
        entity_classifier_pipeline_deployment: DeploymentHandle,
    ) -> None:
        """Test the entity pipeline."""
        request = ATLASRequest(track=in_memory_ais_track_df.to_dict(orient="records"))
        response = entity_classifier_pipeline_deployment.classify_entity.remote(  # type: ignore
            request
        ).result()
        predicted_class_name, predicted_class_details = response.predictions
        inference_config = get_atlas_entity_inference_config()
        model_id = inference_config["model"]["ATLAS_ENTITY_MODEL_ID"]
        assert isinstance(predicted_class_name, str)
        assert isinstance(predicted_class_details, dict)
        assert predicted_class_name == "vessel"  # Buoy or vessel
        assert predicted_class_details["model"] == model_id
        assert (
            predicted_class_details["confidence"] > 0.9
            and predicted_class_details["confidence"] <= 1.0
        )
        assert isinstance(predicted_class_details["outputs"], list)
        assert len(predicted_class_details["outputs"]) == 2
        assert predicted_class_details["postprocessed_classification"] == "vessel"
        assert predicted_class_details["postprocess_rule_applied"] is False

    def test_entity_pipeline_buoy(
        self,
        buoy_df: pd.DataFrame,
        entity_classifier_pipeline_deployment: DeploymentHandle,
    ) -> None:
        """Test the entity pipeline."""
        request = ATLASRequest(track=buoy_df.to_dict(orient="records"))
        response = entity_classifier_pipeline_deployment.classify_entity.remote(  # type: ignore
            request
        ).result()
        inference_config = get_atlas_entity_inference_config()
        model_id = inference_config["model"]["ATLAS_ENTITY_MODEL_ID"]
        predicted_class_name, predicted_class_details = response.predictions
        logger.info(predicted_class_name)
        logger.info(predicted_class_details)
        logger.info(np.array(predicted_class_details["outputs"]).shape)
        assert isinstance(predicted_class_name, str)
        assert isinstance(predicted_class_details, dict)
        assert predicted_class_name == "buoy"
        assert predicted_class_details["model"] == model_id
        assert (
            predicted_class_details["confidence"] > 0.7
            and predicted_class_details["confidence"] <= 1.0
        )
        assert isinstance(predicted_class_details["outputs"], list)
        assert predicted_class_details["postprocessed_classification"] == "buoy"
        assert predicted_class_details["postprocess_rule_applied"] is True

    def test_entity_postprocessor_error_handling(
        self,
        buoy_df: pd.DataFrame,
        entity_classifier_pipeline_deployment: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        modified_buoy_df = buoy_df.copy()
        # pretend that something with a buoy name is listed as a fishing vessel
        modified_buoy_df.loc[:, "category"] = [30] * len(modified_buoy_df)
        try:
            entity_classifier_pipeline_deployment.classify_entity.remote(  # type: ignore
                ATLASRequest(track=modified_buoy_df.to_dict(orient="records"))
            ).result()
        except RayTaskError as e:
            assert isinstance(e.cause, KnownShipTypeAndBuoyName)

    def test_entity_pipeline_schema_error(
        self,
        test_ais_df1: pd.DataFrame,
        entity_classifier_pipeline_deployment: AtlasEntityClassifier,
    ) -> None:
        """Test the entity pipeline."""
        try:
            entity_classifier_pipeline_deployment.classify_entity.remote(  # type: ignore
                ATLASRequest(track=test_ais_df1.head(500).to_dict(orient="records"))
            ).result()
        except RayTaskError as e:
            assert isinstance(e.cause, SchemaError)

        # remove lat column
        test_ais_df1_inference = test_ais_df1.head(500).copy()
        test_ais_df1_inference.drop(columns=["lat"], inplace=True)
        try:
            entity_classifier_pipeline_deployment.classify_entity.remote(  # type: ignore
                ATLASRequest(track=test_ais_df1_inference.to_dict(orient="records"))
            ).result()
        except RayTaskError as e:
            assert isinstance(e.cause, SchemaError)
