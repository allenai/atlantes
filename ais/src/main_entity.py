"""ATLAS inference service Entity
"""

from __future__ import annotations

import os

from atlantes.inference.atlas_entity.model import AtlasEntityModelDeployment
from atlantes.inference.atlas_entity.pipeline import \
    AtlasEntityClassifierDeployment
from atlantes.inference.atlas_entity.postprocessor import \
    AtlasEntityPostProcessorDeployment
from atlantes.inference.atlas_entity.preprocessor import \
    AtlasEntityPreprocessorDeployment
from ray import serve
from ray.serve.config import HTTPOptions

atlas_entity_pipeline = AtlasEntityClassifierDeployment.bind(  # type: ignore
    AtlasEntityPreprocessorDeployment.bind(),  # type: ignore
    AtlasEntityModelDeployment.bind(),  # type: ignore
    AtlasEntityPostProcessorDeployment.bind(),  # type: ignore
)


if __name__ == "__main__":
    HOST = "0.0.0.0"  # nosec B104
    PORT = os.getenv("ATLAS_ENTITY_PORT", default=8001)
    AEC_HTTP_OPTIONS = HTTPOptions(host=HOST, port=PORT)
    serve.start(http_options=AEC_HTTP_OPTIONS)
    serve.run(atlas_entity_pipeline, blocking=True)
