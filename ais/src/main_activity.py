"""ATLAS inference service Only locally ran
"""

from __future__ import annotations

import logging.config
import os

from atlantes.inference.atlas_activity.model import \
    AtlasActivityModelDeployment
from atlantes.inference.atlas_activity.pipeline import \
    AtlasActivityClassifierDeployment
from atlantes.inference.atlas_activity.postprocessor import \
    AtlasActivityPostProcessorDeployment
from atlantes.inference.atlas_activity.preprocessor import \
    AtlasActivityPreprocessorDeployment
from ray import serve
from ray.serve.config import HTTPOptions

logger = logging.getLogger(__name__)


atlas_activity_pipeline = AtlasActivityClassifierDeployment.bind(  # type: ignore
    AtlasActivityPreprocessorDeployment.bind(),  # type: ignore
    AtlasActivityModelDeployment.bind(),  # type: ignore
    AtlasActivityPostProcessorDeployment.bind(),  # type: ignore
)


if __name__ == "__main__":
    # THis only runs locally
    HOST = "0.0.0.0"  # nosec B104
    PORT = os.getenv("ATLAS_ACTIVITY_PORT", default=8000)
    AAC_HTTP_OPTIONS = HTTPOptions(host=HOST, port=PORT)
    serve.start(http_options=AAC_HTTP_OPTIONS)
    serve.run(atlas_activity_pipeline, blocking=True)
