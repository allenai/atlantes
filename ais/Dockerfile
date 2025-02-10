FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime@sha256:3387e598cb94fc248d82e712a65b10931a990cea3a2e76362ca30d135f565de4

ARG MOUNTED_CREDENTIALS_PATH
ENV GOOGLE_APPLICATION_CREDENTIALS=${MOUNTED_CREDENTIALS_PATH} \
    DEBIAN_FRONTEND=noninteractive

# this hash is used to tag the image
ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget netcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Should add the set num threads env variable here
WORKDIR /ais
# THIS means it is easier to rebuild images for beaker then it would be because the layer with the reqs are cached
COPY ./requirements/requirements.txt /ais/requirements/requirements.txt
RUN  pip install --no-cache-dir --upgrade pip==24.0.0 && \
    pip3 install --no-cache-dir -r requirements/requirements.txt
COPY ./ /ais

# This is the latest marine infrastructure file containing coordinates for infrastcuture detected by satlas -- see satlas.allen.ai
RUN mkdir -p /ais/src/atlantes/data && wget -O /ais/src/atlantes/data/latest_marine_infrastructure.geojson https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.geojson --quiet

RUN pip3 install --no-cache-dir .

ENV PORT=8080
EXPOSE ${PORT}

# Note that the line below may need to be specific to service,
# if we want a single dockerfile/or can have multiple docker-compose files.
# Either way, adding this so that tests/main.py can be run in the with gh-actions
CMD ["python3", "src/main_activity.py"]
