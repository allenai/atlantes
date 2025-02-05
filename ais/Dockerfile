FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime-py311

ARG MOUNTED_CREDENTIALS_PATH
ENV GOOGLE_APPLICATION_CREDENTIALS=${MOUNTED_CREDENTIALS_PATH} \
    DEBIAN_FRONTEND=noninteractive

# This hash is used to tag the image
ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget netcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /ais

# Copy requirements and install dependencies
COPY ./requirements/requirements.txt /ais/requirements/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==24.0.0 && \
    pip install --no-cache-dir -r requirements/requirements.txt

# Copy application code
COPY ./ /ais

# Download external marine infrastructure file
RUN mkdir -p /ais/src/atlantes/data && \
    wget -O /ais/src/atlantes/data/latest_marine_infrastructure.geojson \
         https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.geojson \
         --quiet

# Install the package
RUN pip install --no-cache-dir .

# Run the application
CMD ["python", "src/main.py"]
