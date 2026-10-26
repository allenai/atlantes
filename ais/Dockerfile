FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Build arguments for credentials and Git commit hash
ARG MOUNTED_CREDENTIALS_PATH
ARG GIT_COMMIT_HASH

# Environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=${MOUNTED_CREDENTIALS_PATH} \
    DEBIAN_FRONTEND=noninteractive \
    GIT_COMMIT_HASH=${GIT_COMMIT_HASH} \
    UV_SYSTEM_PYTHON=1

# Install extra OS packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      wget \
      netcat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /ais

# Install PyTorch ecosystem with specific versions
RUN uv pip install --no-cache \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies with uv
COPY ./requirements/requirements.txt /ais/requirements/requirements.txt
RUN uv pip install --no-cache -r requirements/requirements.txt

# Copy the rest of your application code
COPY ./ /ais

# Download the external marine infrastructure file and install package
RUN mkdir -p /ais/src/atlantes/data && \
    wget -O /ais/src/atlantes/data/latest_marine_infrastructure.geojson \
        https://storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson \
        --quiet && \
    uv pip install --no-cache .

# Set the default command to run your application
CMD ["python", "src/main_activity.py"]
