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

# Install extra OS packages, add deadsnakes PPA, install Python 3.12, and set up Python environment
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      wget \
      netcat \
      software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
      python3.12 \
      python3.12-venv \
      python3.12-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Set python3.12 as the default "python"
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    # Install PyTorch ecosystem with specific versions
    uv pip install --no-cache \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /ais

# Copy requirements and install Python dependencies using Python 3.12's pip
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
