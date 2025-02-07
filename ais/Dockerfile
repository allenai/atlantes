FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Build arguments for credentials and Git commit hash
ARG MOUNTED_CREDENTIALS_PATH
ARG GIT_COMMIT_HASH

# Environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=${MOUNTED_CREDENTIALS_PATH} \
    DEBIAN_FRONTEND=noninteractive \
    GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

# Install extra OS packages, add deadsnakes PPA, install Python 3.12, and set up Python environment
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      wget \
      netcat \
      software-properties-common && \
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
    # Install pip for Python 3.12
    wget -q -O - https://bootstrap.pypa.io/get-pip.py | python && \
    # Upgrade pip to a specific version
    python -m pip install --no-cache-dir --upgrade pip==24.0.0 && \
    # Install PyTorch ecosystem with specific versions
    python -m pip install --no-cache-dir \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /ais

# Copy requirements and install Python dependencies using Python 3.12's pip
COPY ./requirements/requirements.txt /ais/requirements/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements/requirements.txt

# Copy the rest of your application code
COPY ./ /ais

# Download the external marine infrastructure file and install package
RUN mkdir -p /ais/src/atlantes/data && \
    wget -O /ais/src/atlantes/data/latest_marine_infrastructure.geojson \
        https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.geojson \
        --quiet && \
    python -m pip install --no-cache-dir .

# Set the default command to run your application
CMD ["python", "src/main.py"]
