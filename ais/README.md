# Atlantes: A System for AIS Modeling

---

*Note: Atlantes refers to a collection of ATLAS: AIS Transformer Learning with Active Subpaths models (Name Subject to Change)*

The Atlantes package is a Python library for building and deploying machine learning-based systems for a variety of AIS based modeling tasks as part of Ai2's [Skylight Project](https://www.skylight.global/)


The **Automatic Identification System (AIS)** is a tracking system used on ships and by vessel traffic services (VTS) to identify and locate vessels. AIS data is transmitted via VHF radio frequencies, allowing for real-time tracking of vessel movements. Each ship equipped with AIS automatically sends information such as its unique identification, position, course, and speed, which can be received by other vessels, shore stations, and satellites. For more information, visit the website of our data provider [Spire](https://documentation.spire.com/ais-fundamentals/different-classes-of-ais/ais-channel-access-methods/).



## Overview of Atlantes AIS Modeling Tasks


1. Atlas Activity Classification: Given a trajectory, classify the activity of the trajectory.
   - End of Sequence Activity Classification: Given a trajectory, classify the activity of the trajectory at the end of the trajectory. (e.g. fishing, anchored, transiting, etc.) (Seq2class) (In Integration)
   - Dense Activity Classification: Given a trajectory, classify the activity of each subpath in the trajectory. (e.g. fishing, anchored, transiting, etc.) (seq2seq)


2. Atlas Entity Classification: Given a trajectory, classify the entity of the trajectory.
   - Vessel vs. Buoy Classification: Given an AIS trajectory, classify whether the entity transmitting the AIS messages is a vessel or buoy. (In Integration)
   - Vessel Type Classification: Given a trajectory, classify the vessel type of the trajectory. (e.g. fishing, cargo, tanker, etc.)

## Core Libraries

- **Torch**: Modeling Library
- **Dask**: Parallel and Distributed dataprocessing
- **Pandas**: Data manipulation, visualization, and analysis.
- **Pandera**: Provides a schema-based approach to validate Dataframes
- **Ray[Serve]**: Provides a scalable and fault-tolerant framework for deploying and serving machine learning models.
- **Wandb**: Experiment tracking
- **Beaker**: Ai2 experimentation Platform


## Getting Started

   ### Set up External Accounts

   Please make sure you have access to the following

   Note that training models requires
   1. authentication to GCP (for reading AIS trajectories and metadata) and
   2. authentication to weights and biases (for logging experimental runs and data)
   These can either be set programatically via the included secrets in this repo, via the
   beaker configuration files (if training on AI2 hardware), or interactively (`gcloud auth application-default login` and `wandb login`) depending on your environment. Within CI, authentication occurs via stored secrets and the .github/workflows/
   <summary><b> Set up GCP credentials </b> </summary>

   You will need access to the proto, integration, and prod gcp projects (to varying degrees).
   Talk to skylight engineers to get access.

   In your terminal, run

   `gcloud auth application-default login`


   <summary><b>Set up Weights & Biases (wandb)</b> </summary>

   In your terminal, run

   `wandb login`

   <summary><b> Set up Elastic Search </b> </summary>
   Note that this only applies to developers at Ai2 with access to Skylight's elastic search.
   This is not needed for deployment, it is only used for some specific dataset generation tasks.
   `export SEARCH_USERNAME=<username>` \
   `export SEARCH_PASSWORD=<password>`

   ### Set up Development Environment

   Refer to the repository readme to set up your formatting and precommit hooks.

   1. Set up pyenv:  \
   `python3 -m venv .venv-atlantes` \
   `source .venv-atlantes/bin/activate`


   2. For local development, use the following command to install the package: \
   `pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .`

   3. Install the development dependencies: \
   `pip install -r requirements/dev.txt`

   ### Visualize Trajectories

   (instructions on how to visualize a trajectory)


   ### Run Inference Service Locally

   1. Run the following command to start the service: \
   `python3 src/main.py`

   2. Run the following command to hit the endpoint: \
   `python3 examples/atlas_activity_request.py`


   ### Run Tests

   1. Run the following command to run all tests: \
   `pytest tests --ignore=tests/test_main.py -vv`

   ### Local testing of Services wth docker

   1. Save a path to gcp credentials in `GOOGLE_APPLICATION_CREDENTIALS`
   2. Navigate to eai/ais. Run `docker-compose up`
   3. Example request: `python examples/atlas_activity_request.py`


## Dataset Creation

See the [dataset creation documentation](docs/dataset_creation.md) for more details.


## Training

See the [training documentation](docs/training.md) for more details.


## Deployment

See the [deployment documentation](docs/deployment.md) for more details.
