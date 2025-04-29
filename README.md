

# Atlantes


This repository stores the ATLAS model alongside training, inference, and deployment libraries for real-time global-scale GPS trajectory modeling. The models were developed at Ai2 and are currently deployed in the [skylight](https://www.skylight.global/) maritime intelligence platform. Read more about the model on [arXiv](https://arxiv.org/abs/2504.19036v1). 

## Repository Structure

The repository is organized into the following top-level directories and subdirectories:


- `ais`: Contains scripts and modules for loading, cleaning, and preprocessing AIS data.
    - `src/atlantes`: Holds the core logic for the Atlantes system, including data processing, model training, and deployment.
    - `requirements.txt`: Specifies the required Python packages for running the Atlantes system as well as the dev requirements for development.
    - `data`: Stores various ais related data files used throughout the project in a non-production environment.
    - `tests`: Contains unit tests and integration tests for the Atlantes system.
    - `test-data`: Contains test data for the Atlantes system.


## Getting Started
For all projects, please ensure that you follow the initial steps to setup the environment and then refer to the relevant README file in the specific project's directory.
### Precommit hooks (.pre-commit-config.yaml)

Run:

1.  `$ pip install pre-commit`
2.  `$ pre-commit install`

Note that the code in ais repo will be required to pass these pre commits in order to be merged. In particular that means, besides the typical linting requirement, there are additional requirements on 100% static type annotations (which is enforced via mypy) and at least 90% documentation coverage for including both modules and functions (enforced via interrogate). Note that either setting can be bypassed for a particular directory or file by adding that file/directory to the excluded list of the corresponding module in the .pre-commit-config.yaml file. In other words these checks are opt-out rather than opt-in.

Pre commit hooks can be executed on `git commit` after following the above two steps.


### VSCode Setup

In (add VSCode Settings), we have recommended settings for VSCode complete with required formatter and settings as well as recommended extensions.

Required Extensions:
- black

Reccomended:
- Copilot
- Augment
- Gitless
- Parquet Viewer
- githubactions

## Contributing

Contributions to this repository are welcome! If you'd like to contribute, please follow these steps:

1. Clone the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to the remote repository.
5. Open a pull request against the `main` branch of this repository.

Please ensure that your code follows the project's coding standards and includes appropriate tests.

## License

Apache 2.0

## Acknowledgments
This project was developed by the Allen Institute for Artificial Intelligence (Ai2).

## Contact
patrickb@allenai.org
