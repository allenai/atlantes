# atlantes-cpd

Online changepoint detection for AIS tracks. A changepoint marks where a track
should be split into a new subpath, which lets downstream inference run only
when a vessel's behavior may have changed. The detector is served over gRPC as
`atlanteschangepoint.ChangepointService`.

This is a standalone package. It does not import `atlantes` and has its own
dependencies, tests, Dockerfile, and CI workflow (`.github/workflows/cpd.yml`).

## Develop

```
cd cpd
python -m venv venv
venv/bin/pip install -e '.[dev]'
venv/bin/pytest tests --ignore=tests/integration/serve
```

The detector integration tests read three AIS track CSVs from `tests/data/`
(or `$CPD_TEST_DATA_DIR`); they skip when the files are absent. The tracks are licensed and must not be committed; `tests/data/*.csv` is gitignored.

## Run the service

```
docker compose up --build
```

Configuration comes from environment variables:

| Variable      | Default   | Purpose                  |
|---------------|-----------|--------------------------|
| `CPD_ADDRESS` | `0.0.0.0` | Address the server binds |
| `CPD_PORT`    | `5206`    | Port the server binds    |

With the service running, `CPD_ADDRESS=127.0.0.1 venv/bin/pytest tests/integration/serve` exercises it end to end.

## Tuning

Detection thresholds live in `src/atlantes_cpd/constants.py`.

## Regenerating the gRPC code

See `src/atlantes_cpd/serve/README.md`.

## Release to PyPI

The package publishes to PyPI as `atlantes-cpd` through trusted publishing; no API token is involved.

The repo has a single version, in the `VERSION` file at the repo root. `cpd/VERSION` is a symlink to it, and `pyproject.toml` reads it. A `v<version>` tag releases the whole repo, and `atlantes-cpd` is one of the artifacts.

1. Set `VERSION` and merge it.
2. Create a GitHub release with the tag `v<version>` (for example `v1.1.0`) and publish it. Drafts and pre-releases don't trigger the workflow.
3. `.github/workflows/publish_cpd.yml` checks that the tag matches `VERSION`, builds the sdist and wheel, and uploads them from the `pypi` environment.
