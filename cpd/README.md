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
(or `$CPD_TEST_DATA_DIR`); they skip when the files are absent.

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
