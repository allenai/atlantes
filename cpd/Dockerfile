FROM python:3.12-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH} \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    CPD_ADDRESS=0.0.0.0 \
    CPD_PORT=5206

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /cpd

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache . && \
    useradd --system --uid 10001 --no-create-home cpd
USER 10001

EXPOSE 5206

HEALTHCHECK --interval=2s --timeout=2s --start-period=5s --retries=30 \
    CMD ["python", "-c", "import os, grpc; from grpc_health.v1 import health_pb2, health_pb2_grpc; ch = grpc.insecure_channel(f'127.0.0.1:{os.environ[\"CPD_PORT\"]}'); r = health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(service='ChangepointService'), timeout=1); raise SystemExit(r.status != health_pb2.HealthCheckResponse.SERVING)"]

CMD ["atlantes-cpd-serve"]
