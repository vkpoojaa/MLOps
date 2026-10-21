# syntax=docker/dockerfile:1.7
# Python patch version is pinned; runtime package versions are pinned in the serving lock.
FROM python:3.12.3-slim-bookworm

ARG APP_VERSION=0.1.0
ARG GIT_COMMIT=unknown
ARG IMAGE_NAME=mlops-house-price-api

LABEL org.opencontainers.image.title="MLOps House Price Prediction API" \
      org.opencontainers.image.description="Registry-backed FastAPI serving for the approved production model" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_VERSION=${APP_VERSION} \
    GIT_COMMIT=${GIT_COMMIT} \
    IMAGE_NAME=${IMAGE_NAME} \
    MLOPS_CONFIG=/app/config/config.yaml

WORKDIR /app

# Refuse untraceable images: use scripts/build-image.sh from a real Git checkout.
RUN test -n "${GIT_COMMIT}" && test "${GIT_COMMIT}" != "unknown" || \
    (echo "GIT_COMMIT build argument is required; build from a Git checkout" >&2; exit 1)

# Install only the pinned serving dependencies; training/orchestration/test extras stay out.
COPY requirements-serving.lock /app/requirements-serving.lock
RUN python -m pip install --no-cache-dir pip==25.1.1 && \
    python -m pip install --no-cache-dir -r /app/requirements-serving.lock

# Deliberately do not copy data/, artifacts/, mlflow.db, mlruns/, or candidate models.
COPY src/ /app/src/
COPY config/config.yaml /app/config/config.yaml

# Drop privileges; mounted configuration and credentials should be read-only.
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); raise SystemExit(0 if r.status == 200 else 1)" || exit 1

CMD ["python", "-m", "mlops_pipeline.serving"]
