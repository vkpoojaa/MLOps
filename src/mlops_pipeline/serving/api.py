"""FastAPI serving layer for the approved Phase 7 production model alias."""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from ..config.loader import Settings, load_config
from .registry_loader import load_approved_production_model


class PredictionRequest(BaseModel):
    """Batch prediction input; feature-specific validation uses config.yaml."""
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, Any]] = Field(min_length=1, max_length=100)


class OperationalMetrics:
    """Prometheus metrics for service operations only.

    These metrics intentionally cover request/serving behavior, not data drift or
    model-performance quality. A per-app registry keeps tests isolated and avoids
    duplicate metric-registration errors when multiple FastAPI apps are created.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "mlops_http_requests_total",
            "Total HTTP requests handled by the prediction service.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_request_latency = Histogram(
            "mlops_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "path"],
            registry=self.registry,
        )
        self.prediction_values = Histogram(
            "mlops_prediction_value",
            "Distribution of predicted house prices in configured target units.",
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000),
            registry=self.registry,
        )
        self.prediction_requests = Counter(
            "mlops_prediction_requests_total",
            "Total /predict requests accepted for model inference.",
            registry=self.registry,
        )
        self.prediction_errors = Counter(
            "mlops_prediction_errors_total",
            "Total prediction validation or inference errors.",
            registry=self.registry,
        )


class ServingRuntime:
    def __init__(self, operational_metrics: OperationalMetrics | None = None) -> None:
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.prediction_count = 0
        self.prediction_errors = 0
        self.total_prediction_seconds = 0.0
        self._lock = threading.Lock()
        self.operational_metrics = operational_metrics

    def snapshot_metrics(self) -> dict[str, Any]:
        with self._lock:
            count = self.prediction_count
            errors = self.prediction_errors
            total = self.total_prediction_seconds
        return {
            "prediction_requests_total": count,
            "prediction_errors_total": errors,
            "prediction_latency_seconds_total": round(total, 6),
            "prediction_latency_seconds_mean": round(total / count, 6) if count else 0.0,
        }


def _validate_instances(instances: list[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    cfg = settings.config
    expected = cfg.feature_columns
    numeric = set(cfg.numeric_features)
    categorical = set(cfg.categorical_features)
    normalized: list[dict[str, Any]] = []

    for row_index, row in enumerate(instances):
        missing = [column for column in expected if column not in row]
        extra = [column for column in row if column not in expected]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing feature(s): {', '.join(missing)}")
            if extra:
                details.append(f"unexpected feature(s): {', '.join(extra)}")
            raise ValueError(f"instance {row_index}: {'; '.join(details)}")

        clean: dict[str, Any] = {}
        for column in expected:
            value = row[column]
            if column in numeric:
                if value is None:
                    clean[column] = None
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"instance {row_index}: {column} must be a number or null")
                if not math.isfinite(float(value)):
                    raise ValueError(f"instance {row_index}: {column} must be finite")
                clean[column] = float(value)
            elif column in categorical:
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"instance {row_index}: {column} must be a string or null")
                clean[column] = value
        normalized.append(clean)

    # Keep precisely configured feature order. The logged sklearn pipeline owns
    # all imputation/encoding/scaling; this API does not reimplement it.
    return pd.DataFrame(normalized, columns=expected)


def create_app(
    *,
    model_loader: Callable[[Settings], tuple[Any, dict[str, Any]]] | None = None,
    settings_loader: Callable[[], Settings] | None = None,
) -> FastAPI:
    """Application factory permits deterministic tests without a live registry.

    Production defaults always resolve the configured MLflow production alias.
    Injected loaders are intended for tests only; no local model fallback exists.
    """
    load_settings = settings_loader or load_config
    load_model = model_loader or load_approved_production_model
    operational_metrics = OperationalMetrics()
    runtime = ServingRuntime(operational_metrics)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            settings = load_settings()
            app.state.settings = settings
            runtime.model, runtime.metadata = load_model(settings)
            runtime.load_error = None
        except Exception as exc:
            # Keep the process alive for health diagnostics, but never serve a
            # fallback artifact when the registry-backed model is unavailable.
            runtime.model = None
            runtime.metadata = {}
            runtime.load_error = str(exc)
        yield

    app = FastAPI(
        title="House Price Prediction API",
        version="1.0.0",
        description="Predictions from the Model-Gate-approved MLflow production alias.",
        lifespan=lifespan,
    )
    app.state.serving_runtime = runtime
    app.state.settings = None

    def build_metadata() -> dict[str, str]:
        # Values are injected at image build time and also remain overrideable
        # for orchestrators; they identify the exact source revision deployed.
        return {
            "version": os.getenv("APP_VERSION", "0.1.0"),
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "image": os.getenv("IMAGE_NAME", "mlops-house-price-api"),
        }

    @app.middleware("http")
    async def prometheus_http_metrics(request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            elapsed = time.perf_counter() - started
            path = request.url.path
            operational_metrics.http_requests.labels(
                method=request.method, path=path, status=str(status_code)
            ).inc()
            operational_metrics.http_request_latency.labels(
                method=request.method, path=path
            ).observe(elapsed)

    @app.get("/health")
    def health() -> JSONResponse:
        if runtime.model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "model_loaded": False,
                         "reason": "approved production model could not be loaded",
                         "build": build_metadata()},
            )
        return JSONResponse(status_code=200, content={
            "status": "ok", "model_loaded": True,
            "model": runtime.metadata,
            "build": build_metadata(),
        })

    @app.post("/predict")
    def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        if runtime.model is None:
            raise HTTPException(status_code=503, detail="Approved production model is unavailable")
        settings: Settings | None = app.state.settings
        if settings is None:
            raise HTTPException(status_code=503, detail="Serving configuration is unavailable")
        try:
            frame = _validate_instances(payload.instances, settings)
        except ValueError as exc:
            with runtime._lock:
                runtime.prediction_errors += 1
            operational_metrics.prediction_errors.inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        started = time.perf_counter()
        try:
            raw_predictions = runtime.model.predict(frame)
            predictions = [float(value) for value in list(raw_predictions)]
            if len(predictions) != len(frame) or not all(math.isfinite(value) for value in predictions):
                raise ValueError("model returned invalid prediction output")
        except Exception as exc:
            with runtime._lock:
                runtime.prediction_errors += 1
                runtime.total_prediction_seconds += time.perf_counter() - started
            operational_metrics.prediction_errors.inc()
            raise HTTPException(status_code=500, detail="Prediction failed safely") from exc

        elapsed = time.perf_counter() - started
        with runtime._lock:
            runtime.prediction_count += 1
            runtime.total_prediction_seconds += elapsed
        operational_metrics.prediction_requests.inc()
        for prediction in predictions:
            operational_metrics.prediction_values.observe(prediction)
        return {
            "predictions": predictions,
            "prediction_unit": settings.config.target,
            "count": len(predictions),
            "model": runtime.metadata,
        }

    @app.get("/metrics", response_class=Response)
    def metrics() -> Response:
        """Prometheus scrape endpoint. Operational metrics only."""
        return Response(
            content=generate_latest(operational_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/metrics/json")
    def metrics_json() -> dict[str, Any]:
        """Human-readable compatibility endpoint; not used by Prometheus."""
        return {
            "service": "house-price-prediction-api",
            "model_loaded": runtime.model is not None,
            "build": build_metadata(),
            **runtime.snapshot_metrics(),
        }

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        # Avoid returning stack traces or filesystem/registry internals to callers.
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
