# Project 4 Design Note

## Objective and non-functional requirements

Predict property asking price (`Price_Lakhs`) from the supplied listings. The primary success metric is MAE in ₹ lakhs: lower is better and the number can be read directly as average absolute pricing error. The service exposes a FastAPI endpoint and operational Prometheus/Grafana monitoring. Model freshness is governed by evidence-based drift/performance alerts and retraining remains subject to the Model Gate.

## Architecture

```mermaid
flowchart LR
    A[Versioned raw data] --> B[Prefect: Ingest]
    B --> C[Validate]
    C --> D{Data Gate}
    D -->|fail| X[Stop; keep production]
    D -->|pass| E[Feature Engineering]
    E --> F[Train]
    F --> G[Evaluate + leakage checks]
    G --> H{Model Gate}
    H -->|fail| X
    H -->|pass| I[MLflow Tracking]
    I --> J[MLflow Registry]
    J --> K[Production alias]
    K --> L[FastAPI]
    L --> M[Prometheus]
    M --> N[Grafana]
    A --> O[Drift + performance monitor]
    O --> H
    J --> P[Registry-only rollback]
```

## Lifecycle traceability

| Lifecycle stage | Component |
|---|---|
| Data | YAML paths, ingestion, validation, Data Gate |
| Experiment | training/evaluation, MLflow tracking |
| Model management | Model Gate, MLflow Registry, promotion/rollback |
| Deployment | FastAPI, Docker, CI/CD release workflow |
| Monitoring | Prometheus/Grafana operational metrics; separate drift/performance monitor |

The YAML file is the configuration source of truth. Production transitions are pipeline-controlled. Drift is an alerting signal and cannot bypass the Model Gate.
