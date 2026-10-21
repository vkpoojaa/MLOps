# Prefect Workflow Graph

The graph below matches the named tasks in `src/mlops_pipeline/pipeline/prefect_flow.py`.

```mermaid
flowchart TD
  A[1-ingestion] --> B[2-validation]
  B --> C[3-data-gate]
  C --> D[4-feature-engineering]
  D --> E[5-training]
  E --> F[6-evaluation]
  F --> G[7-model-quality-gate]
  G -->|pass| H[8-register-and-promote]
  G -->|fail| X[Stop; production preserved]
  H --> I[FastAPI production alias]
  I --> J[Operational monitoring]
  K[Incoming batches] --> L[Drift/performance monitoring]
  L --> G
  R[9-rollback-production flow] --> H
```

### Run history

Live Prefect run-history evidence is environment-dependent and is not fabricated in this repository. Run the flow with Prefect installed and capture the resulting run history in the final submission.
