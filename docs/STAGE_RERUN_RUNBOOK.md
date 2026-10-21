# Independent Stage Rerun Runbook

The production stage implementations remain in `src/mlops_pipeline/pipeline/prefect_flow.py`. `scripts/run_stage.py` exposes the non-registry stages as independently runnable entry points without introducing a second implementation.

## Commands

```bash
PYTHONPATH=src python scripts/run_stage.py ingest
PYTHONPATH=src python scripts/run_stage.py validate
PYTHONPATH=src python scripts/run_stage.py data-gate
PYTHONPATH=src python scripts/run_stage.py feature-engineering
PYTHONPATH=src python scripts/run_stage.py train
PYTHONPATH=src python scripts/run_stage.py evaluate
PYTHONPATH=src python scripts/run_stage.py model-gate
PYTHONPATH=src python scripts/run_stage.py monitor
```

Each command starts from the configured input and executes only through the requested stage. A failure exits non-zero and does not silently continue.

Registry promotion and production rollback remain protected pipeline-only operations. They are deliberately not exposed as an unrestricted local stage command. Promotion occurs only through `registry_promotion_task` after the Model Gate; rollback uses the dedicated `rollback-production` command/flow and the MLflow production alias.

## Stage contracts

| Stage | Input | Output | Re-runnable independently? |
|---|---|---|---|
| Ingest | configured CSV | in-memory DataFrame | Yes |
| Validate | ingested DataFrame | `ValidationReport` | Yes |
| Data Gate | DataFrame + validation report | gated DataFrame | Yes |
| Feature Engineering | gated DataFrame | train/validation split | Yes |
| Train | split + config | baseline/candidate `ModelResult` | Yes |
| Evaluate | split + model results | evaluation report + artifacts | Yes |
| Model Gate | model results + production record | gate decision | Yes |
| Registry Promotion | approved gate + candidate | registry version/alias transition | Pipeline-only |
| Deploy/Serve | registry production alias | FastAPI serving process | Containerized acceptance |
| Operational Monitor | serving traffic | Prometheus/Grafana metrics | Yes |
| Drift Monitor | training reference + supplied incoming batches + production model | batch drift/performance report | Yes |

This interface is intended for reproducibility and stage-level demonstrations; it does not bypass Data Gate, Model Gate, registry safeguards, or production safety controls.
