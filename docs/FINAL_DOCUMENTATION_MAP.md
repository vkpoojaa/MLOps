# Final Documentation Map

| Topic | Primary document | Implementation source |
|---|---|---|
| Architecture | `docs/ARCHITECTURE.md` | `src/mlops_pipeline/` |
| Configuration | `README.md`, `config/config.yaml` | `src/mlops_pipeline/config/` |
| Data validation/Data Gate | `docs/DATA_VALIDATION.md` | `src/mlops_pipeline/data/` |
| Pipeline/workflow | `docs/WORKFLOW_GRAPH.md` | `src/mlops_pipeline/pipeline/prefect_flow.py` |
| Stage reruns | `docs/STAGE_RERUN_RUNBOOK.md` | `scripts/run_stage.py` |
| Model Gate | `README.md`, `PHASE_6_HANDOFF.md` | `src/mlops_pipeline/models/quality_gate.py` |
| MLflow/Registry/lineage | `README.md`, `docs/RELEASE_AND_ROLLBACK.md` | `src/mlops_pipeline/monitoring/` |
| API/deployment | `README.md` | `src/mlops_pipeline/serving/`, `Dockerfile` |
| CI/CD | `README.md`, `docs/RELEASE_AND_ROLLBACK.md` | `.github/workflows/` |
| Operational monitoring | `README.md`, `monitoring/README.md` | `src/mlops_pipeline/serving/`, `monitoring/` |
| Drift/performance | `README.md`, `PHASE_14_HANDOFF.md` | `src/mlops_pipeline/monitoring/drift.py` |
| Reproducibility | `README.md` | `src/mlops_pipeline/monitoring/reproducibility.py` |
| AI-use disclosure | `docs/AI_USE_DISCLOSURE.md` | documentation only |
| Final demo | `FINAL_DEMO_SCRIPT.md` | documentation only |
| Final submission evidence | `FINAL_SUBMISSION_CHECKLIST.md` | documentation only |
