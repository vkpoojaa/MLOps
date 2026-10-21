# Final Submission Checklist

This checklist distinguishes repository-complete work from evidence that must be captured in the actual target environment. It does not turn source code into live evidence.

| Item | Status | Evidence / action |
|---|---|---|
| README | **COMPLETE** | `README.md` |
| Architecture diagram | **COMPLETE** | `docs/ARCHITECTURE.md`, `docs/DESIGN_NOTE.md` |
| Workflow graph | **COMPLETE** | `docs/WORKFLOW_GRAPH.md` |
| YAML configuration source of truth | **COMPLETE** | `config/config.yaml`, config tests |
| Configuration validation | **COMPLETE** | `validate-config` evidence |
| Ingestion | **COMPLETE** | `src/mlops_pipeline/data/loading.py`, stage evidence |
| Validation | **COMPLETE** | `src/mlops_pipeline/data/validation.py`, tests |
| Data Gate | **COMPLETE** | gate tests + bad-data evidence |
| Feature engineering | **COMPLETE** | `src/mlops_pipeline/features/`, stage evidence |
| Training | **COMPLETE** | model training tests/evidence |
| Evaluation | **COMPLETE** | evaluation tests/evidence |
| Model Gate | **COMPLETE** | absolute + production-comparison tests |
| MLflow tracking implementation | **COMPLETE** | `src/mlops_pipeline/monitoring/mlflow_tracking.py`; live service still required |
| Live MLflow run screenshot | **NEEDS SCREENSHOT/EVIDENCE** | Capture run, params, metrics and artifacts from real MLflow |
| Registry implementation | **COMPLETE** | registry module + tests |
| Live registered production version | **NEEDS MANUAL ACTION** | Run pipeline against real MLflow Registry |
| Live lineage demonstration | **NEEDS SCREENSHOT/EVIDENCE** | Capture alias → version → run → config/code/data evidence |
| Prefect implementation | **COMPLETE** | `pipeline/prefect_flow.py` |
| Prefect live run history | **NEEDS SCREENSHOT/EVIDENCE** | Capture Prefect flow/task states and run history |
| Independent stage reruns | **COMPLETE** | `scripts/run_stage.py`, runbook, evidence |
| FastAPI implementation | **COMPLETE** | API tests |
| Dockerfile | **COMPLETE** | `Dockerfile`, policy tests |
| Docker build | **NEEDS MANUAL ACTION** | Build from a real Git checkout |
| Docker container/API acceptance | **NEEDS MANUAL ACTION** | Start image and verify `/health`, `/predict`, `/metrics` |
| CI PR workflow | **COMPLETE** | `.github/workflows/pr-checks.yml` |
| Failing PR evidence | **NEEDS SCREENSHOT/EVIDENCE** | Create/capture an intentionally failing PR run in GitHub |
| Passing PR evidence | **NEEDS SCREENSHOT/EVIDENCE** | Capture subsequent passing PR run |
| Release workflow | **COMPLETE** | `.github/workflows/release.yml` |
| Published GitHub Release | **NEEDS SCREENSHOT/EVIDENCE** | Merge to `main` and capture release |
| Commit-traceable image | **NEEDS MANUAL ACTION** | Capture GHCR image tag/digest and Git SHA |
| Rollback implementation | **COMPLETE** | registry-only rollback tests + runbook |
| Live rollback evidence | **NEEDS MANUAL ACTION** | Execute against real MLflow and capture before/after alias + API version |
| Prometheus endpoint | **COMPLETE** | `/metrics` + monitoring tests |
| Real API → Prometheus traffic | **COMPLETE** | Phase 13/15 monitoring smoke evidence |
| Grafana dashboard definition | **COMPLETE** | `monitoring/grafana/dashboards/house-price-operational.json` |
| Live Grafana dashboard | **NEEDS SCREENSHOT/EVIDENCE** | Start Docker stack, send traffic, capture changing panels |
| Request count/rate | **COMPLETE** | Prometheus metric/dashboard query |
| Latency | **COMPLETE** | histogram + p95 dashboard query |
| Errors | **COMPLETE** | HTTP error counters + dashboard |
| Prediction distribution | **COMPLETE** | prediction histogram + dashboard |
| Data drift methodology | **COMPLETE** | Phase 14 report/config |
| Performance drift methodology | **COMPLETE** | Phase 14 report/config |
| Batch 1 result | **COMPLETE** | Phase 14 evidence |
| Batch 2 result | **COMPLETE** | Phase 14 evidence |
| Batch 3 result | **COMPLETE** | Phase 14 evidence |
| Supplied incoming datasets preserved | **COMPLETE** | SHA-256 integrity evidence |
| Reproducibility implementation | **COMPLETE** | reproducibility checker/tests |
| Clean-clone reproducibility evidence | **NEEDS MANUAL ACTION** | Clone repository into fresh environment and rerun required checks |
| AI-use disclosure | **COMPLETE** | `docs/AI_USE_DISCLOSURE.md` |
| Official brief AI wording | **NOT REQUIRED** | The supplied official brief does not specify a separate AI-disclosure section; repository disclosure is included for transparency |
| Two-member commits | **NEEDS SCREENSHOT/EVIDENCE** | Capture Git history showing commits by both members |
| Two-member demonstration | **NEEDS MANUAL ACTION** | Both members explain/modify/justify their sections live or in recording |
| Final 5–10 minute demo | **NEEDS MANUAL ACTION** | Record/present using `FINAL_DEMO_SCRIPT.md` |
| Short reflection | **COMPLETE** | `docs/REFLECTION.md` |
| Final integration report | **COMPLETE** | `FINAL_INTEGRATION_REPORT.md` |
| Final compliance audit | **COMPLETE** | `FINAL_AUDIT.md` |
| Remediation log | **COMPLETE** | `FINAL_AUDIT_REMEDIATION_LOG.md` |
| Secrets/credentials committed | **NOT REQUIRED** | No secrets should be added; verify repository/CI secrets before submission |

## Final manual acceptance order

1. Push the repository to the real GitHub repository.
2. Confirm both members have identifiable commits.
3. Configure MLflow tracking/Registry and run one real approved pipeline.
4. Capture MLflow run, Registry version/alias, and lineage evidence.
5. Run the Prefect flow in the real Prefect-capable environment and capture run history.
6. Build/start the Docker image with a real Git SHA.
7. Start Prometheus/Grafana, generate traffic, and capture the dashboard update.
8. Execute the registry rollback and capture before/after production alias plus API health.
9. Create/capture a failing PR and then a passing PR.
10. Merge to `main` and capture the release/GHCR evidence.
11. Run the clean-clone reproducibility check.
12. Record the 5–10 minute demo with both members.

Do not change the source datasets during these acceptance activities.
