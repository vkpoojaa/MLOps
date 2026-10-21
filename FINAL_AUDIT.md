# FINAL COMPLIANCE AUDIT — Project 4 (Post-Remediation)

**Audit mode:** remediation verification. Only gaps identified in the prior `FINAL_AUDIT.md` were addressed. No unrelated feature work was added.

**Official brief:** `docs/MLOps_Project_4_Capstone_End_to_End_Pipeline.docx`

## Status legend

- **PASS** — evidence in this repository or an executed test supports the requirement.
- **FIXED** — the prior repository gap was corrected and the correction was tested, but the brief's final acceptance still depends on external/live evidence where explicitly stated.
- **FAIL** — the requirement remains unproven or unmet; no evidence is fabricated.
- **UNVERIFIABLE** — evidence depends on the real team/GitHub/submission environment and is not present in the supplied archive.

## Requirement-by-requirement matrix

| Requirement | Required by brief | Implemented? | Evidence | File/location | Test status | Status |
|---|---|---|---|---|---|---|
| YAML source of truth | Task 2 | Yes | Paths, thresholds, hyperparameters and monitoring calibration controls are YAML-driven | `config/config.yaml`, `src/mlops_pipeline/config/`, `src/mlops_pipeline/monitoring/drift.py` | 241 unit tests + config validation pass; no bootstrap constants remain in drift code | **FIXED** |
| Configuration validation | Task 2 | Yes | Typed schema and cross-section validation | `src/mlops_pipeline/config/schema.py`, `src/mlops_pipeline/config/validation.py` | `validate-config`: 0 pending settings; unit suite PASS | **PASS** |
| Reproducibility | Task 2 | Partial | Local reproducibility evidence exists; clean-clone/live environment remains unproven | `reports/reproducibility_check.json`, `src/mlops_pipeline/monitoring/reproducibility.py` | Existing reproducibility tests PASS | **FAIL** |
| Ingestion | Task 3 | Yes | Configured training CSV ingestion task | `src/mlops_pipeline/pipeline/prefect_flow.py` | Independent ingest stage PASS; integration suite PASS | **PASS** |
| Validation | Task 3/4 | Yes | Schema, ranges, missingness, categories, target, duplicates, identifiers | `src/mlops_pipeline/data/validation.py` | Integration/unit tests PASS; independent validate stage PASS | **PASS** |
| Feature engineering | Task 3 | Yes | Config-driven preprocessing/split | `src/mlops_pipeline/features/`, `src/mlops_pipeline/data/splitting.py` | Integration PASS; independent stage PASS | **PASS** |
| Training | Task 3 | Yes | Baseline/candidate training | `src/mlops_pipeline/models/training.py`, `factory.py` | Unit/integration tests PASS | **PASS** |
| Evaluation | Task 3 | Yes | MAE/RMSE/R²/CV/comparison | `src/mlops_pipeline/models/evaluation.py` | Unit/integration tests PASS | **PASS** |
| Data Gate | Task 4 | Yes | Validation failure blocks downstream training | `prefect_flow.py`, `tests/integration/test_prefect_flow.py` | Failure scenarios PASS; independent data-gate PASS | **PASS** |
| Model Gate | Task 4 | Yes | Absolute threshold + production comparison | `models/quality_gate.py`, tests | Gate tests PASS | **PASS** |
| Absolute threshold | Task 4 | Yes | Configured measured MAE threshold | `config/config.yaml`, `PHASE_6_HANDOFF.md` | Gate tests PASS | **PASS** |
| Production comparison | Task 4 | Yes | Candidate must improve production | `models/quality_gate.py` | Worse-but-absolute-pass scenario PASS | **PASS** |
| Production safety | Task 4 | Yes | Failed gates preserve production | `prefect_flow.py`, `serving/registry_loader.py` | Failure tests PASS | **PASS** |
| MLflow tracking | Task 5 | Implemented | Tracking code logs run/data/config/code/model evidence | `monitoring/mlflow_tracking.py` | Controlled tests PASS; live service unavailable in supplied environment | **FAIL** |
| Registry | Task 5 | Implemented | MLflow registered-model/alias implementation | `monitoring/model_registry.py` | Controlled tests PASS; live registry unavailable | **FAIL** |
| Promotion | Task 5 | Implemented | Promotion only after Model Gate | `pipeline/prefect_flow.py`, `monitoring/model_registry.py` | Controlled tests PASS; live transition unverified | **FAIL** |
| Lineage | Task 5 | Not currently evidenced | Existing lineage verifier still requires real registered version/run/config/code chain | `reports/lineage_verification.json`, `monitoring/lineage.py` | Current live lineage report remains FAIL | **FAIL** |
| Prefect | Task 3 | Implemented | Named Prefect tasks/flow | `pipeline/prefect_flow.py` | Local fallback/integration tests PASS; live Prefect UI evidence absent | **FAIL** |
| Independent stage reruns | Task 3 acceptance | Yes | Explicit stage-run CLI and stage contract runbook added | `scripts/run_stage.py`, `docs/STAGE_RERUN_RUNBOOK.md` | ingest/validate/data-gate/feature-engineering independently PASS | **FIXED** |
| Stage-level logs | Task 3 acceptance | Yes in code | Each Prefect task logs stage outcome/errors | `pipeline/prefect_flow.py` | Static/code evidence PASS; live history absent | **FAIL** |
| Run history | Task 3 acceptance | No live evidence | No real Prefect UI/run-history capture in archive | `reports/`, `docs/WORKFLOW_GRAPH.md` | Not run live | **FAIL** |
| FastAPI | Task 7 | Yes | `/health`, `/predict`, `/metrics`, registry-backed serving | `serving/api.py`, `registry_loader.py` | API tests PASS | **PASS** |
| Docker | Task 6/7 | Implemented | Dockerfile, healthcheck, traceability args | `Dockerfile`, `scripts/build-image.sh` | Static policy tests PASS; actual Docker runtime unavailable | **FAIL** |
| CI/CD workflows | Task 6 | Yes | PR and release workflows | `.github/workflows/` | YAML/policy tests PASS; GitHub history absent | **FAIL** |
| Failing PR evidence | Task 6 acceptance | No live evidence | Only local intentional failure | `reports/phase11_evidence/local_intentional_failure.log` | Not a GitHub PR run | **FAIL** |
| Passing PR evidence | Task 6 acceptance | No live evidence | Local corrected pass only | `reports/phase11_evidence/local_corrected_pass.log` | Not a GitHub PR run | **FAIL** |
| Release | Task 6 acceptance | No live evidence | Release workflow exists; no published release record | `.github/workflows/release.yml` | Not run in real GitHub environment | **FAIL** |
| Commit-traceable image | Task 6 acceptance | Implemented | Workflow tags/labels image with Git SHA | `.github/workflows/release.yml`, `Dockerfile` | Static tests PASS; released image absent | **FAIL** |
| Rollback | Task 6 acceptance | Yes | Registry alias v2→v1 controlled rollback; no file copy in rollback | `monitoring/model_registry.py`, rollback tests | Phase 12 rollback tests PASS | **PASS** |
| Prometheus | Task 7 | Yes | Prometheus client + scrape config | `serving/api.py`, `monitoring/prometheus/prometheus.yml` | Real API traffic changed metrics; PASS | **PASS** |
| Grafana | Task 7 | Implemented | Provisioned datasource/dashboard | `monitoring/grafana/` | Dashboard config tests PASS; live Grafana unavailable | **FAIL** |
| Request count | Task 7 | Yes | Counter and dashboard | `serving/api.py`, Grafana dashboard | Real traffic evidence PASS | **PASS** |
| Latency | Task 7 | Yes | Histogram + p95 query; 100 ms alert ceiling configured | `serving/api.py`, `config/config.yaml` | Metrics tests PASS; live Grafana not observed | **PASS** |
| Errors | Task 7 | Yes | HTTP/prediction error counters; 1% alert ceiling configured | `serving/api.py`, `config/config.yaml` | Error-path metrics test PASS | **PASS** |
| Prediction distribution | Task 7 | Yes | Prediction-value histogram | `serving/api.py`, Grafana dashboard | Real traffic evidence PASS | **PASS** |
| Data drift | Task 7 | Yes | PSI with empirical training-reference null calibration | `monitoring/drift.py`, Phase 14 evidence | Batch monitor PASS | **PASS** |
| Performance drift | Task 7 | Yes | Actual `Price_Lakhs` MAE degradation | `monitoring/drift.py`, Phase 14 evidence | Batch monitor PASS | **PASS** |
| Batch 1 | Task 7 | Yes | No data/performance trigger | `reports/phase14_evidence/batch_summary.csv` | PASS | **PASS** |
| Batch 2 | Task 7 | Yes | Data + performance trigger | `reports/phase14_evidence/batch_summary.csv` | PASS | **PASS** |
| Batch 3 | Task 7 | Yes | Data + performance trigger | `reports/phase14_evidence/batch_summary.csv` | PASS | **PASS** |
| Alert response policy | Task 7 | Yes | Drift never bypasses Model Gate | `PHASE_14_HANDOFF.md`, `config/config.yaml` | Tests/evidence PASS | **PASS** |
| Documentation | Deliverables | Improved | README rewritten; design/runbook/workflow/disclosure/reflection added | `README.md`, `docs/` | Documentation tests PASS | **FIXED** |
| AI-use disclosure | Audit requirement | Yes | Honest disclosure added | `docs/AI_USE_DISCLOSURE.md` | Existence/content test PASS | **FIXED** |
| Meaningful contribution by both members | Rules 8 / Task 8 | Cannot verify | Archive has no `.git` history/contributor evidence | Real Git repository required | Not verifiable | **UNVERIFIABLE** |
| No secrets committed | Rules | Appears yes | No obvious secrets in supplied source | repository scan/workflows | Inspection PASS; real Git history still required | **PASS** |
| No manual model copying | Rules | Reviewed | Promotion's local mirror is automated; rollback is registry-only; serving uses registry alias | `monitoring/model_registry.py`, `registry_loader.py` | Controlled tests PASS | **PASS** |
| Only pipeline changes production | Rules | Yes by design | Registry transition task is gate-controlled | `prefect_flow.py`, `model_registry.py` | Controlled gate/registry tests PASS | **PASS** |
| Supplied datasets unmodified | Task 7/data rule | Yes | Incoming SHA-256 unchanged before/after monitoring | Phase 14 evidence | PASS | **PASS** |
| Architecture diagram | Task 1 | Yes | Mermaid architecture added | `docs/DESIGN_NOTE.md`, README | Documentation test PASS | **FIXED** |
| Design note ≤2 pages | Task 1 | Yes | Concise design note with architecture/lifecycle mapping | `docs/DESIGN_NOTE.md` | Content/size reviewed | **FIXED** |
| Numeric success thresholds + interpretation | Task 1 | Yes | MAE/gate/operational thresholds documented with rationale | `config/config.yaml`, `docs/DESIGN_NOTE.md` | Config validation PASS | **FIXED** |
| Workflow graph matching code | Task 3 | Yes | Mermaid graph maps all named Prefect tasks | `docs/WORKFLOW_GRAPH.md` | Code/documentation review PASS | **FIXED** |
| Evidence of two failing-gate scenarios | Deliverables | Yes | Bad-data + bad-model scenarios | `reports/phase15_evidence/`, Phase 15 report | Tests PASS | **PASS** |
| Lineage demonstration | Deliverables | No live evidence | Current lineage report fails without real MLflow | `reports/lineage_verification.json` | FAIL | **FAIL** |
| CI history including release | Deliverables | No live evidence | No GitHub run/release URLs or records | Phase 11 evidence | FAIL | **FAIL** |
| Monitoring dashboard/report covering three batches | Deliverables | Yes | Phase 14 report/charts cover all batches | `reports/phase14_evidence/` | PASS | **PASS** |
| 5–10 minute demonstration | Task 8 | No evidence | No recording/script/timing artifact | Repository | Not evidenced | **FAIL** |
| Both partners explain system | Task 8 | Cannot verify | No member demo/contribution evidence | Real team repository/demo required | Not verifiable | **UNVERIFIABLE** |
| Final reflection | Deliverables | Yes | One-page reflection added | `docs/REFLECTION.md` | Documentation test PASS | **FIXED** |

## Remediated gaps

### G8 — YAML-driven drift calibration — FIXED

Bootstrap run counts and sample sizes were previously constants in `drift.py`. They now live in `config/config.yaml`, are typed/validated by the config schema, and are consumed by the monitor. The old constants were removed. The monitoring report records the configured values.

### G9 — Declared monitoring dependency — FIXED

`matplotlib>=3.8,<4` is now declared in `requirements.txt`, matching its actual use by Phase 14 evidence report generation.

### G10 — Operational alert thresholds — FIXED

Operational ceilings are now configured in YAML:

- p95 latency: **100 ms**
- error rate: **1%**

The Phase 13 real-traffic smoke test observed zero prediction errors. These thresholds are documented as conservative interactive-service alert ceilings rather than being treated as model-quality gates.

### G6 — Independent stage reruns — FIXED

`scripts/run_stage.py` exposes independent entry points for ingestion, validation, Data Gate, feature engineering, training, evaluation, Model Gate and drift monitoring. `docs/STAGE_RERUN_RUNBOOK.md` defines inputs/outputs and explicitly keeps registry promotion and rollback pipeline-controlled.

### G11/G12 — Design/workflow artifacts — FIXED at repository level

Added `docs/DESIGN_NOTE.md` and `docs/WORKFLOW_GRAPH.md`. Live Prefect run-history and MLflow registry screenshots remain external evidence requirements and are not fabricated.

### G13 — AI disclosure — FIXED

Added `docs/AI_USE_DISCLOSURE.md` with an honest statement of AI-assisted development and the team's responsibility to verify and explain the submission.

### G16 — Reflection — FIXED

Added `docs/REFLECTION.md`.

### G17/G20 — Documentation consistency — FIXED

README now describes the final architecture, safety controls, monitoring split, CI/CD, evidence boundaries, and final documentation rather than claiming later phases are not started.

## Remaining gaps — not fabricated

### Live MLflow lineage

Still **FAIL** until a real MLflow environment is run from a real Git checkout and the deployed production version can be traced to code commit, data hash, config hash/path, run ID and registry version.

### GitHub PR/release evidence

Still **FAIL** until the real GitHub repository contains a failing PR run, corrected passing PR run, merge/release run and published release/GHCR image evidence.

### Docker runtime acceptance

Still **FAIL** until Docker is available and the image is built and run against the approved registry model.

### Live Grafana acceptance

Still **FAIL** until Prometheus/Grafana are started, real API traffic is generated, and the provisioned dashboard is observed changing.

### Prefect live run history

Still **FAIL** until the actual Prefect environment records a flow run and task history.

### Two-member contribution and final demonstration

Still **UNVERIFIABLE/FAIL** from this source archive because no real Git history, contribution mapping, recording, or 5–10 minute demonstration evidence is present. These cannot be honestly generated from repository files alone.

### Fresh-clone acceptance

Still **FAIL** as a final acceptance claim because a clean external clone/environment has not been demonstrated. Local reproducibility remains supported and the missing matplotlib dependency identified by the previous audit has been corrected.

## Remediation verification commands

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m pytest tests/unit -q
PYTHONPATH=src python -m pytest tests/integration -q
PYTHONPATH=src python -m mlops_pipeline monitor-batches
PYTHONPATH=src python scripts/run_stage.py ingest
PYTHONPATH=src python scripts/run_stage.py validate
PYTHONPATH=src python scripts/run_stage.py data-gate
PYTHONPATH=src python scripts/run_stage.py feature-engineering
```

Observed results in this audit:

- configuration validation: **PASS**, 0 pending settings;
- unit tests: **241 passed**;
- integration tests: **32 passed**;
- Phase 14 monitor: **PASS**, supplied batches unchanged and results reproduced;
- independent stage commands for ingest/validate/data-gate/feature-engineering: **PASS**.

## Final decision

**Project 4 is not yet fully compliant.** The repository-only remediation is complete for the gaps that can be fixed without inventing external evidence. The remaining blockers are live MLflow lineage, live Prefect history, real GitHub PR/release evidence, Docker runtime acceptance, live Grafana observation, clean-clone acceptance, and verifiable two-member contribution/final-demo evidence.

No unavailable result is marked PASS.
