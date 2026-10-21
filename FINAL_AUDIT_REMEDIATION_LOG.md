# Final Audit Remediation Log

This log records only corrections made in response to `FINAL_AUDIT.md`. No unrelated refactoring was performed.

| Audit gap | Problem | Minimum correction | Verification |
|---|---|---|---|
| G8 | Phase 14 bootstrap run counts/sample sizes were hard-coded in Python | Moved drift/performance bootstrap controls into `config/config.yaml` and typed schema; drift code now reads them from config | 241 unit tests pass; Phase 14 monitoring rerun PASS; hard-coding test PASS |
| G9 | Phase 14 report generation imported matplotlib without declaring it | Added `matplotlib>=3.8,<4` to runtime requirements | Unit suite PASS; dependency declaration inspected |
| G10 | Operational alert thresholds were `null` | Set 100 ms p95 latency and 1% error-rate ceilings in YAML and documented their rationale | Config validation PASS; compliance tests PASS |
| G6 | Independent stage rerun interface was not explicit | Added `scripts/run_stage.py` and `docs/STAGE_RERUN_RUNBOOK.md`, reusing existing stage implementations | Ingest/validate/data-gate/feature-engineering stage commands all PASS |
| G11 | Required design note and architecture diagram were missing | Added `docs/DESIGN_NOTE.md` with lifecycle-traceable Mermaid architecture | Documentation existence test PASS |
| G12/G18 | Workflow/run-history documentation was incomplete | Added `docs/WORKFLOW_GRAPH.md`; live Prefect history remains external acceptance evidence and is not fabricated | Graph matches named Prefect tasks; live history remains FAIL |
| G13 | AI-use disclosure was absent | Added honest `docs/AI_USE_DISCLOSURE.md` | Documentation existence test PASS |
| G16 | Final reflection was absent | Added one-page `docs/REFLECTION.md` | Documentation existence test PASS |
| G17/G20 | README and architecture documentation were stale/inconsistent | Rewrote README to describe the final Phase 15 architecture and evidence boundaries | Documentation inspection + full unit/integration tests PASS |
| G7 | Promotion copied candidate bytes into a second local production artifact | Removed the local model-byte copy from promotion. Registry version/alias is now the sole production artifact authority; metadata records the registry model URI. Existing local artifact is not changed by promotion. | Registry unit tests + rollback tests PASS; 241 unit tests and 32 integration tests PASS |

## Intentionally unresolved external-evidence gaps

No repository edit can honestly create the following evidence without access to the real submission environment: live MLflow lineage, real Prefect run history, GitHub failing/passing PR history, GitHub release/GHCR image, Docker runtime acceptance, live Grafana observation, and real two-member Git contribution history. These remain `FAIL` or `UNVERIFIABLE` in `FINAL_AUDIT.md` rather than being fabricated.

G7 was resolved by removing the promotion-time local model-byte copy. The MLflow registry version/production alias is now the sole production artifact authority. The local `selected_model.joblib` remains only as a legacy/local monitoring artifact and is not modified by production promotion or used by FastAPI serving.
