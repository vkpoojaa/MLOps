# PHASE 11 HANDOFF — GitHub Actions CI/CD and Commit-Traceable Release

## 1. Outcome and honest evidence boundary

Phase 11 implements the CI/CD workflow requested by the official Project 4 brief:

- Pull requests run dependency installation, YAML configuration validation, unit tests, integration tests, and a lightweight pipeline smoke test.
- A push to `main` (normally the result of a merged PR) runs the same checks again. Only if they pass does the release job build and publish a Docker image to GitHub Container Registry (GHCR) and create a GitHub Release.
- The image tag, OCI revision label, GitHub Release, and attached `release-manifest.json` identify the exact source commit.
- CI/release does **not** train, register, promote, or deploy a model. Data Gate and Model Gate remain mandatory in the ML pipeline; passing software tests is not permission to change the production model.

**Important:** local checks were executed and recorded, including an intentionally failing temporary pytest assertion followed by a corrected passing check. They are not represented as GitHub Actions runs. The supplied project ZIP has no `.git` metadata or configured remote, and this session cannot open PRs or run GitHub Actions. Accordingly, actual failing/passing PR run URLs, a published release, and a live image rollback remain pending. No GitHub evidence is fabricated.

## 2. Workflow files

### `.github/workflows/pr-checks.yml`

**Triggers**

- `pull_request`: opened, synchronize, reopened, and ready-for-review.
- `push` to `main` (extra protection when the main branch changes).
- `workflow_dispatch` for manual checks.

**Steps**

1. Check out the exact commit.
2. Set up Python 3.12 with pip caching.
3. Install `requirements-dev.txt` (which includes runtime dependencies plus pytest).
4. Run `python -m mlops_pipeline validate-config`.
5. Run `pytest -q tests/unit`.
6. Run `pytest -q tests/integration`.
7. Run `scripts/ci_pipeline_smoke.py`.
8. Upload JUnit reports even when a prior step fails, when available.

Tests are not marked `continue-on-error`; a failing required command fails the job. Configure GitHub branch protection to require the `PR Checks / Config, tests, and lightweight pipeline smoke` check and restrict direct pushes to `main`.

### `.github/workflows/release.yml`

**Trigger:** push to `main`, after a PR merge under the recommended branch protection settings.

The workflow first repeats configuration validation, unit tests, integration tests, and the lightweight Data Gate smoke. `release-image` has `needs: release-checks`, so the image is not published if any release check fails.

When checks pass, it:

1. Resolves the project version from `pyproject.toml` and the full commit SHA from `github.sha`.
2. Builds the existing Phase 10 Dockerfile with the full commit passed as `GIT_COMMIT`.
3. Pushes to GHCR with immutable tags `<project-version>-<12-char-commit>` and `sha-<full-commit>`.
4. Adds OCI source, version, and revision labels; enables BuildKit provenance and SBOM output.
5. Creates `release-manifest.json` containing the image tags, commit, source ref, workflow run ID, and release policy.
6. Creates a GitHub Release tagged `v<project-version>-<12-char-commit>` and attaches the manifest.

The workflow only releases the API image. It does not run the orchestration command, change MLflow aliases, promote a model, or deploy to production. Model promotion remains pipeline-controlled after the Data Gate and Model Gate pass.

## 3. Lightweight pipeline test

`scripts/ci_pipeline_smoke.py` runs the real configured ingestion, validation, and Data Gate stages against the supplied training CSV. It checks that all rows pass the Data Gate and exits non-zero on failure. It intentionally stops before training, tracking, registration, and promotion so a PR check is fast and cannot change production state.

Observed local result:

```text
CI PIPELINE SMOKE PASSED
rows=4000 columns=11
data_gate=PASS checks=59 failures=0
scope=ingestion+validation+Data Gate; no training/registry/promotion
```

This smoke test supplements, rather than replaces, the integration tests that cover the full flow and gate behavior.

## 4. Test and check results

| Check | Local result | Evidence file |
|---|---:|---|
| Config validation | PASS | `reports/phase11_evidence/config_validation.log` |
| Unit tests | 226 passed | `reports/phase11_evidence/unit_tests.log` |
| Integration tests | 32 passed | `reports/phase11_evidence/integration_tests.log` |
| Lightweight Data Gate smoke | PASS, 59 checks | `reports/phase11_evidence/pipeline_smoke.log` |
| Workflow YAML parse | PASS | Repeated with PyYAML `BaseLoader`; workflow policy tests also cover triggers/commands |
| Python compilation | PASS | `python -m compileall -q src tests scripts` |
| CI policy tests | 3 passed | `tests/unit/test_phase11_ci_policy.py` |

Machine-readable evidence: `reports/phase11_evidence/phase11_evidence.json`.

## 5. A/B intentional failure and correction demonstration

The check-failure behavior was demonstrated locally using a temporary test outside the project repository, then a corrected temporary test. Neither temporary test is committed to the project suite.

**A — intentionally failing check**

- Command: `python -m pytest -q /tmp/phase11-demo/test_intentional_ci_failure.py`
- Observed: `1 failed`; process exit code `1`.
- Exact failure message: `PHASE11_INTENTIONAL_FAILURE_DEMO: this check must fail`.
- Log: `reports/phase11_evidence/local_intentional_failure.log`.

**B — corrected passing check**

- Command: `python -m pytest -q /tmp/phase11-demo/test_corrected_ci_check.py`
- Observed: `1 passed`; process exit code `0`.
- Log: `reports/phase11_evidence/local_corrected_pass.log`.

This demonstrates pytest's non-zero/zero exit behavior locally, not a real PR check. To satisfy the official *GitHub history* criterion, push the repository to GitHub and create two actual PR runs: first add a temporary failing assertion on a branch and capture the red `PR Checks` run, then correct/remove it, push again, and capture the green run. Keep both run URLs with the submission.

## 6. Release evidence status

**Implemented but not yet executed against GitHub/GHCR.** The workflow creates a GitHub Release and attached manifest after the main-branch checks pass. There is no published image digest, GitHub Release URL, or workflow run ID available from this local source ZIP; those values must come from the real Actions run.

After publishing, retain:

- Main Release workflow run URL and run ID.
- GitHub Release URL and tag.
- Attached `release-manifest.json`.
- GHCR image digest and tags.
- Image inspect output showing `org.opencontainers.image.revision` equals the full Git SHA.

## 7. Commands

Run locally from the repository root:

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m pytest -q tests/unit
PYTHONPATH=src python -m pytest -q tests/integration
PYTHONPATH=src python scripts/ci_pipeline_smoke.py
python -m compileall -q src tests scripts
```

Check workflow YAML syntax (PyYAML is installed with the project dependencies):

```bash
python - <<'PY'
from pathlib import Path
import yaml
for path in Path('.github/workflows').glob('*.yml'):
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert 'on' in workflow and 'jobs' in workflow
    print(path, list(workflow['jobs']))
PY
```

Once pushed, open **Actions → PR Checks** and **Actions → Main Release**. The release image name uses the lower-case GitHub repository path under `ghcr.io`.

## 8. Rollback

See `docs/RELEASE_AND_ROLLBACK.md` for the runbook. The rollback is an API image rollback to the previous immutable image tag, with the same registry configuration and secrets. It does not manually change the MLflow production alias. Model rollback, if needed, must use the approved registry/pipeline procedure.

A live rollback has not been tested because no image was published and no Docker engine/remote deployment is available in this session. Do not claim the official rollback-tested criterion is complete until the procedure has been exercised in the actual deployment environment and the result recorded.

## 9. Phase 12 instructions

1. Push this repository to GitHub and configure `main` branch protection, requiring the `PR Checks` job.
2. Capture a genuine failing PR run and then a corrected passing PR run; include their URLs in the evidence report.
3. Merge a passing PR. Verify `Main Release` re-runs checks, publishes the image, and creates the GitHub Release.
4. Compare the release manifest commit to the GitHub commit and OCI `org.opencontainers.image.revision` label; record the pushed image digest.
5. Run the container with the real MLflow registry/artifact store and verify `/health`, `/predict`, and `/metrics`; confirm the approved model version is loaded and no local fallback exists.
6. Perform and record a rollback to the prior image tag, then restore the current version if appropriate.
7. Continue the monitoring/drift task from the official brief: process all three incoming batches, record data/performance drift and operational alerts, and document the response without bypassing the Data Gate or Model Gate.
