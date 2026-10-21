# CI/CD Release and Rollback Runbook

## Release policy

1. Open a pull request against `main`. The `PR Checks` workflow must pass configuration validation, unit tests, integration tests, and the lightweight ingestion/validation/Data Gate smoke run.
2. Configure repository branch protection so the `PR Checks / Config, tests, and lightweight pipeline smoke` job is required and direct pushes to `main` are restricted.
3. Merge the reviewed pull request. `Main Release` re-runs the same checks on the exact `main` commit; only after those pass does it build and publish the serving image to GHCR and create a GitHub Release.
4. Record the GitHub Release's attached `release-manifest.json`, image tag, full Git SHA, workflow run ID, and image digest from the workflow summary/package details.

This release workflow packages the API only. It does not train, register, promote a model, or deploy a production alias. Model promotion remains the responsibility of the ML pipeline after both Data Gate and Model Gate pass.

## Model rollback — Phase 12

A model rollback is different from an API image rollback. The production model is identified by the configured MLflow Model Registry alias (`production`). **Rollback changes that alias from the currently serving version to a previously approved registered version. It does not copy model files.**

The controlled path is the Prefect `house-price-production-rollback` flow and its `9-rollback-production` task. The task:

1. Resolves the current production alias and records the **before production version**.
2. Optionally enforces an `expected_current_version` safety check so a stale rollback command cannot overwrite an unexpected newer production version.
3. Verifies that the target registry version exists and has `model_gate.status=PASS`.
4. Calls `set_registered_model_alias(registered_model_name, production_alias, target_version)`.
5. Re-resolves the alias and verifies that the final production version is exactly the requested target.
6. Updates production metadata from registry/run lineage only. **No model artifact is downloaded, copied, or replaced.**
7. Writes `reports/rollback_history.jsonl` and `reports/rollback_latest.json` containing:
   - before rollback production version
   - promoted/current version being rolled back
   - rollback target and action
   - final production version
   - reason
   - target run ID
   - `model_file_copied: false`

If metadata/audit work fails after the alias transition, the code restores the alias to the version observed before the rollback and raises an error.

### CLI entry point

Use the pipeline-controlled command rather than editing MLflow manually:

```bash
PYTHONPATH=src python -m mlops_pipeline rollback-production 1 \
  --expected-current-version 2 \
  --reason "restore previous known-good production version"
```

The production alias is the authoritative serving identifier. The FastAPI serving loader resolves `models:/<registered-model>/<version>` from the alias at startup, so a restarted API loads the rolled-back version. It never falls back to `selected_model.joblib`.

### API image rollback

The API image rollback is separate. The API image does not contain a model; it loads the approved production model through MLflow. To roll back the API software version, identify the previous immutable GitHub Release image tag from its `release-manifest.json` and redeploy that image with the same MLflow configuration and credentials.

1. Identify the last known-good GitHub Release and exact image tag.
2. Preserve the same deployment configuration, registry URI, artifact-store credentials, and secret mounts.
3. Recreate the API using the previous immutable image tag.
4. Verify `/health` returns `200`, `/metrics` responds, and `/predict` succeeds. Confirm `/health.model.model_version` is still the approved registry version.
5. Record the incident, old/current image tags, reason, and health/prediction checks.

Do not use `latest` for an audited rollback and do not manually copy a model file into the image.

## Phase 12 local test evidence

The rollback mechanism was exercised with the existing registry interfaces using a deterministic controlled registry test double because the supplied environment does not have MLflow installed and cannot download packages from the network. The test still calls the same production promotion and rollback functions, registry alias operations, safety checks, audit recording, and serving-version resolution used by the live MLflow path.

The demonstrated sequence is:

```text
Production v1
    -> successful v2 promotion through existing Phase 7 registry code
    -> production alias = v2
    -> rollback target = v1
    -> production alias = v1
    -> serving loader resolves models:/house-price-regressor/1
```

The test also verifies that the local v2 mirror remains byte-for-byte unchanged during rollback, proving that rollback does not manually copy v1 model bytes.

Actual live MLflow/GitHub/GHCR/deployment evidence is not fabricated. A real deployment should additionally capture the MLflow alias/version before and after rollback and the API `/health` response from the restarted container.

## Evidence status

The local Phase 11 CI-equivalent checks and Phase 12 rollback tests are recorded under `reports/phase12_evidence/`. A real GitHub PR history, published GHCR image, GitHub Release, and live MLflow rollback require the repository to be pushed to GitHub and connected to the real registry/deployment environment.
