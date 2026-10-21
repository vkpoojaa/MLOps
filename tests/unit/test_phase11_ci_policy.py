from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str):
    # BaseLoader preserves GitHub Actions' `on` key as a string.
    path = ROOT / ".github" / "workflows" / name
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_pr_workflow_runs_required_checks_and_fails_on_test_failure():
    workflow = _workflow("pr-checks.yml")
    triggers = workflow["on"]
    assert "pull_request" in triggers
    assert "main" in triggers["push"]["branches"]
    steps = workflow["jobs"]["checks"]["steps"]
    run_text = "\n".join(step.get("run", "") for step in steps)
    assert "pip install -r requirements-dev.txt" in run_text
    assert "validate-config" in run_text
    assert "tests/unit" in run_text
    assert "tests/integration" in run_text
    assert "ci_pipeline_smoke.py" in run_text
    # There is no `continue-on-error: true`; pytest exit status fails the job.
    assert all(step.get("continue-on-error") != "true" for step in steps)


def test_release_workflow_is_commit_traceable_and_does_not_bypass_model_gates():
    workflow = _workflow("release.yml")
    triggers = workflow["on"]
    assert "main" in triggers["push"]["branches"]
    assert workflow["jobs"]["release-image"]["needs"] == "release-checks"
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "GIT_COMMIT=${{ github.sha }}" in text
    assert "org.opencontainers.image.revision=${{ github.sha }}" in text
    assert "release-manifest.json" in text
    assert "Only the ML pipeline may promote a model after Data Gate and Model Gate pass" in text
    assert "orchestrate" not in text
    assert "register_and_promote" not in text
    assert "kubectl apply" not in text


def test_lightweight_pipeline_smoke_has_no_training_or_registry_promotion_calls():
    script = (ROOT / "scripts" / "ci_pipeline_smoke.py").read_text(encoding="utf-8")
    assert "ingestion_task" in script
    assert "validation_task" in script
    assert "data_gate_task" in script
    assert "does not train models" in script
    assert "register_and_promote" not in script
