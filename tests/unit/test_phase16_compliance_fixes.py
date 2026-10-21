from pathlib import Path


def test_phase14_bootstrap_controls_are_not_hardcoded_in_drift_module():
    source = Path("src/mlops_pipeline/monitoring/drift.py").read_text(encoding="utf-8")
    assert "BOOTSTRAP_DRIFT_RUNS" not in source
    assert "BOOTSTRAP_PERFORMANCE_RUNS" not in source


def test_required_final_docs_exist():
    for path in [
        "docs/DESIGN_NOTE.md",
        "docs/WORKFLOW_GRAPH.md",
        "docs/STAGE_RERUN_RUNBOOK.md",
        "docs/AI_USE_DISCLOSURE.md",
        "docs/REFLECTION.md",
    ]:
        assert Path(path).is_file(), path


def test_operational_thresholds_are_configured():
    from mlops_pipeline.config import load_config
    cfg = load_config(check_paths=True).config.monitoring.operational
    assert cfg.latency_p95_ms_max is not None
    assert cfg.error_rate_max is not None
