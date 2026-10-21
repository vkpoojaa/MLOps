from pathlib import Path


def test_dockerfile_has_traceable_image_labels_and_nonroot_runtime():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12.3-slim-bookworm" in dockerfile
    assert "org.opencontainers.image.version" in dockerfile
    assert "org.opencontainers.image.revision" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'CMD ["python", "-m", "mlops_pipeline.serving"]' in dockerfile
    assert "GIT_COMMIT=${GIT_COMMIT}" in dockerfile
    assert 'test "${GIT_COMMIT}" != "unknown"' in dockerfile


def test_container_build_does_not_copy_local_model_or_registry_state():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert "COPY artifacts/" not in dockerfile
    assert "COPY data/" not in dockerfile
    assert "COPY mlflow.db" not in dockerfile
    assert "artifacts/" in dockerignore
    assert "mlflow.db" in dockerignore
    assert "mlruns/" in dockerignore


def test_build_script_requires_real_git_commit():
    script = Path("scripts/build-image.sh").read_text(encoding="utf-8")
    assert "git rev-parse HEAD" in script
    assert "source ZIP has no authoritative commit" in script
    assert '--build-arg GIT_COMMIT="$COMMIT"' in script
    assert 'TAG="${IMAGE}:${VERSION}-${SHORT_COMMIT}"' in script


def test_serving_lock_is_pinned_and_excludes_orchestration_and_test_dependencies():
    lock = Path("requirements-serving.lock").read_text(encoding="utf-8")
    assert "fastapi==" in lock
    assert "uvicorn==" in lock
    assert "mlflow==" in lock
    assert "scikit-learn==" in lock
    dependency_lines = [line.strip().lower() for line in lock.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line.startswith("prefect") for line in dependency_lines)
    assert not any(line.startswith("pytest") for line in dependency_lines)


def test_health_response_reports_image_version_and_git_revision(monkeypatch):
    from fastapi.testclient import TestClient
    from mlops_pipeline.config.loader import load_config
    from mlops_pipeline.serving.api import create_app

    class Model:
        def predict(self, frame):
            return [123.0] * len(frame)

    monkeypatch.setenv("APP_VERSION", "10.0.0-test")
    monkeypatch.setenv("GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("IMAGE_NAME", "test/house-price-api")
    app = create_app(
        model_loader=lambda settings: (
            Model(),
            {
                "registered_model_name": "house-price-regressor",
                "model_version": "42",
                "production_alias": "production",
                "model_gate_status": "PASS",
            },
        ),
        settings_loader=load_config,
    )
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["build"] == {
        "version": "10.0.0-test",
        "git_commit": "a" * 40,
        "image": "test/house-price-api",
    }
