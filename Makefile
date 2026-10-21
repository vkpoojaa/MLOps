.PHONY: setup validate-config check-contract train test monitoring-up monitor-batches

setup:
	python -m pip install -r requirements-dev.txt

validate-config:
	PYTHONPATH=src python -m mlops_pipeline validate-config

check-contract:
	PYTHONPATH=src python -m mlops_pipeline check-contract

train:
	PYTHONPATH=src python -m mlops_pipeline train

test:
	python -m pytest

monitoring-up:
	docker compose -f monitoring/docker-compose.yml up --build

monitor-batches:
	PYTHONPATH=src python -m mlops_pipeline monitor-batches
