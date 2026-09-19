# Runbook. Every step of the platform, in the order it must run.
# `make all` takes a clean machine to two dashboard-ready extract sets.

PY := /opt/anaconda3/envs/retail/bin/python
KEDRO := /opt/anaconda3/envs/retail/bin/kedro
DOCKER := /Applications/Docker.app/Contents/Resources/bin/docker

.PHONY: help up down sources load-sql load-mongo pipeline test lint clean all status

help:
	@echo "make up         start Postgres + MongoDB"
	@echo "make sources    build both source systems from data/raw/"
	@echo "make pipeline   run the full Kedro pipeline"
	@echo "make dashboards generate the Tableau workbooks and the web dashboards"
	@echo "make knime-figures  reference figures to verify the KNIME workflow"
	@echo "make test       run the test suite"
	@echo "make all        up -> sources -> pipeline"
	@echo "make status     show what is running and what has been built"
	@echo "make down       stop the containers (data survives)"
	@echo "make clean      delete all derived data (raw and source systems survive)"

up:
	$(DOCKER) compose up -d
	@echo "waiting for health checks..."
	@until [ "$$($(DOCKER) inspect -f '{{.State.Health.Status}}' retail_pg)" = "healthy" ]; do sleep 2; done
	@until [ "$$($(DOCKER) inspect -f '{{.State.Health.Status}}' retail_mongo)" = "healthy" ]; do sleep 2; done
	@echo "postgres and mongodb are up"

down:
	$(DOCKER) compose down

load-sql:
	cd scripts && $(PY) load_postgres.py

load-mongo:
	cd scripts && $(PY) generate_events.py

sources: load-sql load-mongo

pipeline:
	$(KEDRO) run

# Individual layers, for re-running one stage after a fix rather than all of them
ingestion:   ; $(KEDRO) run --pipeline=ingestion
integrity:   ; $(KEDRO) run --pipeline=integrity
scale:       ; $(KEDRO) run --pipeline=scale
features:    ; $(KEDRO) run --pipeline=features
forecasting: ; $(KEDRO) run --pipeline=forecasting
churn:       ; $(KEDRO) run --pipeline=churn
serving:     ; $(KEDRO) run --pipeline=serving

dashboards:
	$(PY) scripts/build_tableau_workbooks.py
	$(PY) scripts/build_web_dashboards.py

# Reference figures for verifying the KNIME workflow. Deliberately not part of
# the pipeline -- see docs/knime_workflow.md.
knime-figures:
	$(PY) scripts/knime_reference_figures.py

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check src/ tests/ scripts/ || true

status:
	@echo "== containers =="
	@$(DOCKER) compose ps 2>/dev/null || echo "  docker not running"
	@echo "== reporting extracts =="
	@ls -la data/08_reporting/*.csv 2>/dev/null || echo "  none built yet"
	@echo "== quarantine =="
	@ls -la data/quarantine/ 2>/dev/null || echo "  none"

clean:
	rm -rf data/01_bronze/* data/02_silver/* data/03_gold/* data/08_reporting/*
	# expected_orphans.json is ground truth written by generate_events.py, not
	# by the pipeline. Deleting it here would silently disable the integration
	# tests that verify the integrity layer against it.
	find data/quarantine -type f ! -name expected_orphans.json ! -name .gitkeep -delete
	rm -rf spark-warehouse metastore_db derby.log
	@echo "derived data removed; raw data, source systems and orphan manifest untouched"

all: up sources pipeline dashboards
	@echo ""
	@echo "platform built."
	@echo "  extracts:   data/08_reporting/*.csv"
	@echo "  dashboards: tableau/*.twb"
