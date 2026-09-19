"""Shared connection settings for the standalone loader scripts.

The Kedro pipeline reads its credentials from conf/local/credentials.yml;
these loaders run *before* the pipeline exists (they build the source
systems), so they read plain environment variables with local defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5433)),
    "dbname": os.getenv("PG_DB", "retail"),
    "user": os.getenv("PG_USER", "retail"),
    "password": os.getenv("PG_PASSWORD", "retail"),
}

MONGO_URI = os.getenv(
    "MONGO_URI", "mongodb://retail:retail@localhost:27017/?authSource=admin"
)
MONGO_DB = os.getenv("MONGO_DB", "retail_events")

# Fixed seed so the synthetic clickstream and the injected orphan records are
# byte-for-byte reproducible. A portfolio project that cannot be re-run to the
# same numbers is not reproducible, and reproducibility is the whole point of
# the pipeline layer.
SEED = 42
