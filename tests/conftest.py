"""Shared fixtures.

The tests deliberately do not touch Postgres, MongoDB or Spark. Nodes are pure
functions of their inputs -- that is the reason the connections live in the
catalog -- so the logic can be tested on small synthetic frames in
milliseconds, and CI does not need a database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def integrity_params() -> dict:
    return {
        "checks": {
            "reviews": {"max_orphan_rate": 0.10, "min_orphan_rate": 0.01},
            "sessions": {"max_orphan_rate": 0.15, "min_orphan_rate": 0.02},
        }
    }


@pytest.fixture
def valid_order_ids() -> pd.DataFrame:
    return pd.DataFrame({"order_id": [f"order_{i:03d}" for i in range(100)]})


@pytest.fixture
def reviews_with_orphans() -> pd.DataFrame:
    """95 resolvable reviews and 5 pointing at orders that do not exist."""
    good = [f"order_{i:03d}" for i in range(95)]
    bad = [f"ghost_{i:03d}" for i in range(5)]
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(100)],
            "order_id": good + bad,
            "review_score": np.random.default_rng(0).integers(1, 6, 100),
        }
    )


@pytest.fixture
def weekly_demand() -> pd.DataFrame:
    """Two categories, three years of weekly data with a clear seasonal shape."""
    # Anchored the same way production does it -- `to_period(...).start_time`,
    # not `date_range`. A fixture that uses a different week anchor to the code
    # it exercises tests a pipeline that does not exist.
    weeks = pd.period_range("2016-01-04", periods=156, freq="W-MON").start_time
    rng = np.random.default_rng(42)
    frames = []
    for offset, category in enumerate(["toys", "electronics"]):
        seasonal = 40 + 15 * np.sin(2 * np.pi * np.arange(156) / 52 + offset)
        noise = rng.normal(0, 3, 156)
        frames.append(
            pd.DataFrame(
                {
                    "category": category,
                    "week": weeks,
                    "units": np.clip(seasonal + noise, 0, None).round(),
                    "revenue": np.clip(seasonal + noise, 0, None).round() * 50,
                    "orders": np.clip(seasonal + noise, 0, None).round(),
                    "avg_price": 50.0,
                    "avg_freight": 8.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
