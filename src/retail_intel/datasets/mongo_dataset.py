"""A Kedro dataset for reading MongoDB collections.

kedro-datasets ships connectors for SQL and for file formats, but not for
MongoDB, so the document store gets a first-class catalog entry here instead
of being smuggled into a node as a hard-coded connection.

That matters for more than tidiness: with the connection in the catalog, the
nodes stay pure functions of their inputs, which is what makes them testable
without a live database.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from kedro.io import AbstractDataset
from pymongo import MongoClient


class MongoCollectionDataset(AbstractDataset[None, pd.DataFrame]):
    """Read a MongoDB collection into a pandas DataFrame.

    Args:
        collection: collection name within the database.
        credentials: mapping with ``uri`` and ``database``.
        query: optional MongoDB filter document.
        projection: optional field projection, to avoid pulling large
            sub-documents that the node does not need.
        limit: optional row cap, useful for smoke tests.
    """

    def __init__(
        self,
        collection: str,
        credentials: dict[str, str],
        query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> None:
        self._collection = collection
        self._uri = credentials["uri"]
        self._database = credentials["database"]
        self._query = query or {}
        self._projection = projection
        self._limit = limit

    def _client(self) -> MongoClient:
        return MongoClient(self._uri)

    def load(self) -> pd.DataFrame:
        with self._client() as client:
            cursor = client[self._database][self._collection].find(
                self._query, self._projection
            )
            if self._limit:
                cursor = cursor.limit(self._limit)
            frame = pd.DataFrame(list(cursor))
        if "_id" in frame.columns:
            frame = frame.drop(columns=["_id"])
        return frame

    def save(self, data: pd.DataFrame) -> None:
        raise NotImplementedError(
            "The event store is a source system; the pipeline reads from it but "
            "never writes back to it."
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "collection": self._collection,
            "database": self._database,
            "query": self._query,
            "limit": self._limit,
        }


class MongoAggregationDataset(MongoCollectionDataset):
    """Run an aggregation pipeline server-side and return the result.

    Session documents hold a nested array of events. Unwinding and summarising
    that array inside MongoDB moves the work to where the data already is,
    instead of pulling ~2M nested events over the wire to flatten them in
    pandas.
    """

    def __init__(
        self,
        collection: str,
        credentials: dict[str, str],
        pipeline: list[dict[str, Any]],
        allow_disk_use: bool = True,
    ) -> None:
        super().__init__(collection=collection, credentials=credentials)
        self._pipeline = pipeline
        self._allow_disk_use = allow_disk_use

    def load(self) -> pd.DataFrame:
        with self._client() as client:
            cursor = client[self._database][self._collection].aggregate(
                self._pipeline, allowDiskUse=self._allow_disk_use
            )
            frame = pd.DataFrame(list(cursor))
        if "_id" in frame.columns and frame["_id"].apply(lambda v: isinstance(v, dict)).any():
            # unpack composite group keys into real columns
            keys = pd.json_normalize(frame["_id"])
            frame = pd.concat([keys, frame.drop(columns=["_id"])], axis=1)
        elif "_id" in frame.columns:
            frame = frame.drop(columns=["_id"])
        return frame

    def _describe(self) -> dict[str, Any]:
        return {
            "collection": self._collection,
            "database": self._database,
            "stages": len(self._pipeline),
        }
