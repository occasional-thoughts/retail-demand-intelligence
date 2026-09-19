"""Pipeline registry.

Named sub-pipelines exist so that a failing layer can be re-run on its own:
`kedro run --pipeline=integrity` after fixing a source-system defect, rather
than re-running ingestion, Spark and both models to get back to the failure.
"""
from __future__ import annotations

from kedro.pipeline import Pipeline

from retail_intel.pipelines import (
    churn,
    features,
    forecasting,
    ingestion,
    integrity,
    scale,
    serving,
)


def register_pipelines() -> dict[str, Pipeline]:
    ingestion_pipeline = ingestion.create_pipeline()
    integrity_pipeline = integrity.create_pipeline()
    scale_pipeline = scale.create_pipeline()
    features_pipeline = features.create_pipeline()
    forecasting_pipeline = forecasting.create_pipeline()
    churn_pipeline = churn.create_pipeline()
    serving_pipeline = serving.create_pipeline()

    return {
        "ingestion": ingestion_pipeline,
        "integrity": integrity_pipeline,
        "scale": scale_pipeline,
        "features": features_pipeline,
        "forecasting": forecasting_pipeline,
        "churn": churn_pipeline,
        "serving": serving_pipeline,
        # everything up to and including the validated silver layer
        "data_engineering": ingestion_pipeline + integrity_pipeline + scale_pipeline,
        "modelling": features_pipeline + forecasting_pipeline + churn_pipeline,
        "__default__": (
            ingestion_pipeline
            + integrity_pipeline
            + scale_pipeline
            + features_pipeline
            + forecasting_pipeline
            + churn_pipeline
            + serving_pipeline
        ),
    }
