"""Churn pipeline: ablation first, then the model the ablation justifies."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.run_sentiment_ablation,
                inputs=["gold_rfm_segments", "params:churn"],
                outputs="churn_ablation",
                name="run_sentiment_ablation",
                tags=["model", "churn", "validation"],
            ),
            node(
                func=nodes.train_churn_model,
                inputs=["gold_rfm_segments", "churn_ablation", "params:churn"],
                outputs="churn_model",
                name="train_churn_model",
                tags=["model", "churn"],
            ),
            node(
                func=nodes.score_customers,
                inputs=["churn_model", "gold_rfm_segments"],
                outputs="churn_scores",
                name="score_customer_churn_risk",
                tags=["model", "churn"],
            ),
        ]
    )
