"""Forecasting pipeline: backtest, refit, forecast."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.walk_forward_backtest,
                inputs=["gold_demand_features", "params:forecasting"],
                outputs=["demand_backtest", "demand_metrics"],
                name="walk_forward_backtest",
                tags=["model", "demand", "validation"],
            ),
            node(
                func=nodes.train_final_model,
                inputs=["gold_demand_features", "params:forecasting"],
                outputs="demand_model",
                name="train_demand_model",
                tags=["model", "demand"],
            ),
            node(
                func=nodes.generate_forecast,
                inputs=["demand_model", "gold_demand_features", "demand_backtest"],
                outputs="demand_forecast",
                name="generate_demand_forecast",
                tags=["model", "demand"],
            ),
        ]
    )
