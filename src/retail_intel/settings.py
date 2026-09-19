"""Kedro project settings."""
from kedro.config import OmegaConfigLoader

from retail_intel.hooks import SparkHooks

# One Spark session per run, created when the Kedro context is built.
HOOKS = (SparkHooks(),)

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    },
}
