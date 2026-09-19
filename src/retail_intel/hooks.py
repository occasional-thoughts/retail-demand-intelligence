"""Project hooks.

The Spark session is created once per Kedro run rather than once per node.
Spinning up a JVM per node would dominate the runtime of the scale layer.
"""
from __future__ import annotations

import logging
import os
import sys

from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession

log = logging.getLogger(__name__)


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        # Spark launches its Python workers from whatever `python3` PATH
        # resolves to, which on a machine with a base conda install is almost
        # never the interpreter running the driver. A minor-version mismatch
        # fails every task with an opaque serialisation error, so both ends are
        # pinned to this interpreter explicitly.
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        parameters = context.config_loader["spark"]
        conf = SparkConf().setAll(parameters.items())

        session = (
            SparkSession.builder.appName(context.project_path.name)
            .master("local[*]")
            .config(conf=conf)
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("WARN")
        log.info("Spark session started (version %s)", session.version)
