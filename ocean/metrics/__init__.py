"""Metrics integration for Ocean.

Re-exports core metric types from ``paddlemetrics``.  Domain-specific
metrics (``Accuracy``, ``Precision``, ``MeanSquaredError``, …) should
be imported directly from ``paddlemetrics``, following the same pattern
Lightning uses with ``torchmetrics``::

    from paddlemetrics import Accuracy

    acc = Accuracy(task="multiclass", num_classes=10)
    acc(preds, target)
    self.log("val_acc", acc, on_epoch=True)

Core types exposed here for convenience:
    - ``Metric`` / ``CompositionalMetric`` — base classes
    - ``MetricCollection`` — group multiple metrics
    - Aggregation helpers (``MeanMetric``, ``SumMetric``, …)
"""

from __future__ import annotations

from paddlemetrics import *  # noqa: F403 — re-export what paddlemetrics exposes in __all__
from paddlemetrics.aggregation import (  # noqa: F401
    CatMetric,
    MaxMetric,
    MeanMetric,
    MinMetric,
    RunningMean,
    RunningSum,
    SumMetric,
)
from paddlemetrics.collections import MetricCollection  # noqa: F401
from paddlemetrics.metric import CompositionalMetric, Metric  # noqa: F401

__all__ = [
    "Metric",
    "CompositionalMetric",
    "MetricCollection",
    "CatMetric",
    "MaxMetric",
    "MeanMetric",
    "MinMetric",
    "RunningMean",
    "RunningSum",
    "SumMetric",
]
