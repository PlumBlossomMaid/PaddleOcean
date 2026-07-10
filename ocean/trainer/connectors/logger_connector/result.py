"""Result collection: per-stage metric storage with correct reduction.

This module gives each running stage (training vs evaluation) its own metric
store so a mid-epoch validation pass can never clobber the training metrics
accumulated earlier in the same epoch. Storage is keyed by ``fx.name`` and each
value is reduced with a batch-size-weighted mean (or min/max/sum), so the epoch
value matches the reference implementation's behavior rather than an unweighted
average of per-step means.

Design notes (Paddle specifics, intentionally different from the reference):
- Values are stored as Python floats. The reference keeps tensors so it can sync
  across ranks at ``compute()`` time; here ``sync_dist`` is already applied in the
  logging path (``Trainer._log_metric``) before values reach this collection, so
  we only need scalar accumulation. This keeps distributed behavior unchanged.
- ``_ResultMetric`` is a plain object (the reference subclasses
  ``torchmetrics.Metric`` for state sync, which has no direct Paddle equivalent).
- ``paddlemetrics.Metric`` objects are supported by delegating to their own
  ``compute()``, mirroring the metric-object path already present in the Trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_VALID_REDUCE = ("mean", "sum", "min", "max")


def _normalize_reduce_fx(reduce_fx: Any) -> str:
    """Canonicalize a reduce function specifier to one of _VALID_REDUCE."""
    if callable(reduce_fx):
        reduce_fx = getattr(reduce_fx, "__name__", "mean")
    reduce_fx = str(reduce_fx).lower()
    if reduce_fx == "avg":
        reduce_fx = "mean"
    if reduce_fx not in _VALID_REDUCE:
        raise ValueError(
            f"Only reduce_fx in {_VALID_REDUCE} are supported. For a custom reduction, "
            f"log a paddlemetrics.Metric instance instead. Found: {reduce_fx!r}"
        )
    return reduce_fx


def extract_batch_size(batch: Any) -> int:
    """Best-effort batch size from a tensor / sequence / mapping batch.

    Falls back to 1 when the structure has no discernible leading dimension.
    """
    if batch is None:
        return 1
    # tensor-like: use first dim
    shape = getattr(batch, "shape", None)
    if shape is not None:
        try:
            return int(shape[0]) if len(shape) > 0 else 1
        except (TypeError, IndexError):
            return 1
    if isinstance(batch, dict):
        for v in batch.values():
            bs = extract_batch_size(v)
            if bs != 1:
                return bs
        return 1
    if isinstance(batch, (list, tuple)) and batch:
        return extract_batch_size(batch[0])
    return 1


@dataclass
class _Metadata:
    """Static description of one logged metric (how to reduce / where to route)."""

    fx: str
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: str = "mean"
    add_dataloader_idx: bool = True
    dataloader_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.on_step and not self.on_epoch:
            raise ValueError("log(on_step=False, on_epoch=False) is not useful.")
        self.reduce_fx = _normalize_reduce_fx(self.reduce_fx)

    @property
    def is_mean_reduction(self) -> bool:
        return self.reduce_fx == "mean"

    @property
    def is_sum_reduction(self) -> bool:
        return self.reduce_fx == "sum"

    @property
    def is_max_reduction(self) -> bool:
        return self.reduce_fx == "max"

    @property
    def is_min_reduction(self) -> bool:
        return self.reduce_fx == "min"

    @property
    def forked(self) -> bool:
        """True when a metric is logged both on step and on epoch."""
        return self.on_step and self.on_epoch

    def forked_name(self, on_step: bool) -> str:
        """Suffix ``_step`` / ``_epoch`` only when the metric is forked."""
        if self.forked:
            return f"{self.name}_{'step' if on_step else 'epoch'}"
        return self.name


class _ResultMetric:
    """Accumulates one metric's value under a given reduction.

    For scalar values, mean reduction is batch-size weighted:
    ``sum(value_i * bs_i) / sum(bs_i)``. For ``paddlemetrics.Metric`` values the
    object owns its own state and we delegate to its ``compute()``.
    """

    def __init__(self, meta: _Metadata, is_tensor: bool) -> None:
        self.meta = meta
        self.is_tensor = is_tensor
        self.has_reset = False
        self._forward_cache: Optional[float] = None
        self._computed: Optional[float] = None
        self._metric_obj: Any = None  # paddlemetrics.Metric when not is_tensor
        self._reset_state()

    def _reset_state(self) -> None:
        if self.meta.is_min_reduction:
            self.value: float = float("inf")
        elif self.meta.is_max_reduction:
            self.value = float("-inf")
        else:
            self.value = 0.0
        self.cumulated_batch_size: int = 0

    def update(self, value: Any, batch_size: int) -> None:
        self._computed = None
        if not self.is_tensor:
            # paddlemetrics.Metric object: it manages its own accumulation.
            self._metric_obj = value
            fc = getattr(value, "_forward_cache", None)
            if fc is not None and hasattr(fc, "item"):
                try:
                    fc = fc.item()
                except (ValueError, RuntimeError):
                    fc = None
            self._forward_cache = fc
            return

        value = float(value)
        self._forward_cache = value

        if not self.meta.on_epoch:
            # step-only metric: keep just the latest value
            self.value = value
            return

        if self.meta.is_mean_reduction:
            self.value += value * batch_size
            self.cumulated_batch_size += batch_size
        elif self.meta.is_sum_reduction:
            self.value += value
        elif self.meta.is_max_reduction:
            self.value = max(self.value, value)
        elif self.meta.is_min_reduction:
            self.value = min(self.value, value)

    def compute(self) -> float:
        if not self.is_tensor:
            result = self._metric_obj.compute()
            if hasattr(result, "item"):
                result = result.item()
            return float(result)
        if self.meta.is_mean_reduction:
            return self.value / self.cumulated_batch_size if self.cumulated_batch_size else 0.0
        return self.value

    def reset(self) -> None:
        self._reset_state()
        self._forward_cache = None
        self._computed = None
        if not self.is_tensor and self._metric_obj is not None and hasattr(self._metric_obj, "reset"):
            self._metric_obj.reset()
        self.has_reset = True


class _ResultCollection(dict):
    """Dict of ``fx.name -> _ResultMetric`` for a single running stage.

    ``training`` distinguishes the training collection from the eval collection;
    keeping them as separate instances is what makes mid-epoch validation
    incapable of clearing training accumulation.
    """

    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, training: bool, fork_names: bool = True) -> None:
        super().__init__()
        self.training = training
        # When True, a metric logged both on_step and on_epoch is exposed under
        # ``name_step`` / ``name_epoch`` (reference behavior). Ocean's public
        # ``Model.log`` contract stores under the bare ``name`` regardless, so the
        # Trainer constructs collections with fork_names=False.
        self._fork = fork_names
        self.batch: Any = None
        self.batch_size: Optional[int] = None
        self.dataloader_idx: Optional[int] = None

    def _extract_batch_size(self, meta: _Metadata, batch_size: Optional[int]) -> int:
        if batch_size is not None:
            return batch_size
        if self.batch_size is not None:
            return self.batch_size
        bs = 1
        if self.batch is not None and meta.on_epoch and meta.is_mean_reduction:
            bs = extract_batch_size(self.batch)
            self.batch_size = bs
        return bs

    def log(
        self,
        fx: str,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: Any = "mean",
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        is_tensor: Optional[bool] = None,
    ) -> None:
        """Register (once) and update the metric identified by ``fx.name``."""
        key = f"{fx}.{name}"
        if add_dataloader_idx and self.dataloader_idx is not None:
            key += f".{self.dataloader_idx}"
            fx += f".{self.dataloader_idx}"

        if key not in self:
            meta = _Metadata(
                fx=fx,
                name=name,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
                dataloader_idx=self.dataloader_idx,
            )
            # a value is a metric object iff it exposes update()+compute() and isn't a scalar
            if is_tensor is None:
                is_tensor = not (hasattr(value, "compute") and hasattr(value, "update"))
            self[key] = _ResultMetric(meta, is_tensor)

        result_metric = self[key]
        bs = self._extract_batch_size(result_metric.meta, batch_size)
        result_metric.update(value, bs)
        result_metric.has_reset = False

    def valid_items(self):
        """Iterate metrics that are live for the current dataloader."""
        return ((k, v) for k, v in self.items() if not v.has_reset and self.dataloader_idx == v.meta.dataloader_idx)

    def _forked_name(self, result_metric: _ResultMetric, on_step: bool) -> tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step) if self._fork else name
        dl_idx = result_metric.meta.dataloader_idx
        if result_metric.meta.add_dataloader_idx and dl_idx is not None:
            suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += suffix
            forked_name += suffix
        return name, forked_name

    @staticmethod
    def _get_cache(result_metric: _ResultMetric, on_step: bool) -> Optional[float]:
        if on_step and result_metric.meta.on_step:
            return result_metric._forward_cache
        if not on_step and result_metric.meta.on_epoch:
            if result_metric._computed is None:
                result_metric._computed = result_metric.compute()
            return result_metric._computed
        return None

    def metrics(self, on_step: bool) -> dict[str, dict]:
        """Return {callback, log, pbar} dicts for the requested step/epoch view."""
        metrics: dict[str, dict] = {"callback": {}, "log": {}, "pbar": {}}

        for _, result_metric in self.valid_items():
            value = self._get_cache(result_metric, on_step)
            if value is None:
                continue

            name, forked_name = self._forked_name(result_metric, on_step)

            if result_metric.meta.logger:
                metrics["log"][forked_name] = value

            # callback metrics: available during training, or on epoch-end views
            if self.training or (result_metric.meta.on_epoch and not on_step):
                metrics["callback"][name] = value
                metrics["callback"][forked_name] = value

            if result_metric.meta.prog_bar:
                metrics["pbar"][forked_name] = value

        return metrics

    def reset(self, fx: Optional[str] = None) -> None:
        """Reset all metrics, or only those registered under ``fx``."""
        for item in self.values():
            if fx is None or fx == item.meta.fx:
                item.reset()
        self.batch = None
        self.batch_size = None
