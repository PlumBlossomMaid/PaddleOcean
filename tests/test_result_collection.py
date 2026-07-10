"""Unit tests for the per-stage result collection (F9 root-cause fix).

These exercise the collection in isolation — no Trainer, no Paddle model — so
they pin down the reduction/stage-separation contract directly.
"""

import math

from ocean.trainer.connectors.logger_connector.result import (
    _Metadata,
    _ResultCollection,
    _ResultMetric,
    extract_batch_size,
)


def test_batch_size_weighted_mean():
    """Epoch mean must weight by batch size, not average per-step means."""
    rc = _ResultCollection(training=True)
    # two steps: value 1.0 over 10 samples, value 4.0 over 30 samples
    rc.batch_size = 10
    rc.log("training_step", "loss", 1.0, on_step=False, on_epoch=True)
    rc.batch_size = 30
    rc.log("training_step", "loss", 4.0, on_step=False, on_epoch=True)
    # weighted: (1*10 + 4*30) / 40 = 130/40 = 3.25  (unweighted would be 2.5)
    m = rc.metrics(on_step=False)
    assert math.isclose(m["log"]["loss"], 3.25), m["log"]


def test_sum_min_max_reductions():
    rc = _ResultCollection(training=True)
    for v in (2.0, 5.0, 3.0):
        rc.batch_size = 1
        rc.log("training_step", "s", v, on_step=False, on_epoch=True, reduce_fx="sum")
        rc.log("training_step", "mx", v, on_step=False, on_epoch=True, reduce_fx="max")
        rc.log("training_step", "mn", v, on_step=False, on_epoch=True, reduce_fx="min")
    m = rc.metrics(on_step=False)["log"]
    assert m["s"] == 10.0
    assert m["mx"] == 5.0
    assert m["mn"] == 2.0


def test_forked_step_and_epoch_names():
    """on_step and on_epoch together -> name_step / name_epoch."""
    rc = _ResultCollection(training=True)
    rc.batch_size = 1
    rc.log("training_step", "acc", 0.5, on_step=True, on_epoch=True)
    step_log = rc.metrics(on_step=True)["log"]
    epoch_log = rc.metrics(on_step=False)["log"]
    assert "acc_step" in step_log
    assert "acc_epoch" in epoch_log
    assert "acc" not in step_log  # forked, so bare name is not used for logging


def test_non_forked_name_unchanged():
    rc = _ResultCollection(training=True)
    rc.batch_size = 1
    rc.log("training_step", "loss", 1.0, on_step=False, on_epoch=True)
    assert "loss" in rc.metrics(on_step=False)["log"]


def test_stage_separation_val_does_not_clear_train():
    """The heart of F9: a separate eval collection cannot touch train accumulation."""
    train = _ResultCollection(training=True)
    val = _ResultCollection(training=False)

    # first 3 training batches
    for v in (1.0, 1.0, 1.0):
        train.batch_size = 1
        train.log("training_step", "loss", v, on_step=False, on_epoch=True)

    # mid-epoch validation runs on its OWN collection and resets itself
    for v in (9.0, 9.0):
        val.batch_size = 1
        val.log("validation_step", "loss", v, on_step=False, on_epoch=True)
    assert math.isclose(val.metrics(on_step=False)["log"]["loss"], 9.0)
    val.reset()  # eval loop resets its own collection after the pass

    # remaining training batches
    for v in (1.0, 1.0):
        train.batch_size = 1
        train.log("training_step", "loss", v, on_step=False, on_epoch=True)

    # epoch train mean must average ALL 5 batches (=1.0), unaffected by val
    assert math.isclose(train.metrics(on_step=False)["log"]["loss"], 1.0)


def test_reset_by_fx():
    rc = _ResultCollection(training=False)
    rc.batch_size = 1
    rc.log("validation_step", "a", 1.0, on_step=False, on_epoch=True)
    rc.log("test_step", "b", 2.0, on_step=False, on_epoch=True)
    rc.reset(fx="validation_step")
    m = rc.metrics(on_step=False)["log"]
    # 'a' was reset (has_reset -> excluded from valid_items); 'b' remains
    assert "a" not in m
    assert "b" in m


def test_callback_metrics_populated_during_training():
    rc = _ResultCollection(training=True)
    rc.batch_size = 1
    rc.log("training_step", "loss", 2.0, on_step=True, on_epoch=False)
    cb = rc.metrics(on_step=True)["callback"]
    assert cb.get("loss") == 2.0


def test_prog_bar_routing():
    rc = _ResultCollection(training=True)
    rc.batch_size = 1
    rc.log("training_step", "loss", 2.0, prog_bar=True, on_step=False, on_epoch=True)
    rc.log("training_step", "hidden", 3.0, prog_bar=False, on_step=False, on_epoch=True)
    pbar = rc.metrics(on_step=False)["pbar"]
    assert "loss" in pbar
    assert "hidden" not in pbar


def test_logger_false_excluded_from_log():
    rc = _ResultCollection(training=True)
    rc.batch_size = 1
    rc.log("training_step", "loss", 2.0, logger=False, on_step=False, on_epoch=True)
    assert "loss" not in rc.metrics(on_step=False)["log"]


def test_metadata_requires_step_or_epoch():
    try:
        _Metadata(fx="training_step", name="x", on_step=False, on_epoch=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError for on_step=False, on_epoch=False")


def test_metric_object_path():
    """paddlemetrics-like object: collection delegates to its compute()."""

    class _FakeMetric:
        def __init__(self):
            self._forward_cache = 0.7
            self._acc = []

        def update(self, x):
            self._acc.append(x)

        def compute(self):
            return sum(self._acc) / len(self._acc)

        def reset(self):
            self._acc = []

    rc = _ResultCollection(training=False)
    rc.batch_size = 1
    fm = _FakeMetric()
    fm.update(0.6)
    fm.update(0.8)
    rc.log("validation_step", "acc", fm, on_step=False, on_epoch=True)
    assert math.isclose(rc.metrics(on_step=False)["log"]["acc"], 0.7)


def test_extract_batch_size_helper():
    class _T:
        shape = (16, 3)

    assert extract_batch_size(_T()) == 16
    assert extract_batch_size({"x": _T()}) == 16
    assert extract_batch_size([_T(), _T()]) == 16
    assert extract_batch_size(None) == 1
    assert extract_batch_size(42) == 1


def test_result_metric_reset_clears_value():
    meta = _Metadata(fx="training_step", name="loss", on_step=False, on_epoch=True)
    rm = _ResultMetric(meta, is_tensor=True)
    rm.update(5.0, 1)
    assert rm.compute() == 5.0
    rm.reset()
    assert rm.has_reset is True
    assert rm.compute() == 0.0
